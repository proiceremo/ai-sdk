package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	llm "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/oauthx"
	"github.com/gorilla/websocket"
	"golang.org/x/oauth2"
)

const (
	defaultCodexBaseURL = "https://chatgpt.com/backend-api"
	codexJWTClaimPath   = "https://api.openai.com/auth"
	codexWebSocketBeta  = "responses_websockets=2026-02-06"
	codexSSEBeta        = "responses=experimental"
)

type CodexClient struct {
	httpClient  *http.Client
	baseURL     string
	tokenSource oauth2.TokenSource
	headers     map[string]string
	originator  string
}

func NewCodexClient(ctx context.Context, cfg llm.ProviderConfig, source oauth2.TokenSource) (*CodexClient, error) {
	if source == nil {
		return nil, fmt.Errorf("openai codex provider requires OAuth credentials; run `go run ./cmd/eval oauth-login --provider openai-codex`")
	}
	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = defaultCodexBaseURL
	}
	client := &CodexClient{
		httpClient:  &http.Client{},
		baseURL:     baseURL,
		tokenSource: source,
		headers:     map[string]string{},
		originator:  firstNonEmpty(cfg.Options["originator"], "proagent"),
	}
	for key, value := range cfg.Options {
		if strings.HasPrefix(strings.ToLower(key), "header.") {
			client.headers[strings.TrimPrefix(key, "header.")] = value
		}
	}
	_ = ctx
	return client, nil
}

func (c *CodexClient) SupportsStreaming() bool { return true }

func (c *CodexClient) CreateCompletion(ctx context.Context, messages []llm.Message, params llm.InferenceParams) (*llm.Message, error) {
	stream, err := c.CreateCompletionStream(ctx, messages, params)
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	var last *llm.Message
	for {
		event, err := stream.Recv(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		snapshot := event.Snapshot
		last = &snapshot
	}
	if last == nil {
		return nil, fmt.Errorf("codex returned no message")
	}
	return last, nil
}

func filterNativeTools(tools []llm.Tool) []llm.Tool {
	var out []llm.Tool
	for _, tool := range tools {
		name := tool.Schema().Name
		if name == "imagegen" || name == "web_run" || name == "web.run" {
			continue
		}
		out = append(out, tool)
	}
	return out
}

type wsCacheEntry struct {
	conn      *websocket.Conn
	expiresAt time.Time
}

var (
	wsCacheMu sync.Mutex
	wsCache   = make(map[string]*wsCacheEntry)
)

func getCachedWSConn(key string) *websocket.Conn {
	wsCacheMu.Lock()
	defer wsCacheMu.Unlock()
	entry, ok := wsCache[key]
	if !ok {
		return nil
	}
	if time.Now().After(entry.expiresAt) {
		entry.conn.Close()
		delete(wsCache, key)
		return nil
	}
	delete(wsCache, key)
	return entry.conn
}

func cacheWSConn(key string, conn *websocket.Conn) {
	wsCacheMu.Lock()
	defer wsCacheMu.Unlock()
	if entry, ok := wsCache[key]; ok {
		entry.conn.Close()
	}
	wsCache[key] = &wsCacheEntry{
		conn:      conn,
		expiresAt: time.Now().Add(5 * time.Minute),
	}
}

func (c *CodexClient) CreateCompletionStream(ctx context.Context, messages []llm.Message, params llm.InferenceParams) (llm.Stream, error) {
	if err := llm.ValidateMessages(messages); err != nil {
		return nil, err
	}
	token, err := c.tokenSource.Token()
	if err != nil {
		return nil, err
	}
	if token == nil || token.AccessToken == "" {
		return nil, fmt.Errorf("openai codex OAuth token is missing")
	}
	accountID := oauthx.AccountIDFromJWT(token.AccessToken, codexJWTClaimPath)
	if accountID == "" {
		return nil, fmt.Errorf("openai codex OAuth token does not contain chatgpt account id")
	}
	body, err := c.requestBody(messages, params)
	if err != nil {
		return nil, err
	}
	var requestTier *string
	if params.ServiceTier != nil {
		requestTier = params.ServiceTier
	}
	stream := newResponsesStream(requestTier)
	go c.runStream(ctx, stream, body, token.AccessToken, accountID)
	return stream, nil
}

func (c *CodexClient) runStream(ctx context.Context, stream *responsesStream, body map[string]any, token, accountID string) {
	if err := c.runWebSocket(ctx, stream, body, token, accountID); err != nil {
		stream.addDiagnostic("websocket fallback: " + err.Error())
		if err := c.runSSE(ctx, stream, body, token, accountID); err != nil {
			stream.fail(err)
		}
	}
}

func (c *CodexClient) runWebSocket(ctx context.Context, stream *responsesStream, body map[string]any, token, accountID string) error {
	headers := http.Header{}
	for key, value := range c.baseHeaders(token, accountID) {
		headers.Set(key, value)
	}
	headers.Set("OpenAI-Beta", codexWebSocketBeta)
	requestID := fmt.Sprintf("codex_%d", time.Now().UnixNano())
	headers.Set("x-client-request-id", requestID)
	headers.Set("session_id", requestID)

	var conn *websocket.Conn
	cachedConn := getCachedWSConn(accountID)
	if cachedConn != nil {
		conn = cachedConn
	} else {
		dialer := websocket.Dialer{HandshakeTimeout: 45 * time.Second}
		var err error
		conn, _, err = dialer.DialContext(ctx, c.codexWebSocketURL(), headers)
		if err != nil {
			return err
		}
	}

	shouldCache := false
	defer func() {
		if !shouldCache {
			conn.Close()
		}
	}()

	payload, _ := json.Marshal(body)
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return err
	}

	for {
		_, data, err := conn.ReadMessage()
		if err != nil {
			return err
		}
		if len(bytes.TrimSpace(data)) == 0 {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("invalid codex websocket event: %w", err)
		}
		if done, err := stream.handleEvent(event); err != nil || done {
			if done && err == nil {
				shouldCache = true
				cacheWSConn(accountID, conn)
			}
			return err
		}
	}
}

func (c *CodexClient) runSSE(ctx context.Context, stream *responsesStream, body map[string]any, token, accountID string) error {
	payload, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.codexURL(), bytes.NewReader(payload))
	if err != nil {
		return err
	}
	for key, value := range c.baseHeaders(token, accountID) {
		req.Header.Set(key, value)
	}
	req.Header.Set("OpenAI-Beta", codexSSEBeta)
	req.Header.Set("accept", "text/event-stream")
	req.Header.Set("content-type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("codex HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return stream.scanSSE(resp.Body)
}

func (c *CodexClient) baseHeaders(token, accountID string) map[string]string {
	headers := map[string]string{}
	for key, value := range c.headers {
		headers[key] = value
	}
	headers["Authorization"] = "Bearer " + token
	headers["chatgpt-account-id"] = accountID
	headers["originator"] = c.originator
	headers["User-Agent"] = "proagent"
	return headers
}

func (c *CodexClient) codexURL() string {
	if strings.HasSuffix(c.baseURL, "/codex/responses") {
		return c.baseURL
	}
	if strings.HasSuffix(c.baseURL, "/codex") {
		return c.baseURL + "/responses"
	}
	return c.baseURL + "/codex/responses"
}

func (c *CodexClient) codexWebSocketURL() string {
	u, err := url.Parse(c.codexURL())
	if err != nil {
		return c.codexURL()
	}
	switch u.Scheme {
	case "https":
		u.Scheme = "wss"
	case "http":
		u.Scheme = "ws"
	}
	return u.String()
}

func (c *CodexClient) requestBody(messages []llm.Message, params llm.InferenceParams) (map[string]any, error) {
	input, err := responsesInput(messages)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"model":               params.Model,
		"store":               false,
		"stream":              true,
		"instructions":        firstNonEmpty(params.SystemPrompt, "You are a helpful assistant."),
		"input":               input,
		"text":                map[string]any{"verbosity": "low"},
		"include":             []string{"reasoning.encrypted_content"},
		"tool_choice":         "auto",
		"parallel_tool_calls": true,
	}
	if params.Temperature != nil {
		body["temperature"] = *params.Temperature
	}
	if params.Tools != nil {
		body["tools"] = responsesTools(filterNativeTools(params.Tools))
	}
	if params.Thinking != nil && params.Thinking.Enabled {
		body["reasoning"] = map[string]any{"effort": responsesReasoningEffort(params.Thinking.Level), "summary": "auto"}
	}
	if params.ServiceTier != nil && *params.ServiceTier != "" {
		body["service_tier"] = *params.ServiceTier
	}
	return body, nil
}

func init() {
	llm.RegisterProviderFactory(llm.APIFormatOpenAICodex, func(ctx context.Context, cfg llm.ProviderConfig) (llm.Client, error) {
		var tokenSource oauth2.TokenSource
		if cfg.Auth.Type != "" && cfg.Auth.Type != "none" {
			oauthCfg := oauthx.Config{
				Type:         cfg.Auth.Type,
				ProviderID:   cfg.Auth.ProviderID,
				TokenURL:     cfg.Auth.TokenURL,
				AuthURL:      cfg.Auth.AuthURL,
				ClientID:     cfg.Auth.ClientID,
				ClientSecret: cfg.Auth.ClientSecret,
				Scopes:       cfg.Auth.Scopes,
				RedirectURL:  cfg.Auth.RedirectURL,
				CacheKey:     cfg.Auth.CacheKey,
				AuthParams:   cfg.Auth.AuthParams,
			}
			var err error
			tokenSource, err = oauthx.TokenSource(ctx, oauthCfg)
			if err != nil {
				return nil, err
			}
		}

		return NewCodexClient(ctx, cfg, tokenSource)
	})
}

