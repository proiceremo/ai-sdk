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
	"os"
	"strings"
	"time"

	llm "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/oauthx"
	"github.com/gorilla/websocket"
	"golang.org/x/oauth2"
)

type ResponsesClient struct {
	client      *http.Client
	apiKey      string
	baseURL     string
	headers     map[string]string
	tokenSource oauth2.TokenSource
}

func NewResponsesClient(apiKey string, baseURL string) *ResponsesClient {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	return &ResponsesClient{
		client:  &http.Client{Timeout: 10 * time.Minute},
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		headers: make(map[string]string),
	}
}

func (c *ResponsesClient) SupportsStreaming() bool { return true }

func (c *ResponsesClient) CreateCompletion(ctx context.Context, messages []llm.Message, params llm.InferenceParams) (*llm.Message, error) {
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
		return nil, fmt.Errorf("responses api returned no message")
	}
	return last, nil
}

func (c *ResponsesClient) CreateCompletionStream(ctx context.Context, messages []llm.Message, params llm.InferenceParams) (llm.Stream, error) {
	if err := llm.ValidateMessages(messages); err != nil {
		return nil, err
	}
	var token string
	if c.tokenSource != nil {
		t, err := c.tokenSource.Token()
		if err != nil {
			return nil, err
		}
		if t != nil {
			token = t.AccessToken
		}
	} else {
		token = c.apiKey
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
	go c.runStream(ctx, stream, body, token)
	return stream, nil
}

func (c *ResponsesClient) runStream(ctx context.Context, stream *responsesStream, body map[string]any, token string) {
	// Attempt WebSocket connection, fall back to SSE
	if err := c.runWebSocket(ctx, stream, body, token); err != nil {
		stream.addDiagnostic("websocket fallback: " + err.Error())
		if err := c.runSSE(ctx, stream, body, token); err != nil {
			stream.fail(err)
		}
	}
}

func (c *ResponsesClient) runWebSocket(ctx context.Context, stream *responsesStream, body map[string]any, token string) error {
	headers := http.Header{}
	for key, value := range c.baseHeaders(token) {
		headers.Set(key, value)
	}
	headers.Set("OpenAI-Beta", "responses_websockets=2026-02-06")
	requestID := fmt.Sprintf("responses_%d", time.Now().UnixNano())
	headers.Set("x-client-request-id", requestID)
	headers.Set("session_id", requestID)

	dialer := websocket.Dialer{
		HandshakeTimeout: 45 * time.Second,
	}
	conn, _, err := dialer.DialContext(ctx, c.responsesWebSocketURL(), headers)
	if err != nil {
		return err
	}
	defer conn.Close()

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
			return fmt.Errorf("invalid responses websocket event: %w", err)
		}
		if done, err := stream.handleEvent(event); err != nil || done {
			return err
		}
	}
}

func (c *ResponsesClient) runSSE(ctx context.Context, stream *responsesStream, body map[string]any, token string) error {
	payload, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.responsesURL(), bytes.NewReader(payload))
	if err != nil {
		return err
	}
	for key, value := range c.baseHeaders(token) {
		req.Header.Set(key, value)
	}
	req.Header.Set("OpenAI-Beta", "responses=experimental")
	req.Header.Set("accept", "text/event-stream")
	req.Header.Set("content-type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("responses HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return stream.scanSSE(resp.Body)
}

func (c *ResponsesClient) baseHeaders(token string) map[string]string {
	headers := map[string]string{}
	for key, value := range c.headers {
		headers[key] = value
	}
	if token != "" {
		headers["Authorization"] = "Bearer " + token
	}
	headers["User-Agent"] = "proagent"
	return headers
}

func (c *ResponsesClient) responsesURL() string {
	if strings.HasSuffix(c.baseURL, "/responses") {
		return c.baseURL
	}
	return c.baseURL + "/responses"
}

func (c *ResponsesClient) responsesWebSocketURL() string {
	u, err := url.Parse(c.responsesURL())
	if err != nil {
		return c.responsesURL()
	}
	switch u.Scheme {
	case "https":
		u.Scheme = "wss"
	case "http":
		u.Scheme = "ws"
	}
	return u.String()
}

func (c *ResponsesClient) requestBody(messages []llm.Message, params llm.InferenceParams) (map[string]any, error) {
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
		body["tools"] = responsesTools(params.Tools)
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
	llm.RegisterProviderFactory(llm.APIFormatOpenAIResponses, func(ctx context.Context, cfg llm.ProviderConfig) (llm.Client, error) {
		apiKey := os.Getenv(cfg.EnvKey)
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}

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

		client := NewResponsesClient(apiKey, cfg.BaseURL)
		client.tokenSource = tokenSource
		for k, v := range cfg.Options {
			if strings.HasPrefix(strings.ToLower(k), "header.") {
				client.headers[strings.TrimPrefix(k, "header.")] = v
			}
		}
		return client, nil
	})
}
