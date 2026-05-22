package openai

import (
	"bufio"
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

	llm "ai-sdk"
	"ai-sdk/oauthx"
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
	stream := newCodexStream(requestTier)
	go c.runStream(ctx, stream, body, token.AccessToken, accountID)
	return stream, nil
}

func (c *CodexClient) runStream(ctx context.Context, stream *codexStream, body map[string]any, token, accountID string) {
	if err := c.runWebSocket(ctx, stream, body, token, accountID); err != nil {
		stream.addDiagnostic("websocket fallback: " + err.Error())
		if err := c.runSSE(ctx, stream, body, token, accountID); err != nil {
			stream.fail(err)
		}
	}
}

func (c *CodexClient) runWebSocket(ctx context.Context, stream *codexStream, body map[string]any, token, accountID string) error {
	headers := http.Header{}
	for key, value := range c.baseHeaders(token, accountID) {
		headers.Set(key, value)
	}
	headers.Set("OpenAI-Beta", codexWebSocketBeta)
	requestID := newRequestID()
	headers.Set("x-client-request-id", requestID)
	headers.Set("session_id", requestID)
	conn, _, err := websocket.DefaultDialer.DialContext(ctx, c.codexWebSocketURL(), headers)
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
			return fmt.Errorf("invalid codex websocket event: %w", err)
		}
		if done, err := stream.handleEvent(event); err != nil || done {
			return err
		}
	}
}

func (c *CodexClient) runSSE(ctx context.Context, stream *codexStream, body map[string]any, token, accountID string) error {
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
	input, err := codexInput(messages)
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
		body["tools"] = codexTools(params.Tools)
	}
	if params.Thinking != nil && params.Thinking.Enabled {
		body["reasoning"] = map[string]any{"effort": codexReasoningEffort(params.Thinking.Level), "summary": "auto"}
	}
	if params.ServiceTier != nil && *params.ServiceTier != "" {
		body["service_tier"] = *params.ServiceTier
	}
	return body, nil
}

func codexInput(messages []llm.Message) ([]any, error) {
	var out []any
	for _, msg := range messages {
		for _, block := range msg.Content {
			if block.ToolOutput == nil {
				continue
			}
			callID, _ := splitCodexToolID(block.ToolOutput.ToolUseID)
			out = append(out, map[string]any{"type": "function_call_output", "call_id": callID, "output": block.ToolOutput.Output.ToString()})
		}
		switch msg.Role {
		case llm.MessageRoleUser:
			out = append(out, map[string]any{"role": "user", "content": codexInputContent(msg.Content)})
		case llm.MessageRoleAssistant:
			for _, block := range msg.Content {
				switch block.Type {
				case llm.ContentBlockTypeText:
					out = append(out, map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": block.Text, "annotations": []any{}}}, "status": "completed"})
				case llm.ContentBlockTypeToolUse:
					if block.ToolUse != nil {
						callID, itemID := splitCodexToolID(block.ToolUse.ID)
						out = append(out, map[string]any{"type": "function_call", "id": itemID, "call_id": callID, "name": block.ToolUse.Name, "arguments": string(block.ToolUse.Input)})
					}
				}
			}
		}
	}
	return out, nil
}

func codexInputContent(content llm.MessageContent) []any {
	parts := []any{}
	for _, block := range content {
		switch block.Type {
		case llm.ContentBlockTypeText:
			parts = append(parts, map[string]any{"type": "input_text", "text": block.Text})
		case llm.ContentBlockTypeImage:
			if block.Image != nil {
				if block.Image.Type == llm.ImageSourceTypeURL {
					parts = append(parts, map[string]any{"type": "input_image", "detail": "auto", "image_url": block.Image.URL})
				} else if block.Image.Data != "" {
					media := firstNonEmpty(block.Image.MediaType, "image/png")
					parts = append(parts, map[string]any{"type": "input_image", "detail": "auto", "image_url": "data:" + media + ";base64," + block.Image.Data})
				}
			}
		}
	}
	return parts
}

func codexTools(tools []llm.Tool) []any {
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		schema := tool.Schema()
		out = append(out, map[string]any{
			"type":        "function",
			"name":        schema.Name,
			"description": schema.Description,
			"parameters":  schema.InputSchema,
			"strict":      schema.Strict,
		})
	}
	return out
}

func codexReasoningEffort(level llm.ThinkingLevel) string {
	switch level {
	case llm.ThinkingLevelLow:
		return "low"
	case llm.ThinkingLevelMedium:
		return "medium"
	case llm.ThinkingLevelHigh:
		return "high"
	case llm.ThinkingLevelXHigh:
		return "xhigh"
	default:
		return "medium"
	}
}

type codexStream struct {
	base        *llm.BaseStream
	ch          chan *llm.StreamEvent
	done        chan struct{}
	once        sync.Once
	mu          sync.Mutex
	err         error
	toolArgLen  int
	requestTier *string
}

func newCodexStream(requestTier *string) *codexStream {
	s := &codexStream{
		base:        llm.NewBaseStream(),
		ch:          make(chan *llm.StreamEvent, 64),
		done:        make(chan struct{}),
		requestTier: requestTier,
	}
	s.base.EmitMessageStart()
	s.flush()
	return s
}

func (s *codexStream) Recv(ctx context.Context) (*llm.StreamEvent, error) {
	select {
	case event, ok := <-s.ch:
		if !ok {
			s.mu.Lock()
			err := s.err
			s.mu.Unlock()
			if err != nil {
				return nil, err
			}
			return nil, io.EOF
		}
		return event, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *codexStream) Current() *llm.Message {
	msg := s.base.Accumulated()
	return &msg
}

func (s *codexStream) Close() error {
	s.finish(nil)
	return nil
}

func (s *codexStream) fail(err error) {
	s.mu.Lock()
	s.err = err
	s.mu.Unlock()
	s.finish(nil)
}

func (s *codexStream) finish(usage *llm.Usage) {
	s.once.Do(func() {
		msg := s.base.Accumulated()
		stop := llm.StopReasonEndTurn
		if len(msg.Content.ToolCalls()) > 0 {
			stop = llm.StopReasonToolUse
		}
		s.base.Finish(stop, usage)
		s.flush()
		close(s.ch)
		close(s.done)
	})
}

func (s *codexStream) flush() {
	for {
		event := s.base.PopEvent()
		if event == nil {
			return
		}
		s.ch <- event
	}
}

func (s *codexStream) addDiagnostic(text string) {
	_ = text
}

func (s *codexStream) handleEvent(event map[string]any) (bool, error) {
	eventType, _ := event["type"].(string)
	switch eventType {
	case "error":
		return false, fmt.Errorf("codex error: %v", event)
	case "response.failed":
		return false, fmt.Errorf("codex response failed: %v", event)
	case "response.output_item.added":
		item, _ := event["item"].(map[string]any)
		s.openItem(item)
	case "response.output_text.delta", "response.refusal.delta":
		if delta, _ := event["delta"].(string); delta != "" {
			s.base.AppendDelta(llm.NewTextContentBlock(delta))
			s.flush()
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		if delta, _ := event["delta"].(string); delta != "" {
			s.base.AppendDelta(llm.ContentBlock{Type: llm.ContentBlockTypeThinking, Thinking: delta})
			s.flush()
		}
	case "response.function_call_arguments.delta":
		if delta, _ := event["delta"].(string); delta != "" {
			s.toolArgLen += len(delta)
			s.base.AppendDelta(llm.ContentBlock{Type: llm.ContentBlockTypeToolUse, ToolUse: &llm.ToolUse{Input: json.RawMessage(delta)}})
			s.flush()
		}
	case "response.function_call_arguments.done":
		if args, _ := event["arguments"].(string); args != "" {
			if s.toolArgLen < len(args) {
				delta := args[s.toolArgLen:]
				s.toolArgLen = len(args)
				s.base.AppendDelta(llm.ContentBlock{Type: llm.ContentBlockTypeToolUse, ToolUse: &llm.ToolUse{Input: json.RawMessage(delta)}})
				s.flush()
			}
		}
	case "response.output_item.done":
		item, _ := event["item"].(map[string]any)
		s.doneItem(item)
	case "response.done", "response.completed", "response.incomplete":
		usage := s.codexUsage(event)
		s.finish(usage)
		return true, nil
	}
	return false, nil
}

func (s *codexStream) openItem(item map[string]any) {
	switch item["type"] {
	case "message":
		s.base.OpenBlock(llm.NewTextContentBlock(""))
	case "reasoning":
		s.base.OpenBlock(llm.ContentBlock{Type: llm.ContentBlockTypeThinking})
	case "function_call":
		callID, _ := item["call_id"].(string)
		itemID, _ := item["id"].(string)
		name, _ := item["name"].(string)
		args, _ := item["arguments"].(string)
		s.toolArgLen = len(args)
		s.base.OpenBlock(llm.ContentBlock{Type: llm.ContentBlockTypeToolUse, ToolUse: &llm.ToolUse{ID: callID + "|" + itemID, Name: name, Input: json.RawMessage(args)}})
	}
	s.flush()
}

func (s *codexStream) doneItem(item map[string]any) {
	switch item["type"] {
	case "message":
		if content, ok := item["content"].([]any); ok {
			var text strings.Builder
			for _, raw := range content {
				part, _ := raw.(map[string]any)
				if v, _ := part["text"].(string); v != "" {
					text.WriteString(v)
				}
				if v, _ := part["refusal"].(string); v != "" {
					text.WriteString(v)
				}
			}
			if text.Len() > 0 {
				// Keep deltas as source of truth; done closes the block.
			}
		}
		s.base.CloseCurrentBlock("", nil)
	case "reasoning":
		s.base.CloseCurrentBlock("", nil)
	case "function_call":
		s.base.CloseCurrentBlock(llm.StopReasonToolUse, nil)
	}
	s.flush()
}

func (s *codexStream) scanSSE(body io.Reader) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var data []string
	flush := func() (bool, error) {
		if len(data) == 0 {
			return false, nil
		}
		payload := strings.TrimSpace(strings.Join(data, "\n"))
		data = nil
		if payload == "" || payload == "[DONE]" {
			return false, nil
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return false, err
		}
		return s.handleEvent(event)
	}
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			done, err := flush()
			if err != nil || done {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			data = append(data, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if done, err := flush(); err != nil || done {
		return err
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func resolveCodexServiceTier(responseTier *string, requestTier *string) string {
	resTier := ""
	if responseTier != nil {
		resTier = *responseTier
	}
	reqTier := ""
	if requestTier != nil {
		reqTier = *requestTier
	}

	if resTier == "default" && (reqTier == "flex" || reqTier == "priority") {
		return reqTier
	}
	if resTier != "" {
		return resTier
	}
	return reqTier
}

func (s *codexStream) codexUsage(event map[string]any) *llm.Usage {
	response, _ := event["response"].(map[string]any)
	rawUsage, _ := response["usage"].(map[string]any)
	if rawUsage == nil {
		return nil
	}
	input := intFromAny(rawUsage["input_tokens"])
	output := intFromAny(rawUsage["output_tokens"])
	total := intFromAny(rawUsage["total_tokens"])
	// OpenAI's Responses API reports cache hits under
	// input_tokens_details.cached_tokens and bakes them INTO input_tokens.
	// Our TokenUsage convention (matching Anthropic) keeps the two
	// separate: InputTokens is the fresh portion, CacheReadInputTokens is
	// the hit count. So we read the details bucket and subtract — same
	// shape the pi reference uses in openai-responses-shared.ts.
	var cacheRead, inputAudio int
	if details, ok := rawUsage["input_tokens_details"].(map[string]any); ok {
		cacheRead = intFromAny(details["cached_tokens"])
		inputAudio = intFromAny(details["audio_tokens"])
	}
	if cacheRead > input {
		// Defensive: never let the subtraction go negative if the
		// provider ever ships a buggy payload.
		cacheRead = input
	}
	input -= cacheRead
	if total == 0 {
		total = input + cacheRead + output
	}
	// output_tokens_details.reasoning_tokens — GPT-5/Codex reasoning
	// models report how many of the completion tokens were hidden
	// chain-of-thought. Bench surfaces this so reasoning-heavy runs are
	// distinguishable from pure-output ones.
	var reasoning, outputAudio int
	if details, ok := rawUsage["output_tokens_details"].(map[string]any); ok {
		reasoning = intFromAny(details["reasoning_tokens"])
		outputAudio = intFromAny(details["audio_tokens"])
	}

	var responseTier *string
	if response != nil {
		if t, ok := response["service_tier"].(string); ok && t != "" {
			responseTier = &t
		}
	}

	tokens := llm.TokenUsage{
		InputTokens:           input,
		OutputTokens:          output,
		TotalTokens:           total,
		CacheReadInputTokens:  cacheRead,
		ReasoningOutputTokens: reasoning,
		ServiceTier:           resolveCodexServiceTier(responseTier, s.requestTier),
	}
	if inputAudio > 0 {
		tokens.InputTokensDetails = &llm.UsageTokenDetails{AudioTokens: inputAudio}
	}
	if outputAudio > 0 {
		tokens.OutputTokensDetails = &llm.UsageTokenDetails{AudioTokens: outputAudio}
	}
	return llm.NewUsage(llm.UsageOperationCompletion, tokens)
}

func intFromAny(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	default:
		return 0
	}
}

func splitCodexToolID(id string) (callID, itemID string) {
	callID = id
	if strings.Contains(id, "|") {
		parts := strings.SplitN(id, "|", 2)
		callID, itemID = parts[0], parts[1]
	}
	if itemID == "" {
		itemID = "fc_" + strings.TrimPrefix(callID, "call_")
	}
	return callID, itemID
}

func newRequestID() string {
	return fmt.Sprintf("codex_%d", time.Now().UnixNano())
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
