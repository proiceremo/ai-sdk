package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	. "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/providers/internal/httpx"
	"golang.org/x/oauth2"
)

const (
	defaultAnthropicBaseURL = "https://api.anthropic.com"
	defaultAnthropicVersion = "2023-06-01"
	defaultAnthropicTokens  = 4096
)

type AnthropicClient struct {
	client      *http.Client
	apiKey      string
	baseURL     string
	apiVersion  string
	tokenSource oauth2.TokenSource
}

func (c *AnthropicClient) SupportsStreaming() bool { return true }

func NewAnthropicClient(ctx context.Context, apiKey string, baseURL string) (*AnthropicClient, error) {
	_ = ctx
	if baseURL == "" {
		baseURL = defaultAnthropicBaseURL
	}
	return &AnthropicClient{
		client:     &http.Client{Timeout: 10 * time.Minute},
		apiKey:     apiKey,
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiVersion: defaultAnthropicVersion,
	}, nil
}

func NewAnthropicClientWithTokenSource(ctx context.Context, source oauth2.TokenSource, baseURL string) (*AnthropicClient, error) {
	client, err := NewAnthropicClient(ctx, "", baseURL)
	if err != nil {
		return nil, err
	}
	client.tokenSource = source
	return client, nil
}

// isNativeEndpoint returns true if the client is targeting the native Anthropic API.
// Third-party Anthropic-compatible endpoints (DeepSeek, Moonshot, etc.) may not
// support newer Anthropic API features like the "custom" tool type.
func (c *AnthropicClient) isNativeEndpoint() bool {
	return strings.Contains(c.baseURL, "anthropic.com")
}

func (c *AnthropicClient) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	if err := ValidateMessages(messages); err != nil {
		return nil, err
	}
	request, err := c.buildCompletionRequest(messages, params, false)
	if err != nil {
		return nil, err
	}

	var response anthropicMessage
	headers, err := c.requestHeaders(ctx, request)
	if err != nil {
		return nil, err
	}
	if err := httpx.DoJSON(ctx, c.client, http.MethodPost, c.baseURL+"/v1/messages", headers, request, &response); err != nil {
		return nil, err
	}
	return c.fromAnthropicMessage(&response), nil
}

func (c *AnthropicClient) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	if err := ValidateMessages(messages); err != nil {
		return nil, err
	}
	request, err := c.buildCompletionRequest(messages, params, true)
	if err != nil {
		return nil, err
	}

	headers, err := c.requestHeaders(ctx, request)
	if err != nil {
		return nil, err
	}
	stream, err := httpx.DoSSE(ctx, c.client, http.MethodPost, c.baseURL+"/v1/messages", headers, request)
	if err != nil {
		return nil, err
	}

	return &anthropicStream{
		BaseStream: NewBaseStream(),
		stream:     stream,
	}, nil
}

type anthropicStream struct {
	*BaseStream
	stream *httpx.SSEStream
	usage  *Usage
	err    error
}

func (s *anthropicStream) Recv(ctx context.Context) (*StreamEvent, error) {
	_ = ctx
	if event := s.PopEvent(); event != nil {
		return event, nil
	}
	if s.IsFinished() {
		return nil, io.EOF
	}

	for s.stream.Next() {
		rawEvent := s.stream.Current()
		if len(rawEvent.Data) == 0 {
			continue
		}

		var event anthropicStreamEvent
		if err := json.Unmarshal(rawEvent.Data, &event); err != nil {
			s.err = err
			s.Finish("", nil)
			return nil, err
		}
		if err := s.processEvent(event); err != nil {
			s.err = err
			s.Finish("", nil)
			return nil, err
		}
		if pending := s.PopEvent(); pending != nil {
			return pending, nil
		}
	}

	if err := s.stream.Err(); err != nil {
		s.err = err
		s.Finish("", nil)
		return nil, err
	}

	accumulated := s.Accumulated()
	s.Finish(accumulated.StopReason, accumulated.Usage)
	if event := s.PopEvent(); event != nil {
		return event, nil
	}
	return nil, io.EOF
}

func (s *anthropicStream) processEvent(event anthropicStreamEvent) error {
	switch event.Type {
	case "message_start":
		s.EmitMessageStart()
		if event.Message != nil {
			s.usage = mapAnthropicUsage(event.Message.Usage)
			if s.usage != nil {
				s.SetUsage(s.usage)
			}
		}
	case "content_block_start":
		block, err := anthropicContentBlockToContentBlock(event.ContentBlock)
		if err != nil {
			return err
		}
		if block != nil {
			s.OpenBlock(*block)
		}
	case "content_block_delta":
		block, err := anthropicDeltaToContentBlock(event.Delta)
		if err != nil {
			return err
		}
		if block != nil {
			s.AppendDelta(*block)
		}
	case "content_block_stop":
		s.CloseCurrentBlock("", nil)
	case "message_delta":
		var delta anthropicMessageDelta
		if err := json.Unmarshal(event.Delta, &delta); err != nil {
			return err
		}
		if delta.StopReason != "" {
			s.SetStopReason(NormaliseStopReason(APIFormatAnthropic, delta.StopReason))
		}
		if event.Usage != nil {
			s.applyUsageDelta(*event.Usage)
		}
	case "message_stop":
		s.Finish(s.Accumulated().StopReason, s.usage)
	}
	return nil
}

func (s *anthropicStream) applyUsageDelta(delta anthropicMessageDeltaUsage) {
	tokens := s.usage.FirstTokenUsage()
	if delta.InputTokens != nil {
		tokens.InputTokens = *delta.InputTokens
	}
	if delta.CacheCreationInputTokens != nil {
		tokens.CacheCreationInputTokens = *delta.CacheCreationInputTokens
	}
	if delta.CacheReadInputTokens != nil {
		tokens.CacheReadInputTokens = *delta.CacheReadInputTokens
	}
	if delta.OutputTokens > 0 {
		tokens.OutputTokens = delta.OutputTokens
	}
	tokens.CacheBilledSeparately = true
	tokens.TotalTokens = tokens.InputTokens + tokens.OutputTokens
	s.usage = NewUsage(UsageOperationCompletion, tokens)
	s.SetUsage(s.usage)
}

func (s *anthropicStream) Current() *Message {
	accumulated := s.Accumulated()
	return &accumulated
}

func (s *anthropicStream) Close() error {
	s.Finish("", nil)
	if s.stream == nil {
		return nil
	}
	return s.stream.Close()
}

func (c *AnthropicClient) buildCompletionRequest(messages []Message, params InferenceParams, stream bool) (*anthropicMessageRequest, error) {
	anthropicMessages, err := c.toAnthropicMessages(messages, c.isNativeEndpoint())
	if err != nil {
		return nil, err
	}

	maxTokens := defaultAnthropicTokens
	if params.MaxTokens != nil && *params.MaxTokens > 0 {
		maxTokens = *params.MaxTokens
	}

	request := &anthropicMessageRequest{
		Model:       params.Model,
		MaxTokens:   maxTokens,
		Messages:    anthropicMessages,
		Temperature: params.Temperature,
		TopP:        params.TopP,
		TopK:        params.TopK,
		Stream:      stream,
	}
	// OAuth tokens issued by claude.ai/oauth/authorize are scoped to the
	// Claude-Code account. Anthropic feature-gates these tokens unless
	// the request looks like it's coming from the Claude-Code CLI — the
	// canonical signal is an identity system block as the FIRST entry of
	// the system array. pi does this at
	// inspo/pi/packages/ai/src/providers/anthropic.ts:905. Without it
	// some calls succeed but capability-gated ones (tool use on certain
	// models, large context windows, etc.) return 400 or get downgraded
	// silently. The block also lengthens the cacheable prefix.
	useOAuth := c.tokenSource != nil
	cacheTTL := resolveCacheTTL(params.CacheRetention)
	systemBlocks := buildSystemBlocks(params.SystemPrompt, useOAuth, cacheTTL)
	if len(systemBlocks) > 0 {
		request.System = systemBlocks
	}
	if params.Thinking != nil && params.Thinking.Enabled {
		request.Thinking = &anthropicThinking{
			Type:         "enabled",
			BudgetTokens: anthropicThinkingBudget(*params.Thinking, maxTokens),
		}
	}
	tools := []any{}
	if params.WebSearch != nil && params.WebSearch.Enabled {
		if params.WebSearch.CustomTool != "" {
			toolType := params.WebSearch.Type
			if toolType == "" {
				toolType = "custom"
			}
			request.Tools = []any{
				map[string]any{
					"type": toolType,
					"name": params.WebSearch.CustomTool,
				},
			}
		} else {
			request.Tools = []any{
				map[string]any{
					"type": "web_search_20250305",
					"name": "web_search",
				},
			}
		}
	} else if len(params.Tools) > 0 {
		// isNativeAnthropic := c.isNativeEndpoint()
		for _, tool := range params.Tools {
			schema := tool.Schema()
			inputSchema := map[string]any{
				"type":                 "object",
				"properties":           map[string]any{},
				"additionalProperties": false,
			}
			if schema.InputSchema != nil {
				inputSchema = schema.InputSchema
			}

			toolDefinition := map[string]any{
				"name":         schema.Name,
				"description":  schema.Description,
				"input_schema": inputSchema,
			}
			if schema.Strict {
				toolDefinition["strict"] = true
			}
			tools = append(tools, toolDefinition)
		}
	}
	if len(tools) > 0 {
		// Cache the entire tool schema set by marking the last tool
		// definition with cache_control. Tool schemas are large
		// (input_schema JSON per tool) and stable across turns of an
		// agent loop, so this typically captures 1-5k cached tokens
		// per request. pi does the same at anthropic.ts:1185.
		markLastToolCached(tools, cacheTTL)
		request.Tools = tools
	}
	// Mark the last content block of the LAST user/assistant message
	// with cache_control. This caches the CONVERSATION HISTORY up to
	// the new turn, so each subsequent turn's request reuses the prefix
	// {system} + {tools} + {messages[0..n-1]} for free — the only fresh
	// tokens are the latest assistant response. Anthropic supports up
	// to 4 cache breakpoints per request; we use 3 (system, tools,
	// last message) leaving headroom for callers that want a custom
	// 4th. pi reference: anthropic.ts:1135-1156.
	markLastMessageCached(request.Messages, cacheTTL)
	if params.ToolChoice != nil {
		toolChoice, err := toAnthropicToolChoice(*params.ToolChoice)
		if err != nil {
			return nil, err
		}
		request.ToolChoice = toolChoice
	}
	if params.ResponseFormat != nil && params.ResponseFormat.Type == ResponseFormatTypeJSONObject {
		if params.ResponseFormat.JSONSchema == nil {
			return nil, fmt.Errorf("anthropic structured outputs require a JSON schema")
		}
		request.OutputConfig = &anthropicOutputConfig{
			Format: anthropicJSONOutputFormat{
				Type:   "json_schema",
				Schema: params.ResponseFormat.JSONSchema,
			},
		}
	}

	return request, nil
}

func toAnthropicToolChoice(choice ToolChoice) (*anthropicToolChoice, error) {
	disableParallel := true
	switch choice.Type {
	case "", ToolChoiceTypeAuto:
		return &anthropicToolChoice{Type: "auto"}, nil
	case ToolChoiceTypeRequired:
		return &anthropicToolChoice{
			Type:                   "any",
			DisableParallelToolUse: &disableParallel,
		}, nil
	case ToolChoiceTypeTool:
		if strings.TrimSpace(choice.Name) == "" {
			return nil, fmt.Errorf("anthropic named tool choice requires a tool name")
		}
		return &anthropicToolChoice{
			Type:                   "tool",
			Name:                   choice.Name,
			DisableParallelToolUse: &disableParallel,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported anthropic tool choice type %q", choice.Type)
	}
}

func anthropicThinkingBudget(thinking ThinkingParams, maxTokens int) int {
	budget := 1024
	if thinking.Level != "" && thinking.Level != ThinkingLevelUnspecified {
		if pct, ok := ThinkingLevelMap[thinking.Level]; ok {
			budget = maxTokens * pct / 100
		}
	}
	if budget < 1024 {
		budget = 1024
	}
	if budget >= maxTokens {
		budget = maxTokens - 1
	}
	if budget < 1 {
		return 1
	}
	return budget
}

func (c *AnthropicClient) toAnthropicMessages(messages []Message, isNative bool) ([]anthropicMessageParam, error) {
	params := make([]anthropicMessageParam, 0, len(messages))

	for _, msg := range messages {
		content := make([]anthropicInputContentBlock, 0, len(msg.Content))
		for _, block := range msg.Content {
			converted, err := anthropicContentBlockFromContentBlock(block, isNative)
			if err != nil {
				return nil, err
			}
			if converted != nil {
				content = append(content, *converted)
			}
		}
		if len(content) == 0 {
			continue
		}

		role := "assistant"
		if msg.Role == MessageRoleUser {
			role = "user"
		}
		params = append(params, anthropicMessageParam{
			Role:    role,
			Content: content,
		})
	}

	return params, nil
}

func anthropicContentBlockFromContentBlock(block ContentBlock, isNative bool) (*anthropicInputContentBlock, error) {
	switch block.Type {
	case ContentBlockTypeText:
		if block.Text == "" {
			return nil, nil
		}
		return &anthropicInputContentBlock{Type: "text", Text: block.Text}, nil
	case ContentBlockTypeImage:
		if block.Image == nil {
			return nil, nil
		}
		source, err := anthropicImageSource(block.Image)
		if err != nil {
			return nil, err
		}
		return &anthropicInputContentBlock{Type: "image", Source: source}, nil
	case ContentBlockTypeDocument:
		if block.Document == nil {
			return nil, nil
		}
		// Non-native endpoints (Fireworks, etc.) don't support the Anthropic
		// "document" content block type. Fall back to a plain text block so
		// the content is still available to the model.
		if !isNative {
			text := anthropicDocumentText(block.Document)
			if text == "" {
				text = anthropicFallbackText(block)
			}
			if text == "" {
				return nil, nil
			}
			return &anthropicInputContentBlock{Type: "text", Text: text}, nil
		}
		source, err := anthropicDocumentSource(block.Document)
		if err != nil {
			return nil, err
		}
		return &anthropicInputContentBlock{
			Type:    "document",
			Source:  source,
			Title:   block.Document.Title,
			Context: block.Document.Context,
		}, nil
	case ContentBlockTypeToolUse:
		if block.ToolUse == nil {
			return nil, nil
		}
		input, err := rawJSONToAny(block.ToolUse.Input)
		if err != nil {
			return nil, err
		}
		toolType := "tool_use"
		if block.ToolUse.Execution == ToolExecutionModeServer {
			toolType = firstNonEmpty(strings.TrimSpace(block.ToolUse.ProviderType), "server_tool_use")
		}
		return &anthropicInputContentBlock{
			Type:  toolType,
			ID:    block.ToolUse.ID,
			Name:  block.ToolUse.Name,
			Input: input,
		}, nil
	case ContentBlockTypeToolResult:
		if block.ToolOutput == nil {
			return nil, nil
		}
		toolType := strings.TrimSpace(block.ToolOutput.ProviderType)
		if block.ToolOutput.Execution == ToolExecutionModeServer {
			content, err := anthropicProviderDataToAny(block.ToolOutput.ProviderData)
			if err != nil {
				return nil, err
			}
			if content == nil {
				content, err = anthropicToolResultContent(block.ToolOutput.Output, isNative)
				if err != nil {
					return nil, err
				}
			}
			return &anthropicInputContentBlock{
				Type:      toolType,
				ToolUseID: block.ToolOutput.ToolUseID,
				Content:   content,
			}, nil
		}
		content, err := anthropicToolResultContent(block.ToolOutput.Output, isNative)
		if err != nil {
			return nil, err
		}
		toolResult := &anthropicInputContentBlock{
			Type:      "tool_result",
			ToolUseID: block.ToolOutput.ToolUseID,
			Content:   content,
		}
		if block.ToolOutput.IsError {
			toolResult.IsError = boolPtr(true)
		}
		return toolResult, nil
	case ContentBlockTypeThinking:
		if block.Thinking == "" && block.Signature != "" {
			if !isNative {
				return nil, nil // Skip redacted thinking for non-native endpoints
			}
			return &anthropicInputContentBlock{
				Type: "redacted_thinking",
				Data: block.Signature,
			}, nil
		}
		return &anthropicInputContentBlock{
			Type:      "thinking",
			Thinking:  block.Thinking,
			Signature: block.Signature,
		}, nil
	case ContentBlockTypeRedactedThinking:
		if !isNative {
			return nil, nil // Skip redacted thinking for non-native endpoints
		}
		return &anthropicInputContentBlock{
			Type: "redacted_thinking",
			Data: firstNonEmpty(block.Redacted, block.Signature),
		}, nil
	default:
		return nil, nil
	}
}

func anthropicImageSource(image *ImageSource) (any, error) {
	switch image.Type {
	case ImageSourceTypeURL:
		return anthropicURLSource{
			Type: "url",
			URL:  image.URL,
		}, nil
	case ImageSourceTypeBase64:
		data := stripDataPrefix(image.Data)
		mediaType := image.MediaType
		if mediaType == "" {
			_, detected, err := DecodeBase64File(data)
			if err != nil {
				return nil, err
			}
			mediaType = detected
		}
		return anthropicBase64Source{
			Type:      "base64",
			MediaType: mediaType,
			Data:      data,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported image source type %q", image.Type)
	}
}

func anthropicDocumentSource(document *DocumentSource) (any, error) {
	switch document.Type {
	case DocumentSourceTypeURL:
		return anthropicURLSource{
			Type: "url",
			URL:  document.URL,
		}, nil
	case DocumentSourceTypeBase64:
		data := stripDataPrefix(document.Data)
		mediaType := document.MediaType
		if mediaType == "" {
			_, detected, err := DecodeBase64File(data)
			if err != nil {
				return nil, err
			}
			mediaType = detected
		}
		return anthropicBase64Source{
			Type:      "base64",
			MediaType: mediaType,
			Data:      data,
		}, nil
	case DocumentSourceTypeText:
		mediaType := document.MediaType
		if mediaType == "" {
			mediaType = "text/plain"
		}
		return anthropicPlainTextSource{
			Type:      "text",
			MediaType: mediaType,
			Data:      document.Text,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported document source type %q", document.Type)
	}
}

func anthropicToolResultContent(output MessageContent, isNative bool) (any, error) {
	content := make([]anthropicInputContentBlock, 0, len(output))
	for _, outputBlock := range output {
		switch outputBlock.Type {
		case ContentBlockTypeText, ContentBlockTypeImage, ContentBlockTypeDocument:
			converted, err := anthropicContentBlockFromContentBlock(outputBlock, isNative)
			if err != nil {
				return nil, err
			}
			if converted != nil {
				content = append(content, *converted)
			}
		default:
			fallback := anthropicFallbackText(outputBlock)
			content = append(content, anthropicInputContentBlock{
				Type: "text",
				Text: fallback,
			})
		}
	}
	if len(content) == 0 {
		return "", nil
	}
	return content, nil
}

func anthropicFallbackText(block ContentBlock) string {
	switch block.Type {
	case ContentBlockTypeAudio:
		if block.Audio != nil {
			if block.Audio.Type == AudioSourceTypeURL {
				return "[AUDIO] " + block.Audio.URL
			}
			return "[AUDIO] <base64 data>"
		}
	case ContentBlockTypeVideo:
		if block.Video != nil {
			if block.Video.Type == VideoSourceTypeURL {
				return "[VIDEO] " + block.Video.URL
			}
			return "[VIDEO] <base64 data>"
		}
	case ContentBlockTypeDocument:
		if block.Document != nil {
			if text := anthropicDocumentText(block.Document); text != "" {
				return text
			}
			if block.Document.Type == DocumentSourceTypeURL {
				return "[DOCUMENT] " + block.Document.URL
			}
			return "[DOCUMENT] <base64 data>"
		}
	}
	return "[CONTENT] " + string(block.Type)
}

func anthropicDocumentText(document *DocumentSource) string {
	if document == nil {
		return ""
	}
	parts := make([]string, 0, 3)
	for _, value := range []string{document.Title, document.Context, document.Text} {
		value = strings.TrimSpace(value)
		if value != "" {
			parts = append(parts, value)
		}
	}
	return strings.Join(parts, "\n")
}

func rawJSONToAny(value json.RawMessage) (any, error) {
	if len(value) == 0 {
		return map[string]any{}, nil
	}
	var decoded any
	if err := json.Unmarshal(value, &decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

func anthropicProviderDataToAny(value json.RawMessage) (any, error) {
	if len(value) == 0 {
		return nil, nil
	}
	var decoded any
	if err := json.Unmarshal(value, &decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

func (c *AnthropicClient) fromAnthropicMessage(response *anthropicMessage) *Message {
	if response == nil {
		return nil
	}
	content := make([]ContentBlock, 0, len(response.Content))
	for _, block := range response.Content {
		converted, err := anthropicContentBlockToContentBlock(&block)
		if err != nil {
			continue
		}
		if converted != nil {
			content = append(content, *converted)
		}
	}

	return &Message{
		Role:       MessageRole(response.Role),
		Content:    content,
		Timestamp:  time.Now(),
		StopReason: NormaliseStopReason(APIFormatAnthropic, response.StopReason),
		Usage:      mapAnthropicUsage(response.Usage),
	}
}

func anthropicContentBlockToContentBlock(block *anthropicContentBlock) (*ContentBlock, error) {
	if block == nil {
		return nil, nil
	}
	switch block.Type {
	case "text":
		content := NewTextContentBlock(block.Text)
		return &content, nil
	case "tool_use":
		content := NewToolUseContentBlock(block.ID, block.Name, block.Input, nil)
		return &content, nil
	case "server_tool_use":
		content := NewToolUseContentBlock(block.ID, block.Name, block.Input, nil)
		content.ToolUse.Execution = ToolExecutionModeServer
		content.ToolUse.ProviderType = "server_tool_use"
		return &content, nil
	case "web_search_tool_result", "web_fetch_tool_result":
		output, err := anthropicServerToolResultToToolOutput(block)
		if err != nil {
			return nil, err
		}
		content := NewToolResultContentBlock(*output)
		return &content, nil
	case "thinking":
		content := NewThinkingContentBlock(block.Thinking, block.Signature)
		return &content, nil
	case "redacted_thinking":
		content := ContentBlock{
			Type:     ContentBlockTypeRedactedThinking,
			Redacted: block.Data,
		}
		return &content, nil
	default:
		return nil, nil
	}
}

func anthropicDeltaToContentBlock(raw json.RawMessage) (*ContentBlock, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	var delta anthropicContentBlockDelta
	if err := json.Unmarshal(raw, &delta); err != nil {
		return nil, err
	}

	switch delta.Type {
	case "text_delta":
		content := NewTextContentBlock(delta.Text)
		return &content, nil
	case "input_json_delta":
		return &ContentBlock{
			Type: ContentBlockTypeToolUse,
			ToolUse: &ToolUse{
				Input: json.RawMessage(delta.PartialJSON),
			},
		}, nil
	case "thinking_delta":
		content := NewThinkingContentBlock(delta.Thinking, "")
		return &content, nil
	case "signature_delta":
		content := NewThinkingContentBlock("", delta.Signature)
		return &content, nil
	default:
		return nil, nil
	}
}

func mapAnthropicUsage(usage anthropicUsage) *Usage {
	return NewUsage(UsageOperationCompletion, TokenUsage{
		InputTokens:              usage.InputTokens,
		OutputTokens:             usage.OutputTokens,
		TotalTokens:              usage.InputTokens + usage.OutputTokens,
		CacheCreationInputTokens: usage.CacheCreationInputTokens,
		CacheReadInputTokens:     usage.CacheReadInputTokens,
		CacheBilledSeparately:    true,
		ServerToolUse:            cloneServerToolUseMap(usage.ServerToolUse),
	})
}

func stripDataPrefix(data string) string {
	if i := strings.Index(data, ","); i != -1 {
		return data[i+1:]
	}
	return data
}

func boolPtr(value bool) *bool {
	return &value
}

func anthropicServerToolResultToToolOutput(block *anthropicContentBlock) (*ToolOutput, error) {
	if block == nil {
		return nil, nil
	}
	output := &ToolOutput{
		ToolUseID:    block.ToolUseID,
		Name:         block.Name,
		Execution:    ToolExecutionModeServer,
		ProviderType: block.Type,
		ProviderData: append(json.RawMessage(nil), block.Content...),
	}
	content, err := anthropicServerToolOutputContent(block.Type, block.Content)
	if err != nil {
		return nil, err
	}
	output.Output = content
	return output, nil
}

func anthropicServerToolOutputContent(toolType string, raw json.RawMessage) (MessageContent, error) {
	switch toolType {
	case "web_fetch_tool_result":
		return anthropicWebFetchToolResultContent(raw)
	case "web_search_tool_result":
		return anthropicWebSearchToolResultContent(raw)
	default:
		return nil, nil
	}
}

func anthropicWebFetchToolResultContent(raw json.RawMessage) (MessageContent, error) {
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}

	contentValue, ok := payload["content"]
	if !ok {
		return nil, nil
	}

	documentPayload, ok := contentValue.(map[string]any)
	if !ok {
		return nil, nil
	}

	block, err := anthropicDocumentContentBlockFromMap(documentPayload)
	if err != nil || block == nil {
		return nil, err
	}
	return MessageContent{*block}, nil
}

func anthropicWebSearchToolResultContent(raw json.RawMessage) (MessageContent, error) {
	var payload any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}

	blocks := anthropicSearchResultBlocks(payload)
	if len(blocks) == 0 {
		return nil, nil
	}
	return blocks, nil
}

func anthropicSearchResultBlocks(payload any) MessageContent {
	switch value := payload.(type) {
	case []any:
		blocks := make(MessageContent, 0, len(value))
		for _, item := range value {
			blocks = append(blocks, anthropicSearchResultBlocks(item)...)
		}
		return blocks
	case map[string]any:
		title, _ := value["title"].(string)
		url, _ := value["url"].(string)
		text := firstNonEmptyString(
			value["text"],
			value["snippet"],
			value["summary"],
			value["description"],
			value["excerpt"],
		)
		if title != "" || url != "" || text != "" {
			block := NewDocumentContentBlockFromText(text, "text/plain", url)
			if block.Document != nil {
				block.Document.Title = title
			}
			return MessageContent{block}
		}

		blocks := MessageContent{}
		for _, item := range value {
			blocks = append(blocks, anthropicSearchResultBlocks(item)...)
		}
		return blocks
	default:
		return nil
	}
}

func anthropicDocumentContentBlockFromMap(payload map[string]any) (*ContentBlock, error) {
	sourceValue, ok := payload["source"].(map[string]any)
	if !ok {
		return nil, nil
	}

	sourceType, _ := sourceValue["type"].(string)
	mediaType, _ := sourceValue["media_type"].(string)
	title, _ := payload["title"].(string)
	contextValue, _ := payload["context"].(string)

	switch sourceType {
	case "text":
		text, _ := sourceValue["data"].(string)
		block := NewDocumentContentBlockFromText(text, mediaType, "")
		if block.Document != nil {
			block.Document.Title = title
			block.Document.Context = contextValue
		}
		return &block, nil
	case "base64":
		data, _ := sourceValue["data"].(string)
		block := NewDocumentContentBlockFromBase64(data, mediaType)
		if block.Document != nil {
			block.Document.Title = title
			block.Document.Context = contextValue
		}
		return &block, nil
	case "url":
		url, _ := sourceValue["url"].(string)
		block := NewDocumentContentBlockFromURL(url, mediaType)
		if block.Document != nil {
			block.Document.Title = title
			block.Document.Context = contextValue
		}
		return &block, nil
	default:
		return nil, nil
	}
}

func cloneServerToolUseMap(value map[string]int) map[string]int {
	if len(value) == 0 {
		return nil
	}
	cloned := make(map[string]int, len(value))
	for key, count := range value {
		cloned[key] = count
	}
	return cloned
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func firstNonEmptyString(values ...any) string {
	for _, value := range values {
		text, ok := value.(string)
		if ok && strings.TrimSpace(text) != "" {
			return text
		}
	}
	return ""
}

// claudeCodeIdentityPrompt is the canonical opener Anthropic expects on
// requests authenticated with a Claude-Code-scoped OAuth token. pi sends
// the identical string at anthropic.ts:905; the value is load-bearing
// for feature gating, so do not paraphrase.
const claudeCodeIdentityPrompt = "You are Claude Code, Anthropic's official CLI for Claude."

// buildSystemBlocks assembles the request's `system` array. For OAuth
// the identity block goes FIRST so it is recognised before any custom
// system prompt; the user's prompt becomes a separate block so the two
// can be reasoned about independently. cache_control lives on the LAST
// block so the cached prefix covers BOTH (identity + user prompt).
//
// For API-key auth there is no identity-prefix requirement; we still
// cache the single user-system block to capture the typically-large
// agent prompt.
func buildSystemBlocks(userPrompt string, useOAuth bool, ttl string) []anthropicTextBlock {
	userPrompt = strings.TrimSpace(userPrompt)
	if !useOAuth {
		if userPrompt == "" {
			return nil
		}
		return []anthropicTextBlock{{
			Type:         "text",
			Text:         userPrompt,
			CacheControl: ephemeralCache(ttl),
		}}
	}
	blocks := []anthropicTextBlock{{
		Type: "text",
		Text: claudeCodeIdentityPrompt,
	}}
	if userPrompt != "" {
		blocks = append(blocks, anthropicTextBlock{
			Type: "text",
			Text: userPrompt,
		})
	}
	// Cache_control on the LAST block — Anthropic caches the prefix UP
	// TO AND INCLUDING that block, so placing it on the tail captures
	// both the identity prefix and the user prompt in one breakpoint.
	blocks[len(blocks)-1].CacheControl = ephemeralCache(ttl)
	return blocks
}

// ephemeralCache builds the cache_control marker for an input block.
// ttl is "" for the default 5-minute tier or "1h" for the extended
// tier. Per pi, the 1h tier does NOT require a separate anthropic-beta
// header — it's GA on the native Anthropic API. Caching is still
// "ephemeral"; only the retention window changes.
func ephemeralCache(ttl string) *anthropicCacheControl {
	return &anthropicCacheControl{Type: "ephemeral", TTL: ttl}
}

// resolveCacheTTL maps llm.InferenceParams.CacheRetention onto the
// Anthropic ttl field. Empty / "default" / unrecognised values yield
// "" (5-minute tier); "long" yields "1h" (extended tier). pi performs
// the same translation at anthropic.ts:62.
//
// Note: pi also conditions 1h on a per-model compat flag
// (supportsLongCacheRetention). On native Anthropic API every model
// supports it; the flag exists primarily to opt OUT for Fireworks-
// proxied Anthropic models. Our anthropic client only runs against
// the native endpoint when tokenSource != nil OR x-api-key auth is
// used, so we don't need the flag yet. If we ever add a Fireworks
// Anthropic path here, gate the "long" → "1h" upgrade behind a model
// allowlist.
func resolveCacheTTL(retention string) string {
	switch strings.ToLower(strings.TrimSpace(retention)) {
	case "long":
		return "1h"
	default:
		return ""
	}
}

// markLastToolCached stamps cache_control on the last tool definition.
// Tools are `map[string]any` rather than a typed struct (the schema is
// freeform) so we mutate the map in place — pi does the same at
// anthropic.ts:1185.
func markLastToolCached(tools []any, ttl string) {
	if len(tools) == 0 {
		return
	}
	last, ok := tools[len(tools)-1].(map[string]any)
	if !ok {
		return
	}
	cc := map[string]any{"type": "ephemeral"}
	if ttl != "" {
		cc["ttl"] = ttl
	}
	last["cache_control"] = cc
}

// markLastMessageCached places cache_control on the last content block
// of the last message. This is the load-bearing call for multi-turn
// efficiency: it tells Anthropic that the entire conversation history
// up through this message is a stable prefix that the NEXT turn will
// reuse verbatim. Without it cache hits drop sharply after turn 1.
func markLastMessageCached(messages []anthropicMessageParam, ttl string) {
	if len(messages) == 0 {
		return
	}
	last := &messages[len(messages)-1]
	if len(last.Content) == 0 {
		return
	}
	last.Content[len(last.Content)-1].CacheControl = ephemeralCache(ttl)
}

func (c *AnthropicClient) requestHeaders(ctx context.Context, request *anthropicMessageRequest) (map[string]string, error) {
	headers := map[string]string{
		"anthropic-version": c.apiVersion,
	}
	// Beta-features list assembled per request. pi appends features
	// conditionally and joins with commas at anthropic.ts:851. We do
	// the same so models that don't need a particular beta don't get
	// it as overhead. Fine-grained-tool-streaming is the only feature
	// we condition on request shape today; others can be added here.
	betas := []string{}
	if request != nil && len(request.Tools) > 0 {
		// pi gates this on a per-model compat flag
		// (model.compat.supportsEagerToolInputStreaming). Native
		// Anthropic models all leave that flag falsy in pi's models
		// table, so the effective rule is "send the beta whenever
		// tools are present." We mirror that — if we ever route an
		// Anthropic call through Fireworks/Bedrock where eager
		// streaming IS supported natively, we'd need a per-model
		// allowlist here. pi reference: anthropic.ts:1161.
		betas = append(betas, "fine-grained-tool-streaming-2025-05-14")
	}
	if c.tokenSource != nil {
		token, err := c.tokenSource.Token()
		if err != nil {
			return nil, err
		}
		if token == nil || token.AccessToken == "" {
			return nil, fmt.Errorf("anthropic oauth token source returned no access token")
		}
		headers["authorization"] = "Bearer " + token.AccessToken
		// OAuth-specific betas come FIRST so Anthropic's gateway
		// recognises the Claude-Code identity surface before applying
		// feature betas. pi enforces the same order: claude-code-* +
		// oauth-* + dynamic features (anthropic.ts:851).
		oauthBetas := append([]string{"claude-code-20250219", "oauth-2025-04-20"}, betas...)
		headers["anthropic-beta"] = strings.Join(oauthBetas, ",")
		headers["user-agent"] = "claude-cli/2.1.75"
		headers["x-app"] = "cli"
		_ = ctx
		return headers, nil
	}
	if len(betas) > 0 {
		headers["anthropic-beta"] = strings.Join(betas, ",")
	}
	headers["x-api-key"] = c.apiKey
	return headers, nil
}
