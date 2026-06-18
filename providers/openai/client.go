package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	. "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/oauthx"
	"github.com/proiceremo/ai-sdk/providers/internal/httpx"
	"golang.org/x/oauth2"
)

const defaultOpenAIBaseURL = "https://api.openai.com/v1"

type OpenAIClient struct {
	client      *http.Client
	apiKey      string
	baseURL     string
	headers     map[string]string
	tokenSource oauth2.TokenSource
}

func (c *OpenAIClient) SupportsStreaming() bool { return true }

type RequestOption interface {
	apply(*OpenAIClient)
}

type requestOptionFunc func(*OpenAIClient)

func (f requestOptionFunc) apply(client *OpenAIClient) {
	f(client)
}

func WithAPIKey(apiKey string) RequestOption {
	return requestOptionFunc(func(client *OpenAIClient) {
		client.apiKey = apiKey
	})
}

func WithBaseURL(baseURL string) RequestOption {
	return requestOptionFunc(func(client *OpenAIClient) {
		client.baseURL = baseURL
	})
}

func WithHTTPClient(httpClient *http.Client) RequestOption {
	return requestOptionFunc(func(client *OpenAIClient) {
		client.client = httpClient
	})
}

func WithHeader(key, value string) RequestOption {
	return requestOptionFunc(func(client *OpenAIClient) {
		if client.headers == nil {
			client.headers = map[string]string{}
		}
		client.headers[key] = value
	})
}

func WithTokenSource(source oauth2.TokenSource) RequestOption {
	return requestOptionFunc(func(client *OpenAIClient) {
		client.tokenSource = source
	})
}

func WithOrganization(organization string) RequestOption {
	return WithHeader("OpenAI-Organization", organization)
}

func WithProject(project string) RequestOption {
	return WithHeader("OpenAI-Project", project)
}

func NewOpenAIClient(apiKey string, baseURL string) *OpenAIClient {
	return NewOpenAIClientWithOptions(
		WithAPIKey(apiKey),
		WithBaseURL(baseURL),
	)
}

func NewOpenAIClientWithOptions(options ...RequestOption) *OpenAIClient {
	client := &OpenAIClient{
		client:  &http.Client{Timeout: 10 * time.Minute},
		baseURL: defaultOpenAIBaseURL,

		headers: map[string]string{},
	}

	for _, option := range options {
		if option != nil {
			option.apply(client)
		}
	}

	if client.client == nil {
		client.client = &http.Client{Timeout: 10 * time.Minute}
	}
	if client.baseURL == "" {
		client.baseURL = defaultOpenAIBaseURL
	}
	client.baseURL = strings.TrimRight(client.baseURL, "/")

	return client
}

func (c *OpenAIClient) requestHeaders(ctx context.Context) (map[string]string, error) {
	headers := map[string]string{}
	if c.tokenSource != nil {
		token, err := c.tokenSource.Token()
		if err != nil {
			return nil, err
		}
		if token != nil && token.AccessToken != "" {
			headers["Authorization"] = fmt.Sprintf("Bearer %s", token.AccessToken)
		}
	} else if c.apiKey != "" {
		headers["Authorization"] = fmt.Sprintf("Bearer %s", c.apiKey)
	}
	for key, value := range c.headers {
		headers[key] = value
	}
	_ = ctx
	return headers, nil
}

func (c *OpenAIClient) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	input := strings.TrimSpace(req.Inputs.Text())
	if input == "" {
		return nil, fmt.Errorf("embedding input must contain text for openai embeddings")
	}

	request := openAIEmbeddingRequest{
		Model:          req.Model,
		Input:          []string{input},
		EncodingFormat: "float",
	}
	if req.Dimensions != nil {
		request.Dimensions = req.Dimensions
	}

	var response openAIEmbeddingResponse
	headers, err := c.requestHeaders(ctx)
	if err != nil {
		return nil, err
	}
	if err := httpx.DoJSON(
		ctx,
		c.client,
		http.MethodPost,
		c.baseURL+"/embeddings",
		headers,
		request,
		&response,
	); err != nil {
		return nil, fmt.Errorf("failed to create embeddings: %w", err)
	}

	if len(response.Data) == 0 {
		return nil, fmt.Errorf("openai embeddings returned no data")
	}

	return &EmbeddingResponse{
		Model:      response.Model,
		Embeddings: float64SliceToFloat32(response.Data[0].Embedding),
		Usage: NewUsage(UsageOperationEmbedding, TokenUsage{
			InputTokens: response.Usage.PromptTokens,
			TotalTokens: response.Usage.TotalTokens,
			InputTokensDetails: &UsageTokenDetails{
				TextTokens: response.Usage.PromptTokens,
			},
		}),
	}, nil
}

func (c *OpenAIClient) buildCompletionParams(messages []Message, params InferenceParams) (*openAIChatCompletionRequest, error) {
	openAIMessages, err := c.toOpenAIMessages(messages, params.SystemPrompt)
	if err != nil {
		return nil, err
	}

	request := &openAIChatCompletionRequest{
		Model:       params.Model,
		Messages:    openAIMessages,
		Temperature: params.Temperature,
		MaxTokens:   params.MaxTokens,
		TopP:        params.TopP,
		ServiceTier: params.ServiceTier,
	}

	if len(params.Tools) > 0 {
		request.Tools = make([]openAIChatCompletionTool, 0, len(params.Tools))
		for _, tool := range params.Tools {
			schema := tool.Schema()
			function := openAIChatCompletionFunctionDefinition{
				Name: schema.Name,
			}
			if schema.Description != "" {
				function.Description = Ptr(schema.Description)
			}
			if schema.InputSchema != nil {
				function.Parameters = schema.InputSchema
			}
			if schema.Strict {
				function.Strict = Ptr(true)
				if function.Parameters == nil {
					function.Parameters = map[string]any{
						"type":                 "object",
						"properties":           map[string]any{},
						"additionalProperties": false,
					}
				}
			}
			request.Tools = append(request.Tools, openAIChatCompletionTool{
				Type:     "function",
				Function: &function,
			})
		}
	}
	if len(request.Tools) == 0 {
		request.Tools = nil
	}
	if params.ToolChoice != nil {
		toolChoice, err := toOpenAIToolChoice(*params.ToolChoice)
		if err != nil {
			return nil, err
		}
		request.ToolChoice = toolChoice
	}

	if params.ResponseFormat != nil {
		request.ResponseFormat = toOpenAIResponseFormat(*params.ResponseFormat)
	}

	return request, nil
}

func toOpenAIToolChoice(choice ToolChoice) (*openAIChatCompletionToolChoice, error) {
	switch choice.Type {
	case "", ToolChoiceTypeAuto:
		return &openAIChatCompletionToolChoice{Type: "auto"}, nil
	case ToolChoiceTypeRequired:
		return &openAIChatCompletionToolChoice{Type: "required"}, nil
	case ToolChoiceTypeTool:
		if strings.TrimSpace(choice.Name) == "" {
			return nil, fmt.Errorf("openai named tool choice requires a tool name")
		}
		return &openAIChatCompletionToolChoice{
			Type: "function",
			Function: &openAIChatCompletionToolChoiceFunctionRef{
				Name: choice.Name,
			},
		}, nil
	default:
		return nil, fmt.Errorf("unsupported openai tool choice type %q", choice.Type)
	}
}

func (c *OpenAIClient) buildStreamingCompletionParams(messages []Message, params InferenceParams) (*openAIChatCompletionRequest, error) {
	request, err := c.buildCompletionParams(messages, params)
	if err != nil {
		return nil, err
	}
	request.Stream = true
	request.StreamOptions = &openAIChatCompletionStreamOptions{
		IncludeUsage: true,
	}
	return request, nil
}

func (c *OpenAIClient) toOpenAIMessages(messages []Message, systemPrompt string) ([]openAIChatCompletionRequestMessage, error) {
	result := make([]openAIChatCompletionRequestMessage, 0, len(messages)+1)

	if systemPrompt != "" {
		result = append(result, openAIChatCompletionRequestMessage{
			Role:    "system",
			Content: SanitizeSurrogates(systemPrompt),
		})
	}

	for _, message := range messages {
		if len(message.Content) == 0 {
			continue
		}

		if toolResultOnly(message.Content) {
			toolMessages, err := c.toOpenAIToolMessages(message.Content)
			if err != nil {
				return nil, err
			}
			result = append(result, toolMessages...)
			continue
		}

		switch message.Role {
		case MessageRoleUser:
			userMessages, err := c.toOpenAIUserMessages(message.Content)
			if err != nil {
				return nil, err
			}
			result = append(result, userMessages...)
		case MessageRoleAssistant:
			assistantMessage, err := c.toOpenAIAssistantMessage(message.Content)
			if err != nil {
				return nil, err
			}
			if assistantMessage.Role != "" {
				result = append(result, assistantMessage)
			}
		default:
			return nil, fmt.Errorf("unsupported message role %q", message.Role)
		}
	}

	return result, nil
}

func (c *OpenAIClient) toOpenAIUserMessages(content MessageContent) ([]openAIChatCompletionRequestMessage, error) {
	result := make([]openAIChatCompletionRequestMessage, 0, len(content))
	pending := make(MessageContent, 0, len(content))

	flushPending := func() error {
		if len(pending) == 0 {
			return nil
		}
		payload, err := c.toOpenAIUserContent(pending)
		if err != nil {
			return err
		}
		result = append(result, openAIChatCompletionRequestMessage{
			Role:    "user",
			Content: payload,
		})
		pending = pending[:0]
		return nil
	}

	for _, block := range content {
		if block.Type == ContentBlockTypeToolResult {
			if err := flushPending(); err != nil {
				return nil, err
			}
			toolMessages, err := c.toOpenAIToolMessages(MessageContent{block})
			if err != nil {
				return nil, err
			}
			result = append(result, toolMessages...)
			continue
		}
		pending = append(pending, block)
	}

	if err := flushPending(); err != nil {
		return nil, err
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("user message has no supported content")
	}
	return result, nil
}

func toolResultOnly(content MessageContent) bool {
	if len(content) == 0 {
		return false
	}
	for _, block := range content {
		if block.Type != ContentBlockTypeToolResult {
			return false
		}
	}
	return true
}

func (c *OpenAIClient) toOpenAIToolMessages(content MessageContent) ([]openAIChatCompletionRequestMessage, error) {
	result := make([]openAIChatCompletionRequestMessage, 0, len(content))

	for _, block := range content {
		if block.Type != ContentBlockTypeToolResult || block.ToolOutput == nil {
			return nil, fmt.Errorf("tool result message contains unsupported block type %q", block.Type)
		}

		outputText := strings.TrimSpace(block.ToolOutput.Output.Text())
		if outputText == "" {
			outputText = strings.TrimSpace(block.ToolOutput.Output.ToString())
		}
		outputText = SanitizeSurrogates(outputText)

		result = append(result, openAIChatCompletionRequestMessage{
			Role:       "tool",
			ToolCallID: block.ToolOutput.ToolUseID,
			Content:    outputText,
		})

		rich := richToolResultSidecarContent(*block.ToolOutput)
		if len(rich) > 0 {
			sidecar := append(MessageContent{
				NewTextContentBlock(fmt.Sprintf("Rich content returned by tool call %s (%s):", block.ToolOutput.ToolUseID, block.ToolOutput.Name)),
			}, rich...)
			payload, err := c.toOpenAIUserContent(sidecar)
			if err != nil {
				return nil, err
			}
			result = append(result, openAIChatCompletionRequestMessage{
				Role:    "user",
				Content: payload,
			})
		}
	}

	return result, nil
}

func richToolResultSidecarContent(output ToolOutput) MessageContent {
	var rich MessageContent
	for _, block := range output.Output {
		switch block.Type {
		case ContentBlockTypeImage, ContentBlockTypeAudio, ContentBlockTypeVideo:
			rich = append(rich, block)
		case ContentBlockTypeDocument:
			if block.Document != nil && block.Document.Type != DocumentSourceTypeText {
				rich = append(rich, block)
			}
		}
	}
	if output.Metadata != nil {
		for _, item := range output.Metadata.Content {
			if item.Content == nil {
				continue
			}
			block := item.Content.Block
			switch block.Type {
			case ContentBlockTypeImage, ContentBlockTypeAudio, ContentBlockTypeVideo:
				rich = append(rich, block)
			case ContentBlockTypeDocument:
				if block.Document != nil && block.Document.Type != DocumentSourceTypeText {
					rich = append(rich, block)
				}
			}
		}
	}
	return dedupeContentBlocks(rich)
}

func dedupeContentBlocks(content MessageContent) MessageContent {
	seen := map[string]bool{}
	var out MessageContent
	for _, block := range content {
		data, _ := json.Marshal(block)
		key := string(data)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, block)
	}
	return out
}

func (c *OpenAIClient) toOpenAIUserContent(content MessageContent) (any, error) {
	var textOnly strings.Builder
	isTextOnly := true
	parts := make([]any, 0, len(content))

	for _, block := range content {
		switch block.Type {
		case ContentBlockTypeText:
			if block.Text != "" {
				sanitized := SanitizeSurrogates(block.Text)
				textOnly.WriteString(sanitized)
				parts = append(parts, openAIChatCompletionTextPart{
					Type: "text",
					Text: sanitized,
				})
			}
		case ContentBlockTypeImage:
			isTextOnly = false
			part, err := openAIImagePart(block.Image)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case ContentBlockTypeAudio:
			isTextOnly = false
			part, err := openAIAudioPart(block.Audio)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case ContentBlockTypeDocument:
			isTextOnly = false
			documentParts, err := openAIDocumentParts(block.Document)
			if err != nil {
				return nil, err
			}
			parts = append(parts, documentParts...)
		default:
			return nil, fmt.Errorf("cannot convert user content block type %q to openai chat content", block.Type)
		}
	}

	if len(parts) == 0 {
		return nil, fmt.Errorf("user message has no supported content")
	}

	// Use a plain string for text-only messages. Most OpenAI-compatible
	// providers accept the structured array format only when non-text blocks
	// (images, audio, documents) are present. Sending a single-element or
	// multi-element text array for a text-only message breaks providers that
	// strictly follow the simpler string-content variant of the API.
	if isTextOnly {
		return textOnly.String(), nil
	}
	return parts, nil
}

func openAIImagePart(source *ImageSource) (openAIChatCompletionImagePart, error) {
	if source == nil {
		return openAIChatCompletionImagePart{}, fmt.Errorf("image content is missing payload")
	}

	url, err := openAIImageURL(source)
	if err != nil {
		return openAIChatCompletionImagePart{}, err
	}

	return openAIChatCompletionImagePart{
		Type: "image_url",
		ImageURL: openAIChatCompletionImageURL{
			URL: url,
		},
	}, nil
}

func openAIAudioPart(source *AudioSource) (openAIChatCompletionInputAudioPart, error) {
	if source == nil {
		return openAIChatCompletionInputAudioPart{}, fmt.Errorf("audio content is missing payload")
	}
	if source.Type != AudioSourceTypeBase64 {
		return openAIChatCompletionInputAudioPart{}, fmt.Errorf("audio source type %q is not supported by openai chat completions", source.Type)
	}

	mediaType := openAIMediaType(source.MediaType, source.Data)
	format, err := openAIInputAudioFormat(mediaType)
	if err != nil {
		return openAIChatCompletionInputAudioPart{}, err
	}

	return openAIChatCompletionInputAudioPart{
		Type: "input_audio",
		InputAudio: openAIChatCompletionInputAudio{
			Data:   openAIBase64Payload(source.Data),
			Format: format,
		},
	}, nil
}

func openAIDocumentParts(source *DocumentSource) ([]any, error) {
	if source == nil {
		return nil, fmt.Errorf("document content is missing payload")
	}

	parts := make([]any, 0, 2)
	if text := openAIDocumentText(source); text != "" {
		parts = append(parts, openAIChatCompletionTextPart{
			Type: "text",
			Text: text,
		})
	}

	switch source.Type {
	case DocumentSourceTypeText:
		if len(parts) == 0 {
			return nil, fmt.Errorf("document text content is empty")
		}
	case DocumentSourceTypeBase64:
		if strings.TrimSpace(source.Data) != "" {
			fileData := openAIBase64Payload(source.Data)
			part := openAIChatCompletionFilePart{Type: "file"}
			part.File.FileData = &fileData
			if source.Name != "" {
				part.File.Filename = Ptr(source.Name)
			}
			parts = append(parts, part)
		}
		if len(parts) == 0 {
			return nil, fmt.Errorf("document base64 content is empty")
		}
	case DocumentSourceTypeURL:
		if len(parts) == 0 {
			return nil, fmt.Errorf("document urls are not supported by openai chat completions without extracted text")
		}
	default:
		return nil, fmt.Errorf("unsupported document source type %q", source.Type)
	}

	return parts, nil
}

func openAIDocumentText(source *DocumentSource) string {
	if source == nil {
		return ""
	}

	parts := make([]string, 0, 3)
	for _, part := range []string{source.Title, source.Context, source.Text} {
		part = strings.TrimSpace(part)
		if part != "" {
			parts = append(parts, part)
		}
	}

	return strings.Join(parts, "\n")
}

func (c *OpenAIClient) toOpenAIAssistantMessage(content MessageContent) (openAIChatCompletionRequestMessage, error) {
	var text strings.Builder
	var reasoning strings.Builder
	toolCalls := make([]openAIChatCompletionToolCall, 0)

	for _, block := range content {
		switch block.Type {
		case ContentBlockTypeText:
			text.WriteString(SanitizeSurrogates(block.Text))
		case ContentBlockTypeThinking:
			reasoning.WriteString(SanitizeSurrogates(block.Thinking))
		case ContentBlockTypeToolUse:
			if block.ToolUse == nil {
				return openAIChatCompletionRequestMessage{}, fmt.Errorf("assistant tool use content is missing payload")
			}
			toolType := strings.TrimSpace(block.ToolUse.ProviderType)
			if toolType == "" {
				toolType = "function"
			}
			toolCalls = append(toolCalls, openAIChatCompletionToolCall{
				ID:   block.ToolUse.ID,
				Type: toolType,
				Function: openAIChatCompletionFunctionCall{
					Name:      block.ToolUse.Name,
					Arguments: string(block.ToolUse.Input),
				},
			})
		default:
			return openAIChatCompletionRequestMessage{}, fmt.Errorf("cannot convert assistant content block type %q to openai chat message", block.Type)
		}
	}

	if text.Len() == 0 && len(toolCalls) == 0 {
		return openAIChatCompletionRequestMessage{}, nil
	}

	message := openAIChatCompletionRequestMessage{
		Role: "assistant",
	}
	if text.Len() > 0 {
		message.Content = text.String()
	}
	if reasoning.Len() > 0 {
		message.ReasoningContent = reasoning.String()
	}
	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
	}

	return message, nil
}

func openAIImageURL(source *ImageSource) (string, error) {
	if source == nil {
		return "", fmt.Errorf("image content is missing payload")
	}

	switch source.Type {
	case ImageSourceTypeURL:
		if source.URL == "" {
			return "", fmt.Errorf("image url is required")
		}
		return source.URL, nil
	case ImageSourceTypeBase64:
		if source.Data == "" {
			return "", fmt.Errorf("image data is required")
		}
		if strings.HasPrefix(source.Data, "data:") {
			return source.Data, nil
		}
		mediaType := openAIMediaType(source.MediaType, source.Data)
		if mediaType == "" {
			mediaType = "image/png"
		}
		return fmt.Sprintf("data:%s;base64,%s", mediaType, source.Data), nil
	default:
		return "", fmt.Errorf("unsupported image source type %q", source.Type)
	}
}

func openAIMediaType(mediaType, data string) string {
	if mediaType != "" {
		return mediaType
	}
	if !strings.HasPrefix(data, "data:") {
		return ""
	}

	rest := strings.TrimPrefix(data, "data:")
	header, _, found := strings.Cut(rest, ",")
	if !found {
		return ""
	}
	parsed, _, _ := strings.Cut(header, ";")
	return parsed
}

func openAIBase64Payload(data string) string {
	if !strings.HasPrefix(data, "data:") {
		return data
	}
	_, payload, found := strings.Cut(data, ",")
	if !found {
		return data
	}
	return payload
}

func openAIInputAudioFormat(mediaType string) (string, error) {
	mediaType = strings.ToLower(strings.TrimSpace(mediaType))
	switch {
	case strings.HasSuffix(mediaType, "/mp3"), strings.HasSuffix(mediaType, "/mpeg"):
		return "mp3", nil
	case strings.HasSuffix(mediaType, "/wav"), strings.HasSuffix(mediaType, "/x-wav"), strings.HasSuffix(mediaType, "/wave"):
		return "wav", nil
	default:
		return "", fmt.Errorf("cannot convert audio media type %q to openai input audio format", mediaType)
	}
}

func toOpenAIResponseFormat(format ResponseFormat) *openAIChatCompletionResponseFormat {
	switch format.Type {
	case ResponseFormatTypeText:
		return &openAIChatCompletionResponseFormat{Type: "text"}
	case ResponseFormatTypeJSONObject:
		if format.JSONSchema != nil {
			return &openAIChatCompletionResponseFormat{
				Type: "json_schema",
				JSONSchema: &openAIChatCompletionJSONSchema{
					Name:   "structured_output",
					Schema: format.JSONSchema,
				},
			}
		}
		return &openAIChatCompletionResponseFormat{Type: "json_object"}
	default:
		return nil
	}
}

func (c *OpenAIClient) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	if err := ValidateMessages(messages); err != nil {
		return nil, err
	}

	request, err := c.buildCompletionParams(messages, params)
	if err != nil {
		return nil, err
	}

	var response openAIChatCompletionResponse
	headers, err := c.requestHeaders(ctx)
	if err != nil {
		return nil, err
	}
	if err := httpx.DoJSON(
		ctx,
		c.client,
		http.MethodPost,
		c.baseURL+"/chat/completions",
		headers,
		request,
		&response,
	); err != nil {
		return nil, err
	}
	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("openai completion returned no choices")
	}

	return c.fromOpenAIResponse(&response, request.ServiceTier), nil
}

func (c *OpenAIClient) fromOpenAIResponse(response *openAIChatCompletionResponse, requestServiceTier *string) *Message {
	if response == nil || len(response.Choices) == 0 {
		return nil
	}

	choice := response.Choices[0]
	content := make(MessageContent, 0, len(choice.Message.ToolCalls)+2)

	if reasoningBlock := openAIReasoningContentBlock(choice.Message.ReasoningContent, choice.Message.Reasoning); reasoningBlock != nil {
		content = append(content, *reasoningBlock)
	}

	if choice.Message.Content != "" {
		content = append(content, ContentBlock{
			Type: ContentBlockTypeText,
			Text: choice.Message.Content,
		})
	}

	for _, toolCall := range choice.Message.ToolCalls {
		var raw json.RawMessage
		if toolCall.Function.Arguments != "" {
			raw = json.RawMessage(toolCall.Function.Arguments)
		}
		execution := ToolExecutionModeClient
		if toolCall.Type != "" && toolCall.Type != "function" {
			execution = ToolExecutionModeServer
		}
		content = append(content, ContentBlock{
			Type: ContentBlockTypeToolUse,
			ToolUse: &ToolUse{
				ID:           toolCall.ID,
				Name:         toolCall.Function.Name,
				Input:        raw,
				Execution:    execution,
				ProviderType: toolCall.Type,
			},
		})
	}

	return &Message{
		Role:       MessageRoleAssistant,
		Content:    content,
		Timestamp:  time.Now(),
		StopReason: NormaliseStopReason(APIFormatOpenAI, choice.FinishReason),
		Usage:      mapOpenAIUsage(response.Usage, response.ServiceTier, requestServiceTier),
	}
}

func (c *OpenAIClient) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	if err := ValidateMessages(messages); err != nil {
		return nil, err
	}
	request, err := c.buildStreamingCompletionParams(messages, params)
	if err != nil {
		return nil, err
	}

	headers, err := c.requestHeaders(ctx)
	if err != nil {
		return nil, err
	}
	stream, err := httpx.DoSSE(
		ctx,
		c.client,
		http.MethodPost,
		c.baseURL+"/chat/completions",
		headers,
		request,
	)
	if err != nil {
		return nil, err
	}

	return newOpenAIStream(stream, request.ServiceTier), nil
}

type openAIToolCallState struct {
	id        string
	name      string
	toolType  string
	args      strings.Builder
	finalized bool
}

type openaiStream struct {
	stream       *httpx.SSEStream
	accumulated  Message
	pending      []*StreamEvent
	finished     bool
	usage        *Usage
	stopReason   StopReason
	toolCalls    map[int]*openAIToolCallState
	toolOrder    []int
	toolIDPrefix string
	requestTier  *string
}

func newOpenAIStream(stream *httpx.SSEStream, requestTier *string) *openaiStream {
	return &openaiStream{
		stream:       stream,
		toolCalls:    map[int]*openAIToolCallState{},
		toolIDPrefix: fmt.Sprintf("t-%d", time.Now().UnixNano()),
		accumulated: Message{
			Role:      MessageRoleAssistant,
			Timestamp: time.Now(),
			Content:   MessageContent{},
		},
		requestTier: requestTier,
	}
}

func (s *openaiStream) Recv(ctx context.Context) (*StreamEvent, error) {
	_ = ctx

	if event := s.popEvent(); event != nil {
		return event, nil
	}
	if s.finished {
		return nil, io.EOF
	}

	for s.stream.Next() {
		rawEvent := s.stream.Current()
		data := strings.TrimSpace(string(rawEvent.Data))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			s.finish()
			return s.popEventOrEOF()
		}

		var chunk openAIChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			s.finished = true
			return nil, err
		}
		if err := s.processChunk(chunk); err != nil {
			s.finished = true
			return nil, err
		}
		if event := s.popEvent(); event != nil {
			return event, nil
		}
	}

	if err := s.stream.Err(); err != nil {
		s.finished = true
		return nil, err
	}

	s.finish()
	return s.popEventOrEOF()
}

func (s *openaiStream) processChunk(chunk openAIChatCompletionChunk) error {
	if len(chunk.Choices) > 0 {
		choice := chunk.Choices[0]

		if reasoning := openAIReasoningText(choice.Delta.ReasoningContent, choice.Delta.Reasoning); reasoning != "" {
			s.appendReasoningDelta(reasoning)
		}
		if choice.Delta.Content != nil && *choice.Delta.Content != "" {
			s.appendTextDelta(*choice.Delta.Content)
		}
		if len(choice.Delta.ToolCalls) > 0 {
			if err := s.appendToolCallDeltas(choice.Delta.ToolCalls); err != nil {
				return err
			}
		}
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			s.stopReason = NormaliseStopReason(APIFormatOpenAI, *choice.FinishReason)
			s.accumulated.StopReason = s.stopReason
			s.finalizeToolCalls()
		}
	}

	if chunk.Usage != nil {
		s.usage = mapOpenAIUsage(chunk.Usage, chunk.ServiceTier, s.requestTier)
		s.accumulated.Usage = s.usage
		s.finish()
	}

	return nil
}

func (s *openaiStream) appendTextDelta(text string) {
	if len(s.accumulated.Content) == 0 || s.accumulated.Content[len(s.accumulated.Content)-1].Type != ContentBlockTypeText {
		s.accumulated.Content = append(s.accumulated.Content, ContentBlock{
			Type: ContentBlockTypeText,
			Text: text,
		})
	} else {
		s.accumulated.Content[len(s.accumulated.Content)-1].Text += text
	}

	s.pending = append(s.pending, &StreamEvent{
		Type: EventTypeContentDelta,
		Delta: MessageDelta{
			Content: MessageContent{{
				Type: ContentBlockTypeText,
				Text: text,
			}},
		},
		Snapshot: s.accumulated,
	})
}

func (s *openaiStream) appendReasoningDelta(reasoning string) {
	if len(s.accumulated.Content) == 0 || s.accumulated.Content[len(s.accumulated.Content)-1].Type != ContentBlockTypeThinking {
		s.accumulated.Content = append(s.accumulated.Content, NewThinkingContentBlock(reasoning, ""))
	} else {
		s.accumulated.Content[len(s.accumulated.Content)-1].Thinking += reasoning
	}

	s.pending = append(s.pending, &StreamEvent{
		Type: EventTypeContentDelta,
		Delta: MessageDelta{
			Content: MessageContent{NewThinkingContentBlock(reasoning, "")},
		},
		Snapshot: s.accumulated,
	})
}

func (s *openaiStream) appendToolCallDeltas(toolCalls []openAIChatCompletionChunkToolCall) error {
	for _, toolCall := range toolCalls {
		state, ok := s.toolCalls[toolCall.Index]
		if !ok {
			state = &openAIToolCallState{}
			s.toolCalls[toolCall.Index] = state
			s.toolOrder = append(s.toolOrder, toolCall.Index)
		}

		if toolCall.ID != nil && *toolCall.ID != "" {
			state.id = *toolCall.ID
		}
		if toolCall.Type != nil && *toolCall.Type != "" {
			state.toolType = *toolCall.Type
		}
		if toolCall.Function != nil {
			if toolCall.Function.Name != nil && *toolCall.Function.Name != "" {
				state.name = *toolCall.Function.Name
			}
			if toolCall.Function.Arguments != nil {
				state.args.WriteString(*toolCall.Function.Arguments)
			}
		}
	}

	return nil
}

func (s *openaiStream) finalizeToolCalls() {
	for _, index := range s.toolOrder {
		state := s.toolCalls[index]
		if state == nil || state.finalized {
			continue
		}

		id := state.id
		if id == "" {
			id = fmt.Sprintf("%s-toolcall-%d", s.toolIDPrefix, index)
		}

		var raw json.RawMessage
		if args := state.args.String(); args != "" {
			raw = json.RawMessage(args)
		}

		blockIndex := index
		block := ContentBlock{
			Type: ContentBlockTypeToolUse,
			ToolUse: &ToolUse{
				ID:           id,
				Name:         state.name,
				Input:        raw,
				Index:        &blockIndex,
				Execution:    openAIToolExecutionMode(state.toolType),
				ProviderType: state.toolType,
			},
		}
		s.accumulated.Content = append(s.accumulated.Content, block)
		s.pending = append(s.pending, &StreamEvent{
			Type: EventTypeContentStart,
			Delta: MessageDelta{
				Content: MessageContent{block},
			},
			Snapshot: s.accumulated,
		})
		state.finalized = true
	}
}

func openAIToolExecutionMode(toolType string) ToolExecutionMode {
	if strings.TrimSpace(toolType) == "" || toolType == "function" {
		return ToolExecutionModeClient
	}
	return ToolExecutionModeServer
}

func (s *openaiStream) finish() {
	if s.finished {
		return
	}

	s.finalizeToolCalls()
	if s.usage != nil {
		s.accumulated.Usage = s.usage
	}
	if s.stopReason != "" {
		s.accumulated.StopReason = s.stopReason
	}

	s.pending = append(s.pending, &StreamEvent{
		Type: EventTypeMessageEnd,
		Delta: MessageDelta{
			StopReason: s.accumulated.StopReason,
			Usage:      s.accumulated.Usage,
		},
		Snapshot: s.accumulated,
	})
	s.finished = true
}

func (s *openaiStream) popEvent() *StreamEvent {
	if len(s.pending) == 0 {
		return nil
	}
	event := s.pending[0]
	s.pending = s.pending[1:]
	return event
}

func (s *openaiStream) popEventOrEOF() (*StreamEvent, error) {
	if event := s.popEvent(); event != nil {
		return event, nil
	}
	return nil, io.EOF
}

func (s *openaiStream) Current() *Message {
	return &s.accumulated
}

func (s *openaiStream) Close() error {
	s.finish()
	if s.stream == nil {
		return nil
	}
	return s.stream.Close()
}

func resolveOpenAIServiceTier(responseTier *string, requestTier *string) string {
	if responseTier != nil && *responseTier != "" {
		return *responseTier
	}
	if requestTier != nil && *requestTier != "" {
		return *requestTier
	}
	return ""
}

func mapOpenAIUsage(usage *openAICompletionUsage, responseTier *string, requestTier *string) *Usage {
	if usage == nil {
		return nil
	}

	tokens := TokenUsage{
		InputTokens:  usage.PromptTokens,
		OutputTokens: usage.CompletionTokens,
		TotalTokens:  usage.TotalTokens,
		ServiceTier:  resolveOpenAIServiceTier(responseTier, requestTier),
	}
	if details := usage.PromptTokensDetails; details != nil {
		tokens.CacheReadInputTokens = details.CachedTokens
		// OpenRouter-style cache-write count. Real OpenAI never sets
		// this; OpenRouter-proxied providers do. We mirror it onto
		// CacheCreationInputTokens so the existing accounting plumbing
		// (Anthropic cache writes are billed against the same field)
		// picks it up automatically.
		tokens.CacheCreationInputTokens = details.CacheWriteTokens
		if details.AudioTokens > 0 {
			tokens.InputTokensDetails = &UsageTokenDetails{AudioTokens: details.AudioTokens}
		}
	}
	if details := usage.CompletionTokensDetails; details != nil {
		// Reasoning tokens are a sub-count of OutputTokens, NOT an
		// additional bucket — don't add to OutputTokens. Surfacing the
		// sub-count lets dashboards show "X% reasoning."
		tokens.ReasoningOutputTokens = details.ReasoningTokens
		if details.AudioTokens > 0 {
			tokens.OutputTokensDetails = &UsageTokenDetails{AudioTokens: details.AudioTokens}
		}
	}

	return NewUsage(UsageOperationCompletion, tokens)
}

func openAIReasoningContentBlock(primary, fallback json.RawMessage) *ContentBlock {
	reasoning := openAIReasoningText(primary, fallback)
	if reasoning == "" {
		return nil
	}
	block := NewThinkingContentBlock(reasoning, "")
	return &block
}

func openAIReasoningText(raws ...json.RawMessage) string {
	for _, raw := range raws {
		if reasoning := openAIReasoningValue(raw); reasoning != "" {
			return reasoning
		}
	}
	return ""
}

func openAIReasoningValue(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}

	var object struct {
		Text             string            `json:"text"`
		Content          json.RawMessage   `json:"content"`
		Value            string            `json:"value"`
		Reasoning        json.RawMessage   `json:"reasoning"`
		ReasoningContent json.RawMessage   `json:"reasoning_content"`
		Parts            []json.RawMessage `json:"parts"`
	}
	if err := json.Unmarshal(raw, &object); err == nil {
		switch {
		case object.Text != "":
			return object.Text
		case object.Value != "":
			return object.Value
		}
		for _, nested := range []json.RawMessage{
			object.Content,
			object.ReasoningContent,
			object.Reasoning,
		} {
			if reasoning := openAIReasoningValue(nested); reasoning != "" {
				return reasoning
			}
		}
		if len(object.Parts) > 0 {
			return openAIReasoningParts(object.Parts)
		}
	}

	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err == nil {
		return openAIReasoningParts(parts)
	}

	return ""
}

func openAIReasoningParts(parts []json.RawMessage) string {
	var reasoning strings.Builder
	for _, part := range parts {
		if text := openAIReasoningValue(part); text != "" {
			reasoning.WriteString(text)
		}
	}
	return reasoning.String()
}

func float64SliceToFloat32(values []float64) []float32 {
	if len(values) == 0 {
		return nil
	}
	result := make([]float32, len(values))
	for i, value := range values {
		result[i] = float32(value)
	}
	return result
}

func init() {
	RegisterProviderFactory(APIFormatOpenAI, func(ctx context.Context, cfg ProviderConfig) (Client, error) {
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

		client := NewOpenAIClientWithOptions(
			WithAPIKey(apiKey),
			WithBaseURL(cfg.BaseURL),
		)
		client.tokenSource = tokenSource
		for k, v := range cfg.Options {
			if strings.HasPrefix(strings.ToLower(k), "header.") {
				client.headers[strings.TrimPrefix(k, "header.")] = v
			}
		}
		return client, nil
	})
}
