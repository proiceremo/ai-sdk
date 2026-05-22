package openai

import "encoding/json"

type openAIEmbeddingRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	Dimensions     *int32   `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

type openAIEmbeddingResponse struct {
	Data  []openAIEmbedding    `json:"data"`
	Model string               `json:"model"`
	Usage openAIEmbeddingUsage `json:"usage"`
}

type openAIEmbedding struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index,omitempty"`
	Object    string    `json:"object,omitempty"`
}

type openAIEmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type openAIChatCompletionRequest struct {
	Model            string                                `json:"model"`
	Messages         []openAIChatCompletionRequestMessage  `json:"messages"`
	Tools            []openAIChatCompletionTool            `json:"tools,omitempty"`
	ToolChoice       *openAIChatCompletionToolChoice       `json:"tool_choice,omitempty"`
	Temperature      *float64                              `json:"temperature,omitempty"`
	MaxTokens        *int                                  `json:"max_tokens,omitempty"`
	TopP             *float64                              `json:"top_p,omitempty"`
	WebSearchOptions *openAIChatCompletionWebSearchOptions `json:"web_search_options,omitempty"`
	Stream           bool                                  `json:"stream,omitempty"`
	StreamOptions    *openAIChatCompletionStreamOptions    `json:"stream_options,omitempty"`
	ResponseFormat   *openAIChatCompletionResponseFormat   `json:"response_format,omitempty"`
	ServiceTier      *string                               `json:"service_tier,omitempty"`
}

type openAIChatCompletionRequestMessage struct {
	Role             string                         `json:"role"`
	Content          any                            `json:"content,omitempty"`
	ReasoningContent string                         `json:"reasoning_content,omitempty"`
	ToolCallID       string                         `json:"tool_call_id,omitempty"`
	ToolCalls        []openAIChatCompletionToolCall `json:"tool_calls,omitempty"`
}

type openAIChatCompletionTextPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type openAIChatCompletionImagePart struct {
	Type     string                       `json:"type"`
	ImageURL openAIChatCompletionImageURL `json:"image_url"`
}

type openAIChatCompletionImageURL struct {
	URL string `json:"url"`
}

type openAIChatCompletionInputAudioPart struct {
	Type       string                         `json:"type"`
	InputAudio openAIChatCompletionInputAudio `json:"input_audio"`
}

type openAIChatCompletionInputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type openAIChatCompletionFilePart struct {
	Type string                   `json:"type"`
	File openAIChatCompletionFile `json:"file"`
}

type openAIChatCompletionFile struct {
	FileData *string `json:"file_data,omitempty"`
	Filename *string `json:"filename,omitempty"`
}

type openAIChatCompletionTool struct {
	Type     string                                  `json:"type"`
	Function *openAIChatCompletionFunctionDefinition `json:"function,omitempty"`
}

type openAIChatCompletionFunctionDefinition struct {
	Name        string         `json:"name"`
	Description *string        `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

type openAIChatCompletionToolChoice struct {
	Type     string                                     `json:"type"`
	Function *openAIChatCompletionToolChoiceFunctionRef `json:"function,omitempty"`
}

type openAIChatCompletionToolChoiceFunctionRef struct {
	Name string `json:"name"`
}

type openAIChatCompletionStreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type openAIChatCompletionResponseFormat struct {
	Type       string                          `json:"type"`
	JSONSchema *openAIChatCompletionJSONSchema `json:"json_schema,omitempty"`
}

type openAIChatCompletionWebSearchOptions struct {
	UserLocation      *openAIChatCompletionUserLocation `json:"user_location,omitempty"`
	SearchContextSize string                            `json:"search_context_size,omitempty"`
}

type openAIChatCompletionUserLocation struct {
	Type        string                                   `json:"type,omitempty"`
	Approximate *openAIChatCompletionApproximateLocation `json:"approximate,omitempty"`
}

type openAIChatCompletionApproximateLocation struct {
	City    string `json:"city,omitempty"`
	Region  string `json:"region,omitempty"`
	Country string `json:"country,omitempty"`
}

type openAIChatCompletionJSONSchema struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema,omitempty"`
}

type openAIChatCompletionResponse struct {
	ID          string                       `json:"id,omitempty"`
	Choices     []openAIChatCompletionChoice `json:"choices"`
	Usage       *openAICompletionUsage       `json:"usage,omitempty"`
	ServiceTier *string                      `json:"service_tier,omitempty"`
}

type openAIChatCompletionChoice struct {
	FinishReason string                              `json:"finish_reason,omitempty"`
	Index        int                                 `json:"index,omitempty"`
	Message      openAIChatCompletionResponseMessage `json:"message"`
}

type openAIChatCompletionResponseMessage struct {
	Content          string                         `json:"content,omitempty"`
	Refusal          string                         `json:"refusal,omitempty"`
	Reasoning        json.RawMessage                `json:"reasoning,omitempty"`
	ReasoningContent json.RawMessage                `json:"reasoning_content,omitempty"`
	ToolCalls        []openAIChatCompletionToolCall `json:"tool_calls,omitempty"`
}

type openAIChatCompletionToolCall struct {
	ID       string                           `json:"id,omitempty"`
	Type     string                           `json:"type,omitempty"`
	Function openAIChatCompletionFunctionCall `json:"function"`
}

type openAIChatCompletionFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type openAIChatCompletionChunk struct {
	ID          string                            `json:"id,omitempty"`
	Choices     []openAIChatCompletionChunkChoice `json:"choices"`
	Usage       *openAICompletionUsage            `json:"usage,omitempty"`
	ServiceTier *string                           `json:"service_tier,omitempty"`
}

type openAIChatCompletionChunkChoice struct {
	Delta        openAIChatCompletionChunkDelta `json:"delta"`
	FinishReason *string                        `json:"finish_reason,omitempty"`
	Index        int                            `json:"index"`
}

type openAIChatCompletionChunkDelta struct {
	Content          *string                             `json:"content,omitempty"`
	Refusal          *string                             `json:"refusal,omitempty"`
	Reasoning        json.RawMessage                     `json:"reasoning,omitempty"`
	ReasoningContent json.RawMessage                     `json:"reasoning_content,omitempty"`
	ToolCalls        []openAIChatCompletionChunkToolCall `json:"tool_calls,omitempty"`
}

type openAIChatCompletionChunkToolCall struct {
	Index    int                                    `json:"index"`
	ID       *string                                `json:"id,omitempty"`
	Type     *string                                `json:"type,omitempty"`
	Function *openAIChatCompletionFunctionCallDelta `json:"function,omitempty"`
}

type openAIChatCompletionFunctionCallDelta struct {
	Name      *string `json:"name,omitempty"`
	Arguments *string `json:"arguments,omitempty"`
}

type openAICompletionUsage struct {
	PromptTokens            int                            `json:"prompt_tokens"`
	CompletionTokens        int                            `json:"completion_tokens"`
	TotalTokens             int                            `json:"total_tokens"`
	PromptTokensDetails     *openAIPromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *openAICompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type openAIPromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
	// CacheWriteTokens is an OpenRouter-style extension reporting the
	// portion of the prompt that became fresh cache writes. Real OpenAI
	// doesn't surface this (their prompt cache is implicit and free),
	// but providers proxying through OpenRouter do — the pi reference
	// reads it from this same field name (openai-completions.ts:1008).
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	// AudioTokens is set by the gpt-4o-audio family when the prompt
	// includes an audio input part — count of tokens that came from
	// audio rather than text.
	AudioTokens int `json:"audio_tokens,omitempty"`
}

type openAICompletionTokensDetails struct {
	// ReasoningTokens is the portion of completion_tokens spent on
	// hidden chain-of-thought reasoning (o-series, GPT-5/Codex on the
	// chat-completions path). Already included in CompletionTokens —
	// surfaced separately for diagnostic visibility.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	AudioTokens     int `json:"audio_tokens,omitempty"`
}
