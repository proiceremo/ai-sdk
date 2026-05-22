package anthropic

import "encoding/json"

type anthropicMessageRequest struct {
	Model        string                  `json:"model"`
	MaxTokens    int                     `json:"max_tokens"`
	Messages     []anthropicMessageParam `json:"messages"`
	System       []anthropicTextBlock    `json:"system,omitempty"`
	Temperature  *float64                `json:"temperature,omitempty"`
	TopP         *float64                `json:"top_p,omitempty"`
	TopK         *int                    `json:"top_k,omitempty"`
	Thinking     *anthropicThinking      `json:"thinking,omitempty"`
	Tools        []any                   `json:"tools,omitempty"`
	ToolChoice   *anthropicToolChoice    `json:"tool_choice,omitempty"`
	OutputConfig *anthropicOutputConfig  `json:"output_config,omitempty"`
	Stream       bool                    `json:"stream,omitempty"`
}

type anthropicToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

type anthropicMessageParam struct {
	Role    string                       `json:"role"`
	Content []anthropicInputContentBlock `json:"content"`
}

type anthropicTextBlock struct {
	Type         string                  `json:"type"`
	Text         string                  `json:"text"`
	CacheControl *anthropicCacheControl  `json:"cache_control,omitempty"`
}

type anthropicInputContentBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text,omitempty"`
	Source       any                    `json:"source,omitempty"`
	ID           string                 `json:"id,omitempty"`
	Input        any                    `json:"input,omitempty"`
	Name         string                 `json:"name,omitempty"`
	ToolUseID    string                 `json:"tool_use_id,omitempty"`
	Content      any                    `json:"content,omitempty"`
	IsError      *bool                  `json:"is_error,omitempty"`
	Thinking     string                 `json:"thinking,omitempty"`
	Signature    string                 `json:"signature,omitempty"`
	Data         string                 `json:"data,omitempty"`
	Title        string                 `json:"title,omitempty"`
	Context      string                 `json:"context,omitempty"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// anthropicCacheControl marks an input block as a prompt-cache breakpoint.
// Anthropic caches the PREFIX of a request up to the LAST block carrying
// cache_control, so placement is load-bearing — wrong placement = no
// cache hit even when the content is byte-identical to the last call.
//
// Type is always "ephemeral" today. TTL defaults to 5 minutes; setting
// it to "1h" enables the extended cache tier (requires the
// `extended-cache-control-2025-04-30` anthropic-beta header — we do
// NOT send that today, so leave TTL empty unless you also add the beta).
//
// Up to 4 cache breakpoints per request. Our placement strategy
// matches pi's reference (inspo/pi/packages/ai/src/providers/anthropic.ts):
//   1. Last system block (with the OAuth identity prefix appended)
//   2. Last tool definition (caches the tool schema set across turns)
//   3. Last content block of the last user message (caches the
//      conversation history including tool_results, so each new turn
//      reuses the full prefix from the previous one)
type anthropicCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type anthropicBase64Source struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data"`
}

type anthropicURLSource struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

type anthropicPlainTextSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data"`
}

type anthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type anthropicOutputConfig struct {
	Format anthropicJSONOutputFormat `json:"format"`
}

type anthropicJSONOutputFormat struct {
	Type   string         `json:"type"`
	Schema map[string]any `json:"schema"`
}

type anthropicMessage struct {
	Content      []anthropicContentBlock `json:"content"`
	Role         string                  `json:"role"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence string                  `json:"stop_sequence,omitempty"`
	Usage        anthropicUsage          `json:"usage"`
}

type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Name      string          `json:"name,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	Data      string          `json:"data,omitempty"`
}

type anthropicUsage struct {
	InputTokens              int            `json:"input_tokens"`
	OutputTokens             int            `json:"output_tokens"`
	CacheCreationInputTokens int            `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int            `json:"cache_read_input_tokens,omitempty"`
	ServerToolUse            map[string]int `json:"server_tool_use,omitempty"`
}

type anthropicStreamEvent struct {
	Type         string                      `json:"type"`
	Message      *anthropicMessage           `json:"message,omitempty"`
	Delta        json.RawMessage             `json:"delta,omitempty"`
	Usage        *anthropicMessageDeltaUsage `json:"usage,omitempty"`
	ContentBlock *anthropicContentBlock      `json:"content_block,omitempty"`
	Index        int                         `json:"index,omitempty"`
}

type anthropicMessageDelta struct {
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

type anthropicMessageDeltaUsage struct {
	InputTokens              *int `json:"input_tokens,omitempty"`
	OutputTokens             int  `json:"output_tokens,omitempty"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens,omitempty"`
}

type anthropicContentBlockDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
}
