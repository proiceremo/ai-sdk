package llm

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// TODO: extend llm client interface to have optional realtime completion api
type APIFormat string

const (
	APIFormatOpenAI          APIFormat = "openai"
	APIFormatOpenAICodex     APIFormat = "openai_codex"
	APIFormatOpenAIResponses APIFormat = "openai_responses"
	APIFormatAnthropic       APIFormat = "anthropic"
	APIFormatGoogle          APIFormat = "google"
)

// TODO: add support for hosted tools/built in tools
type HostedToolStyle string

const (
	HostedToolStyleToolLoop     HostedToolStyle = "tool_loop"
	HostedToolStyleInlineResult HostedToolStyle = "inline_result"
	HostedToolStyleMetadataOnly HostedToolStyle = "metadata_only"
)

type ToolExecutionMode string

const (
	ToolExecutionModeClient ToolExecutionMode = "client"
	ToolExecutionModeServer ToolExecutionMode = "server"
)

type UsageOperation string

const (
	UsageOperationCompletion UsageOperation = "completion"
	UsageOperationEmbedding  UsageOperation = "embedding"
	UsageOperationTool       UsageOperation = "tool"
)

type Usage struct {
	Entries []UsageEntry                 `json:"entries,omitempty"`
	Totals  TokenUsage                   `json:"totals"`
	ByModel map[string]ModelUsageSummary `json:"by_model,omitempty"`
	Cost    float64                      `json:"cost,omitempty"`
}
type UsageEntry struct {
	Operation       UsageOperation `json:"operation"`
	ProviderID      string         `json:"provider_id,omitempty"`
	ModelID         string         `json:"model_id,omitempty"`
	ProviderModelID string         `json:"provider_model_id,omitempty"`
	Tokens          TokenUsage     `json:"tokens"`
	Cost            float64        `json:"cost,omitempty"`
}
type ModelUsageSummary struct {
	ProviderID      string     `json:"provider_id,omitempty"`
	ModelID         string     `json:"model_id,omitempty"`
	ProviderModelID string     `json:"provider_model_id,omitempty"`
	Tokens          TokenUsage `json:"tokens"`
	Cost            float64    `json:"cost,omitempty"`
}
type TokenUsage struct {
	InputTokens              int            `json:"input_tokens"`
	OutputTokens             int            `json:"output_tokens"`
	TotalTokens              int            `json:"total_tokens"`
	CacheCreationInputTokens int            `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int            `json:"cache_read_input_tokens,omitempty"`
	ToolUseInputTokens       int            `json:"tool_use_input_tokens,omitempty"`
	ServerToolUse            map[string]int `json:"server_tool_use,omitempty"`
	ServiceTier              string         `json:"service_tier,omitempty"`
	// CacheBilledSeparately indicates that the provider does NOT count
	// cache tokens inside InputTokens/TotalTokens. When true, cache tokens
	// are billed separately and should NOT be subtracted from InputTokens.
	// Default (false) preserves backward-compatible behaviour: cache is
	// assumed to be included in InputTokens.
	CacheBilledSeparately bool `json:"cache_billed_separately,omitempty"`
	// ReasoningOutputTokens is a sub-count of OutputTokens reported by
	// reasoning models (OpenAI o-series, GPT-5/Codex, etc). The provider
	// already includes these in OutputTokens — this field exposes how
	// much of the output was "hidden" chain-of-thought reasoning so the
	// bench can show e.g. "47% of output spent on reasoning". Surfacing
	// it here does NOT change billing math: don't add it to OutputTokens
	// when normalising.
	ReasoningOutputTokens           int                `json:"reasoning_output_tokens,omitempty"`
	InputTokensDetails              *UsageTokenDetails `json:"input_tokens_details,omitempty"`
	OutputTokensDetails             *UsageTokenDetails `json:"output_tokens_details,omitempty"`
	CacheCreationInputTokensDetails *UsageTokenDetails `json:"cache_creation_input_tokens_details,omitempty"`
	CacheReadInputTokensDetails     *UsageTokenDetails `json:"cache_read_input_tokens_details,omitempty"`
	ToolUseInputTokensDetails       *UsageTokenDetails `json:"tool_use_input_tokens_details,omitempty"`
}
type UsageTokenDetails struct {
	TextTokens     int `json:"text_tokens,omitempty"`
	AudioTokens    int `json:"audio_tokens,omitempty"`
	ImageTokens    int `json:"image_tokens,omitempty"`
	VideoTokens    int `json:"video_tokens,omitempty"`
	DocumentTokens int `json:"document_tokens,omitempty"`
}

func (d *UsageTokenDetails) Add(other *UsageTokenDetails) {
	if d == nil || other == nil {
		return
	}
	d.TextTokens += other.TextTokens
	d.AudioTokens += other.AudioTokens
	d.ImageTokens += other.ImageTokens
	d.VideoTokens += other.VideoTokens
	d.DocumentTokens += other.DocumentTokens
}
func (d *UsageTokenDetails) Total() int {
	if d == nil {
		return 0
	}
	return d.TextTokens + d.AudioTokens + d.ImageTokens + d.VideoTokens + d.DocumentTokens
}
func (d *UsageTokenDetails) Empty() bool {
	return d == nil || d.Total() == 0
}
func cloneUsageTokenDetails(details *UsageTokenDetails) *UsageTokenDetails {
	if details == nil {
		return nil
	}
	cloned := *details
	return &cloned
}
func addUsageTokenDetails(dst **UsageTokenDetails, src *UsageTokenDetails) {
	if src == nil {
		return
	}
	if *dst == nil {
		*dst = cloneUsageTokenDetails(src)
		return
	}
	(*dst).Add(src)
}
func NewUsage(operation UsageOperation, tokens TokenUsage) *Usage {
	usage := &Usage{}
	usage.AddEntry(UsageEntry{
		Operation: operation,
		Tokens:    tokens.normalized(),
	})
	return usage
}
func (u *Usage) AddEntry(entry UsageEntry) {
	if u == nil {
		return
	}
	entry.Tokens = entry.Tokens.normalized().clone()
	u.Entries = append(u.Entries, entry)
	u.Recompute()
}
func (u *Usage) Merge(other *Usage) {
	if u == nil || other == nil {
		return
	}
	for _, entry := range other.Entries {
		u.Entries = append(u.Entries, entry.clone())
	}
	u.Recompute()
}
func (u *Usage) Recompute() {
	if u == nil {
		return
	}
	u.Totals = TokenUsage{}
	u.ByModel = nil
	u.Cost = 0
	for i := range u.Entries {
		u.Entries[i].Tokens = u.Entries[i].Tokens.normalized()
		u.Totals.Add(u.Entries[i].Tokens)
		u.Cost += u.Entries[i].Cost
		key := u.Entries[i].ModelID
		if key == "" {
			key = u.Entries[i].ProviderModelID
		}
		if key == "" {
			continue
		}
		if u.ByModel == nil {
			u.ByModel = map[string]ModelUsageSummary{}
		}
		summary := u.ByModel[key]
		summary.ProviderID = u.Entries[i].ProviderID
		summary.ModelID = u.Entries[i].ModelID
		summary.ProviderModelID = u.Entries[i].ProviderModelID
		summary.Tokens.Add(u.Entries[i].Tokens)
		summary.Cost += u.Entries[i].Cost
		u.ByModel[key] = summary
	}
}
func (u *Usage) Clone() *Usage {
	if u == nil {
		return nil
	}
	cloned := &Usage{}
	cloned.Merge(u)
	return cloned
}
func (u *Usage) FirstTokenUsage() TokenUsage {
	if u == nil {
		return TokenUsage{}
	}
	if len(u.Entries) > 0 {
		return u.Entries[0].Tokens.clone()
	}
	return u.Totals.clone()
}

// ContextOccupied returns the total token count that occupies the model
// context window, including cache tokens when they are billed separately.
func (t TokenUsage) ContextOccupied() int {
	t = t.normalized()
	occupied := t.InputTokens + t.OutputTokens + t.ToolUseInputTokens
	if t.CacheBilledSeparately {
		occupied += t.CacheCreationInputTokens + t.CacheReadInputTokens
	}
	return occupied
}

// LastCompletionContextOccupied returns ContextOccupied for the most recent
// completion entry, or zero if there are no completion entries.
func (u *Usage) LastCompletionContextOccupied() int {
	if u == nil {
		return 0
	}
	for i := len(u.Entries) - 1; i >= 0; i-- {
		if u.Entries[i].Operation == UsageOperationCompletion {
			return u.Entries[i].Tokens.ContextOccupied()
		}
	}
	return 0
}
func (e UsageEntry) clone() UsageEntry {
	e.Tokens = e.Tokens.clone()
	return e
}
func (t TokenUsage) normalized() TokenUsage {
	if t.TotalTokens == 0 {
		t.TotalTokens = t.InputTokens + t.OutputTokens + t.ToolUseInputTokens
		// When cache is billed separately (not included in InputTokens),
		// include it in the total so context-usage calculations are accurate.
		if t.CacheBilledSeparately {
			t.TotalTokens += t.CacheCreationInputTokens + t.CacheReadInputTokens
		}
	}
	return t
}
func (t TokenUsage) clone() TokenUsage {
	t.ServerToolUse = cloneServerToolUseMap(t.ServerToolUse)
	t.InputTokensDetails = cloneUsageTokenDetails(t.InputTokensDetails)
	t.OutputTokensDetails = cloneUsageTokenDetails(t.OutputTokensDetails)
	t.CacheCreationInputTokensDetails = cloneUsageTokenDetails(t.CacheCreationInputTokensDetails)
	t.CacheReadInputTokensDetails = cloneUsageTokenDetails(t.CacheReadInputTokensDetails)
	t.ToolUseInputTokensDetails = cloneUsageTokenDetails(t.ToolUseInputTokensDetails)
	return t
}
func (t *TokenUsage) Add(other TokenUsage) {
	if t == nil {
		return
	}
	other = other.normalized()
	t.InputTokens += other.InputTokens
	t.OutputTokens += other.OutputTokens
	t.TotalTokens += other.TotalTokens
	t.CacheCreationInputTokens += other.CacheCreationInputTokens
	t.CacheReadInputTokens += other.CacheReadInputTokens
	t.ToolUseInputTokens += other.ToolUseInputTokens
	t.ReasoningOutputTokens += other.ReasoningOutputTokens
	if len(other.ServerToolUse) > 0 {
		if t.ServerToolUse == nil {
			t.ServerToolUse = map[string]int{}
		}
		for key, value := range other.ServerToolUse {
			t.ServerToolUse[key] += value
		}
	}
	if t.ServiceTier == "" {
		t.ServiceTier = other.ServiceTier
	}
	addUsageTokenDetails(&t.InputTokensDetails, other.InputTokensDetails)
	addUsageTokenDetails(&t.OutputTokensDetails, other.OutputTokensDetails)
	addUsageTokenDetails(&t.CacheCreationInputTokensDetails, other.CacheCreationInputTokensDetails)
	addUsageTokenDetails(&t.CacheReadInputTokensDetails, other.CacheReadInputTokensDetails)
	addUsageTokenDetails(&t.ToolUseInputTokensDetails, other.ToolUseInputTokensDetails)
}
func cloneServerToolUseMap(values map[string]int) map[string]int {
	if len(values) == 0 {
		return nil
	}
	cloned := map[string]int{}
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

type MessageRole string

const (
	MessageRoleUser      MessageRole = "user"
	MessageRoleAssistant MessageRole = "assistant"
)

type StopReason string

const (
	StopReasonEndTurn       StopReason = "end_turn"
	StopReasonMaxTokens     StopReason = "max_tokens"
	StopReasonStopSequence  StopReason = "stop_sequence"
	StopReasonToolUse       StopReason = "tool_use"
	StopReasonContentFilter StopReason = "content_filter"
	StopReasonPauseTurn     StopReason = "pause_turn"
	StopReasonUnknown       StopReason = "unknown"
)

type MessageContent []ContentBlock
type Message struct {
	Role       MessageRole    `json:"role"`
	Content    MessageContent `json:"content"`
	Timestamp  time.Time      `json:"timestamp"`
	Metadata   map[string]any `json:"metadata,omitempty"`
	StopReason StopReason     `json:"stop_reason,omitempty"`
	Usage      *Usage         `json:"usage,omitempty"`
}

func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	out := make([]Message, len(messages))
	for i, msg := range messages {
		out[i] = msg.Clone()
	}
	return out
}

func StripEphemeralMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}
	out := make([]Message, 0, len(messages))
	for _, msg := range messages {
		msg = msg.Clone()
		msg.Content = msg.Content.WithoutEphemeral()
		if len(msg.Content) == 0 {
			continue
		}
		out = append(out, msg)
	}
	return out
}

func (m Message) Clone() Message {
	out := m
	out.Content = m.Content.Clone()
	out.Metadata = cloneAnyMap(m.Metadata)
	if m.Usage != nil {
		out.Usage = m.Usage.Clone()
	}
	return out
}

func (m *Message) AddContent(content ...ContentBlock) {
	m.Content = append(m.Content, content...)
}
func (m MessageContent) Clone() MessageContent {
	if m == nil {
		return nil
	}
	out := make(MessageContent, len(m))
	for i, block := range m {
		out[i] = block.Clone()
	}
	return out
}
func (m MessageContent) WithoutEphemeral() MessageContent {
	if len(m) == 0 {
		return m
	}
	out := make(MessageContent, 0, len(m))
	for _, block := range m {
		if block.Ephemeral {
			continue
		}
		block = block.Clone()
		if block.ToolUse != nil && block.ToolUse.Metadata != nil {
			metadata := block.ToolUse.Metadata.WithoutEphemeral()
			block.ToolUse.Metadata = &metadata
		}
		if block.ToolOutput != nil {
			block.ToolOutput.Output = block.ToolOutput.Output.WithoutEphemeral()
			if block.ToolOutput.Metadata != nil {
				metadata := block.ToolOutput.Metadata.WithoutEphemeral()
				block.ToolOutput.Metadata = &metadata
			}
		}
		out = append(out, block)
	}
	return out
}
func (m MessageContent) Validate() error {
	for i, block := range m {
		if err := block.Validate(); err != nil {
			return fmt.Errorf("content block %d: %w", i, err)
		}
	}
	return nil
}
func (m *MessageContent) ToolCalls() []ToolUse {
	var toolCalls []ToolUse
	for _, content := range *m {
		if content.Type == ContentBlockTypeToolUse && content.ToolUse != nil {
			toolCalls = append(toolCalls, *content.ToolUse)
		}
	}
	return toolCalls
}
func (m *MessageContent) Text() string {
	var text strings.Builder
	for _, content := range *m {
		switch content.Type {
		case ContentBlockTypeText:
			if content.Text != "" {
				text.WriteString(content.Text)
			}
		case ContentBlockTypeDocument:
			appendDocumentText(&text, content.Document)
		case ContentBlockTypeToolResult:
			if content.ToolOutput != nil {
				text.WriteString(content.ToolOutput.Output.Text())
			}
		}
	}
	return text.String()
}
func (m *MessageContent) Thinking() string {
	var thinking string
	for _, content := range *m {
		if content.Type == ContentBlockTypeThinking && content.Thinking != "" {
			thinking += content.Thinking
		}
	}
	return thinking
}
func (m *MessageContent) ToString() string {
	var text strings.Builder
	for _, content := range *m {
		switch content.Type {
		case ContentBlockTypeText:
			fmt.Fprintf(&text, "%s", content.Text)
		case ContentBlockTypeThinking:
			fmt.Fprintf(&text, "[THINKING] %s", content.Thinking)
		case ContentBlockTypeRedactedThinking:
			fmt.Fprintf(&text, "[REDACTED_THINKING]")
		case ContentBlockTypeToolUse:
			fmt.Fprintf(&text, "[TOOL_CALL:%s] %s", content.ToolUse.ID, string(content.ToolUse.Input))
		case ContentBlockTypeToolResult:
			fmt.Fprintf(&text, "[TOOL_RESULT:%s] %s", content.ToolOutput.ToolUseID, content.ToolOutput.Output.ToString())
		case ContentBlockTypeImage:
			if content.Image.Type == ImageSourceTypeURL {
				fmt.Fprintf(&text, "[IMAGE] %s", content.Image.URL)
			} else {
				fmt.Fprintf(&text, "[IMAGE] %s...", content.Image.Data[:min(len(content.Image.Data), 100)])
			}
		case ContentBlockTypeDocument:
			if content.Document != nil {
				if documentText := strings.TrimSpace(content.documentText()); documentText != "" {
					fmt.Fprintf(&text, "[DOCUMENT] %s", documentText)
					break
				}
				switch content.Document.Type {
				case DocumentSourceTypeURL:
					fmt.Fprintf(&text, "[DOCUMENT] %s", content.Document.URL)
				case DocumentSourceTypeBase64:
					fmt.Fprintf(&text, "[DOCUMENT] %s...", content.Document.Data[:min(len(content.Document.Data), 100)])
				case DocumentSourceTypeText:
					fmt.Fprintf(&text, "[DOCUMENT]")
				}
			}
		case ContentBlockTypeAudio:
			if content.Audio.Type == AudioSourceTypeURL {
				fmt.Fprintf(&text, "[AUDIO] %s", content.Audio.URL)
			} else {
				fmt.Fprintf(&text, "[AUDIO] %s...", content.Audio.Data[:min(len(content.Audio.Data), 100)])
			}
		case ContentBlockTypeVideo:
			if content.Video.Type == VideoSourceTypeURL {
				fmt.Fprintf(&text, "[VIDEO] %s", content.Video.URL)
			} else {
				fmt.Fprintf(&text, "[VIDEO] %s...", content.Video.Data[:min(len(content.Video.Data), 100)])
			}
		}
	}
	return text.String()
}
func (b ContentBlock) documentText() string {
	if b.Document == nil {
		return ""
	}
	parts := make([]string, 0, 3)
	for _, part := range []string{b.Document.Title, b.Document.Context, b.Document.Text} {
		part = strings.TrimSpace(part)
		if part != "" {
			parts = append(parts, part)
		}
	}
	return strings.Join(parts, "\n")
}
func appendDocumentText(text *strings.Builder, doc *DocumentSource) {
	if doc == nil {
		return
	}
	for _, part := range []string{doc.Title, doc.Context, doc.Text} {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if text.Len() > 0 {
			text.WriteByte('\n')
		}
		text.WriteString(part)
	}
}
func (b ContentBlock) Validate() error {
	switch b.Type {
	case ContentBlockTypeText:
		return nil
	case ContentBlockTypeThinking:
		return nil
	case ContentBlockTypeRedactedThinking:
		return nil
	case ContentBlockTypeImage:
		if b.Image == nil {
			return errors.New("image content is missing payload")
		}
		return b.Image.Validate()
	case ContentBlockTypeDocument:
		if b.Document == nil {
			return errors.New("document content is missing payload")
		}
		return b.Document.Validate()
	case ContentBlockTypeAudio:
		if b.Audio == nil {
			return errors.New("audio content is missing payload")
		}
		return b.Audio.Validate()
	case ContentBlockTypeVideo:
		if b.Video == nil {
			return errors.New("video content is missing payload")
		}
		return b.Video.Validate()
	case ContentBlockTypeToolUse:
		if b.ToolUse == nil {
			return errors.New("tool use content is missing payload")
		}
		if strings.TrimSpace(b.ToolUse.Name) == "" {
			return errors.New("tool use name is required")
		}
		if len(b.ToolUse.Input) > 0 && !json.Valid(b.ToolUse.Input) {
			return errors.New("tool use input must be valid JSON")
		}
		if len(b.ToolUse.ProviderData) > 0 && !json.Valid(b.ToolUse.ProviderData) {
			return errors.New("tool use provider_data must be valid JSON")
		}
		return nil
	case ContentBlockTypeToolResult:
		if b.ToolOutput == nil {
			return errors.New("tool result content is missing payload")
		}
		if strings.TrimSpace(b.ToolOutput.ToolUseID) == "" {
			return errors.New("tool result tool_use_id is required")
		}
		return b.ToolOutput.Output.Validate()
	case "":
		return errors.New("content block type is required")
	default:
		return fmt.Errorf("unsupported content block type %q", b.Type)
	}
}
func ValidateMessages(messages []Message) error {
	for i, message := range messages {
		if err := message.Content.Validate(); err != nil {
			return fmt.Errorf("message %d (%s): %w", i, message.Role, err)
		}
	}
	return nil
}

type ContentBlockType string

const (
	ContentBlockTypeText             ContentBlockType = "text"
	ContentBlockTypeImage            ContentBlockType = "image"
	ContentBlockTypeDocument         ContentBlockType = "document"
	ContentBlockTypeAudio            ContentBlockType = "audio"
	ContentBlockTypeVideo            ContentBlockType = "video"
	ContentBlockTypeToolUse          ContentBlockType = "tool_use"
	ContentBlockTypeToolResult       ContentBlockType = "tool_result"
	ContentBlockTypeThinking         ContentBlockType = "thinking"
	ContentBlockTypeRedactedThinking ContentBlockType = "redacted_thinking"
)

type ContentBlock struct {
	Type       ContentBlockType `json:"type"`
	Text       string           `json:"text,omitempty"`
	Thinking   string           `json:"thinking,omitempty"`
	Redacted   string           `json:"redacted_thinking,omitempty"`
	Signature  string           `json:"signature,omitempty"`
	Image      *ImageSource     `json:"image,omitempty"`
	Document   *DocumentSource  `json:"document,omitempty"`
	Audio      *AudioSource     `json:"audio,omitempty"`
	Video      *VideoSource     `json:"video,omitempty"`
	ToolUse    *ToolUse         `json:"tool_use,omitempty"`
	ToolOutput *ToolOutput      `json:"tool_output,omitempty"`
	Ephemeral  bool             `json:"ephemeral,omitempty"`
}

func EphemeralContentBlock(block ContentBlock) ContentBlock {
	block.Ephemeral = true
	return block
}

func NewEphemeralTextContentBlock(text string) ContentBlock {
	return EphemeralContentBlock(NewTextContentBlock(text))
}

func (b ContentBlock) Clone() ContentBlock {
	out := b
	if b.Image != nil {
		image := *b.Image
		out.Image = &image
	}
	if b.Document != nil {
		document := *b.Document
		out.Document = &document
	}
	if b.Audio != nil {
		audio := *b.Audio
		out.Audio = &audio
	}
	if b.Video != nil {
		video := *b.Video
		out.Video = &video
	}
	if b.ToolUse != nil {
		toolUse := *b.ToolUse
		toolUse.Input = cloneRawMessage(b.ToolUse.Input)
		toolUse.ProviderData = cloneRawMessage(b.ToolUse.ProviderData)
		if b.ToolUse.Index != nil {
			index := *b.ToolUse.Index
			toolUse.Index = &index
		}
		if b.ToolUse.Metadata != nil {
			metadata := b.ToolUse.Metadata.Clone()
			toolUse.Metadata = &metadata
		}
		out.ToolUse = &toolUse
	}
	if b.ToolOutput != nil {
		toolOutput := *b.ToolOutput
		toolOutput.Output = b.ToolOutput.Output.Clone()
		toolOutput.ProviderData = cloneRawMessage(b.ToolOutput.ProviderData)
		if b.ToolOutput.Metadata != nil {
			metadata := b.ToolOutput.Metadata.Clone()
			toolOutput.Metadata = &metadata
		}
		out.ToolOutput = &toolOutput
	}
	return out
}

type ToolUse struct {
	ID           string            `json:"id,omitempty"`
	Name         string            `json:"name,omitempty"`
	Input        json.RawMessage   `json:"input,omitempty"`
	Index        *int              `json:"index,omitempty"`
	Signature    string            `json:"signature,omitempty"`
	Execution    ToolExecutionMode `json:"execution,omitempty"`
	ProviderType string            `json:"provider_type,omitempty"`
	ProviderData json.RawMessage   `json:"provider_data,omitempty"`
	Metadata     *ToolMetadata     `json:"metadata,omitempty"`
	partialInput string            `json:"-"`
}
type ToolOutput struct {
	ToolUseID    string            `json:"tool_use_id,omitempty"`
	Name         string            `json:"name,omitempty"`
	Output       MessageContent    `json:"output,omitempty"`
	IsError      bool              `json:"is_error,omitempty"`
	Execution    ToolExecutionMode `json:"execution,omitempty"`
	ProviderType string            `json:"provider_type,omitempty"`
	ProviderData json.RawMessage   `json:"provider_data,omitempty"`
	Metadata     *ToolMetadata     `json:"metadata,omitempty"`
}
type ImageSourceType string

const (
	ImageSourceTypeURL    ImageSourceType = "url"
	ImageSourceTypeBase64 ImageSourceType = "base64"
)

type ImageSource struct {
	Type      ImageSourceType `json:"type"`
	URL       string          `json:"url,omitempty"`
	Data      string          `json:"data,omitempty"`
	MediaType string          `json:"media_type,omitempty"`
}

func (s *ImageSource) Validate() error {
	if s == nil {
		return errors.New("image source is required")
	}
	switch s.Type {
	case ImageSourceTypeURL:
		if strings.TrimSpace(s.URL) == "" {
			return errors.New("image url is required")
		}
	case ImageSourceTypeBase64:
		if strings.TrimSpace(s.Data) == "" {
			return errors.New("image data is required")
		}
	default:
		return fmt.Errorf("unsupported image source type %q", s.Type)
	}
	return nil
}

type DocumentSourceType string

const (
	DocumentSourceTypeURL    DocumentSourceType = "url"
	DocumentSourceTypeBase64 DocumentSourceType = "base64"
	DocumentSourceTypeText   DocumentSourceType = "text"
)

type DocumentSource struct {
	Type      DocumentSourceType `json:"type"`
	URL       string             `json:"url,omitempty"`
	Data      string             `json:"data,omitempty"`
	MediaType string             `json:"media_type,omitempty"`
	Name      string             `json:"name,omitempty"`
	Text      string             `json:"text,omitempty"`
	Title     string             `json:"title,omitempty"`
	Context   string             `json:"context,omitempty"`
}

func (s *DocumentSource) Validate() error {
	if s == nil {
		return errors.New("document source is required")
	}
	switch s.Type {
	case DocumentSourceTypeURL:
		if strings.TrimSpace(s.URL) == "" {
			return errors.New("document url is required")
		}
	case DocumentSourceTypeBase64:
		if strings.TrimSpace(s.Data) == "" {
			return errors.New("document data is required")
		}
	case DocumentSourceTypeText:
		if strings.TrimSpace(s.Text) == "" && strings.TrimSpace(s.Title) == "" && strings.TrimSpace(s.Context) == "" {
			return errors.New("document text, title, or context is required")
		}
	default:
		return fmt.Errorf("unsupported document source type %q", s.Type)
	}
	return nil
}

type AudioSource struct {
	Type      AudioSourceType `json:"type"`
	URL       string          `json:"url,omitempty"`
	Data      string          `json:"data,omitempty"`
	MediaType string          `json:"media_type,omitempty"`
}

func (s *AudioSource) Validate() error {
	if s == nil {
		return errors.New("audio source is required")
	}
	switch s.Type {
	case AudioSourceTypeURL:
		if strings.TrimSpace(s.URL) == "" {
			return errors.New("audio url is required")
		}
	case AudioSourceTypeBase64:
		if strings.TrimSpace(s.Data) == "" {
			return errors.New("audio data is required")
		}
	default:
		return fmt.Errorf("unsupported audio source type %q", s.Type)
	}
	return nil
}

type AudioSourceType string

const (
	AudioSourceTypeURL    AudioSourceType = "url"
	AudioSourceTypeBase64 AudioSourceType = "base64"
)

type VideoSource struct {
	Type      VideoSourceType `json:"type"`
	URL       string          `json:"url,omitempty"`
	Data      string          `json:"data,omitempty"`
	MediaType string          `json:"media_type,omitempty"`
}

func (s *VideoSource) Validate() error {
	if s == nil {
		return errors.New("video source is required")
	}
	switch s.Type {
	case VideoSourceTypeURL:
		if strings.TrimSpace(s.URL) == "" {
			return errors.New("video url is required")
		}
	case VideoSourceTypeBase64:
		if strings.TrimSpace(s.Data) == "" {
			return errors.New("video data is required")
		}
	default:
		return fmt.Errorf("unsupported video source type %q", s.Type)
	}
	return nil
}

type VideoSourceType string

const (
	VideoSourceTypeURL    VideoSourceType = "url"
	VideoSourceTypeBase64 VideoSourceType = "base64"
)

func NewTextContentBlock(text string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeText,
		Text: text,
	}
}

func NewImageContentBlockFromURL(url string, mediaType string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeImage,
		Image: &ImageSource{
			Type:      ImageSourceTypeURL,
			URL:       url,
			MediaType: mediaType,
		},
	}
}
func NewImageContentBlockFromBase64(data string, mediaType string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeImage,
		Image: &ImageSource{
			Type:      ImageSourceTypeBase64,
			Data:      data,
			MediaType: mediaType,
		},
	}
}
func NewDocumentContentBlockFromURL(url string, mediaType string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeDocument,
		Document: &DocumentSource{
			Type:      DocumentSourceTypeURL,
			URL:       url,
			MediaType: mediaType,
		},
	}
}
func NewDocumentContentBlockFromBase64(data string, mediaType string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeDocument,
		Document: &DocumentSource{
			Type:      DocumentSourceTypeBase64,
			Data:      data,
			MediaType: mediaType,
		},
	}
}
func NewDocumentContentBlockFromText(text string, mediaType string, url string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeDocument,
		Document: &DocumentSource{
			Type:      DocumentSourceTypeText,
			Text:      text,
			MediaType: mediaType,
			URL:       url,
		},
	}
}
func NewAudioContentBlockFromURL(url string, mediaType string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeAudio,
		Audio: &AudioSource{
			Type:      AudioSourceTypeURL,
			URL:       url,
			MediaType: mediaType,
		},
	}
}
func NewAudioContentBlockFromBase64(data string, mediaType string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeAudio,
		Audio: &AudioSource{
			Type:      AudioSourceTypeBase64,
			Data:      data,
			MediaType: mediaType,
		},
	}
}
func NewVideoContentBlockFromURL(url string, mediaType string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeVideo,
		Video: &VideoSource{
			Type:      VideoSourceTypeURL,
			URL:       url,
			MediaType: mediaType,
		},
	}
}
func NewVideoContentBlockFromBase64(data string, mediaType string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeVideo,
		Video: &VideoSource{
			Type:      VideoSourceTypeBase64,
			Data:      data,
			MediaType: mediaType,
		},
	}
}
func NewToolUseContentBlock(id, name string, input json.RawMessage, index *int) ContentBlock {
	if id == "" {
		id = fmt.Sprintf("call_%s_%d", name, time.Now().UnixNano())
	}
	return ContentBlock{
		Type: ContentBlockTypeToolUse,
		ToolUse: &ToolUse{
			ID:        id,
			Name:      name,
			Input:     input,
			Index:     index,
			Execution: ToolExecutionModeClient,
		},
	}
}
func NewToolResultContentBlock(result ToolOutput) ContentBlock {
	return ContentBlock{
		Type:       ContentBlockTypeToolResult,
		ToolOutput: &result,
	}
}
func NewThinkingContentBlock(thinking string, signature string) ContentBlock {
	return ContentBlock{
		Type:      ContentBlockTypeThinking,
		Thinking:  thinking,
		Signature: signature,
	}
}

func (m ToolMetadata) Clone() ToolMetadata {
	out := m
	if m.Locations != nil {
		out.Locations = make([]ToolCallLocation, len(m.Locations))
		for i, loc := range m.Locations {
			out.Locations[i] = loc
			out.Locations[i].Metadata = cloneAnyMap(loc.Metadata)
		}
	}
	if m.Content != nil {
		out.Content = make([]ToolCallContent, len(m.Content))
		for i, content := range m.Content {
			out.Content[i] = cloneToolCallContent(content)
		}
	}
	return out
}

func (m ToolMetadata) WithoutEphemeral() ToolMetadata {
	out := m.Clone()
	if len(out.Content) == 0 {
		return out
	}
	content := make([]ToolCallContent, 0, len(out.Content))
	for _, item := range out.Content {
		if item.Content != nil && item.Content.Block.Ephemeral {
			continue
		}
		if item.Content != nil {
			item.Content.Block = item.Content.Block.Clone()
			item.Content.Block.Ephemeral = false
		}
		content = append(content, item)
	}
	out.Content = content
	return out
}

func cloneToolCallContent(content ToolCallContent) ToolCallContent {
	out := content
	if content.Content != nil {
		out.Content = &ToolCallContentBlock{Block: content.Content.Block.Clone()}
	}
	if content.Diff != nil {
		diff := *content.Diff
		diff.Metadata = cloneAnyMap(content.Diff.Metadata)
		if content.Diff.OldText != nil {
			oldText := *content.Diff.OldText
			diff.OldText = &oldText
		}
		out.Diff = &diff
	}
	if content.Terminal != nil {
		terminal := *content.Terminal
		out.Terminal = &terminal
	}
	return out
}

func cloneRawMessage(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}

func cloneAnyMap(values map[string]any) map[string]any {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

type Client interface {
	CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error)
	CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error)
}
type StreamingCapable interface {
	SupportsStreaming() bool
}
type InferenceParams struct {
	Model        string      `json:"model,omitempty"`
	SystemPrompt string      `json:"system_prompt,omitempty"`
	Tools        []Tool      `json:"tools,omitempty"`
	ToolChoice   *ToolChoice `json:"tool_choice,omitempty"`
	// TODO: add hosted/built in tools
	Temperature    *float64             `json:"temperature,omitempty"`
	MaxTokens      *int                 `json:"max_tokens,omitempty"`
	TopP           *float64             `json:"top_p,omitempty"`
	TopK           *int                 `json:"top_k,omitempty"`
	Thinking       *ThinkingParams      `json:"thinking,omitempty"`
	WebSearch      *WebSearchToolParams `json:"web_search,omitempty"`
	ResponseFormat *ResponseFormat      `json:"response_format,omitempty"`
	ServiceTier    *string              `json:"service_tier,omitempty"`
	// CacheRetention selects the prompt-cache TTL when the provider
	// supports configurable cache lifetimes. Today only Anthropic acts
	// on this; other providers ignore it. Accepted values:
	//
	//   ""        — same as "default"; use the provider's standard
	//               ephemeral cache (5 minutes on Anthropic).
	//   "default" — explicit short-cache request, equivalent to "".
	//   "long"    — extended cache tier (1h on Anthropic). Cache WRITES
	//               cost 1.25x more per token but reads still cost
	//               0.1x, so this only pays off when the same prefix
	//               is re-hit many times within the hour. Use for
	//               multi-turn agent loops with stable system + tools.
	//
	// pi reference: inspo/pi/packages/ai/src/providers/anthropic.ts:62.
	// pi does NOT send any anthropic-beta header for this — the 1h
	// tier is GA on Anthropic's native API since ~2025-04.
	CacheRetention string `json:"cache_retention,omitempty"`
}
type ToolChoiceType string

const (
	ToolChoiceTypeAuto     ToolChoiceType = "auto"
	ToolChoiceTypeRequired ToolChoiceType = "required"
	ToolChoiceTypeTool     ToolChoiceType = "tool"
)

type ToolChoice struct {
	Type ToolChoiceType `json:"type,omitempty"`
	Name string         `json:"name,omitempty"`
}
type WebSearchToolParams struct {
	Enabled    bool   `json:"enabled,omitempty"`
	Type       string `json:"type,omitempty"`
	CustomTool string `json:"custom_tool,omitempty"`
}
type ResponseFormatType string

const (
	ResponseFormatTypeText       ResponseFormatType = "text"
	ResponseFormatTypeJSONObject ResponseFormatType = "json_object"
)

type ResponseFormat struct {
	Type       ResponseFormatType `json:"type"`
	JSONSchema map[string]any     `json:"json_schema,omitempty"`
}
type ThinkingParams struct {
	Enabled bool          `json:"enabled,omitempty"`
	Level   ThinkingLevel `json:"level,omitempty"`
}
type ThinkingLevel string

const (
	ThinkingLevelUnspecified ThinkingLevel = "unspecified"
	ThinkingLevelMinimal     ThinkingLevel = "minimal"
	ThinkingLevelLow         ThinkingLevel = "low"
	ThinkingLevelMedium      ThinkingLevel = "medium"
	ThinkingLevelHigh        ThinkingLevel = "high"
	ThinkingLevelXHigh       ThinkingLevel = "xhigh"
)

var (
	ThinkingLevelMap = map[ThinkingLevel]int{
		ThinkingLevelMinimal: 5,
		ThinkingLevelLow:     10,
		ThinkingLevelMedium:  25,
		ThinkingLevelHigh:    50,
		ThinkingLevelXHigh:   75,
	}
)

type MessageDelta struct {
	Content    MessageContent `json:"content,omitempty"`
	StopReason StopReason     `json:"stop_reason,omitempty"`
	Usage      *Usage         `json:"usage,omitempty"`
}
type StreamEvent struct {
	Type     StreamEventType `json:"type"`
	Delta    MessageDelta    `json:"delta"`
	Snapshot Message         `json:"snapshot"`
}
type StreamEventType string

const (
	EventTypeContentStart StreamEventType = "content_start"
	EventTypeContentDelta StreamEventType = "content_delta"
	EventTypeContentEnd   StreamEventType = "content_end"
	EventTypeMessageStart StreamEventType = "message_start"
	EventTypeMessageEnd   StreamEventType = "message_end"
)

type Stream interface {
	Recv(ctx context.Context) (*StreamEvent, error)
	Current() *Message
	Close() error
}

func NormaliseStopReason(provider APIFormat, reason string) StopReason {
	switch provider {
	case APIFormatOpenAI:
		switch reason {
		case "stop":
			return StopReasonEndTurn
		case "length":
			return StopReasonMaxTokens
		case "tool_calls", "function_call":
			return StopReasonToolUse
		case "content_filter":
			return StopReasonContentFilter
		default:
			return StopReasonUnknown
		}
	case APIFormatAnthropic:
		switch reason {
		case "end_turn":
			return StopReasonEndTurn
		case "max_tokens":
			return StopReasonMaxTokens
		case "stop_sequence":
			return StopReasonStopSequence
		case "tool_use":
			return StopReasonToolUse
		case "pause_turn":
			return StopReasonPauseTurn
		case "refusal":
			return StopReasonContentFilter
		default:
			return StopReasonUnknown
		}
	case APIFormatGoogle:
		switch reason {
		case "STOP":
			return StopReasonEndTurn
		case "MAX_TOKENS":
			return StopReasonMaxTokens
		case "SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
			return StopReasonContentFilter
		case "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL":
			return StopReasonToolUse
		case "OTHER", "NO_IMAGE", "IMAGE_OTHER", "FINISH_REASON_UNSPECIFIED":
			return StopReasonUnknown
		default:
			return StopReasonUnknown
		}
	default:
		return StopReasonUnknown
	}
}
func DecodeBase64File(b64Input string) ([]byte, string, error) {
	if i := strings.Index(b64Input, ","); i != -1 {
		b64Input = b64Input[i+1:]
	}
	data, err := base64.StdEncoding.DecodeString(b64Input)
	if err != nil {
		return nil, "", err
	}
	mimeType := http.DetectContentType(data)
	return data, mimeType, nil
}
func Ptr[T any](v T) *T {
	return &v
}

// ID Generation
func GenerateID(prefix ...string) string {
	b := make([]byte, 8)
	rand.Read(b)
	id := hex.EncodeToString(b)
	if len(prefix) > 0 && prefix[0] != "" {
		return fmt.Sprintf("%s_%s", prefix[0], id)
	}
	return id
}
func GenerateTimestampID() string {
	return fmt.Sprintf("%d_%s", time.Now().UnixNano(), GenerateID())
}
