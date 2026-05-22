package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type RetryStrategy string

const (
	RetryStrategyExponentialBackoff RetryStrategy = "exponential_backoff"
	RetryStrategyFixedDelay         RetryStrategy = "fixed_delay"
	RetryStrategyLinearBackoff      RetryStrategy = "linear_backoff"
	RetryStrategyNoRetry            RetryStrategy = "no_retry"
)

type RetryPolicy struct {
	Strategy          RetryStrategy
	MaxRetries        int
	InitialDelay      time.Duration
	MaxDelay          time.Duration
	Multiplier        float64
	Jitter            float64 // +/- fraction of delay, e.g. 0.3 = ±30%
	RespectRetryAfter bool
}

func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		Strategy:          RetryStrategyExponentialBackoff,
		MaxRetries:        3,
		InitialDelay:      time.Second,
		MaxDelay:          10 * time.Second,
		Multiplier:        2,
		Jitter:            0.3,
		RespectRetryAfter: true,
	}
}

func NoRetryPolicy() RetryPolicy {
	return RetryPolicy{Strategy: RetryStrategyNoRetry}
}

func (p RetryPolicy) Enabled() bool {
	p = p.normalized()
	return p.Strategy != RetryStrategyNoRetry && p.MaxRetries > 0
}

func (p RetryPolicy) normalized() RetryPolicy {
	if p.Strategy == "" {
		p.Strategy = RetryStrategyExponentialBackoff
	}
	if p.InitialDelay <= 0 {
		p.InitialDelay = time.Second
	}
	if p.MaxDelay <= 0 || p.MaxDelay < p.InitialDelay {
		p.MaxDelay = p.InitialDelay
	}
	if p.Multiplier <= 0 {
		if p.Strategy == RetryStrategyExponentialBackoff {
			p.Multiplier = 2
		} else {
			p.Multiplier = 1
		}
	}
	if p.MaxRetries < 0 {
		p.MaxRetries = 0
	}
	return p
}

type ClientWithRetry struct {
	client    Client
	embedding EmbeddingCapable
	policy    RetryPolicy
}

func WrapClientWithRetry(client Client, policy RetryPolicy) Client {
	policy = policy.normalized()
	if policy.Strategy == RetryStrategyNoRetry || policy.MaxRetries == 0 {
		return client
	}
	return newClientWithRetry(client, policy)
}

func NewClientWithExponentialBackoff(client Client) *ClientWithRetry {
	return newClientWithRetry(client, DefaultRetryPolicy())
}

func NewClientWithFixedDelay(client Client, delay time.Duration, maxRetries int) *ClientWithRetry {
	return newClientWithRetry(client, RetryPolicy{
		Strategy:     RetryStrategyFixedDelay,
		MaxRetries:   maxRetries,
		InitialDelay: delay,
		MaxDelay:     delay,
		Multiplier:   1,
	})
}

func NewClientWithLinearBackoff(client Client, initialDelay time.Duration, maxDelay time.Duration, maxRetries int) *ClientWithRetry {
	return newClientWithRetry(client, RetryPolicy{
		Strategy:     RetryStrategyLinearBackoff,
		MaxRetries:   maxRetries,
		InitialDelay: initialDelay,
		MaxDelay:     maxDelay,
		Multiplier:   1,
	})
}

func NewClientWithRetry(client Client, policy RetryPolicy) *ClientWithRetry {
	return newClientWithRetry(client, policy)
}

func newClientWithRetry(client Client, policy RetryPolicy) *ClientWithRetry {
	wrapped := &ClientWithRetry{
		client: client,
		policy: policy.normalized(),
	}
	if embedding, ok := client.(EmbeddingCapable); ok {
		wrapped.embedding = embedding
	}
	return wrapped
}

func isTransientError(err error) bool {
	if err == nil {
		return false
	}
	if err == context.Canceled || err == context.DeadlineExceeded {
		return false
	}

	if code, ok := extractRetryStatusCode(err); ok {
		return code == 429 || code >= 500
	}

	errStr := strings.ToLower(err.Error())
	for _, transient := range []string{
		"429",
		"500",
		"502",
		"503",
		"504",
		"timeout",
		"temporary",
		"connection refused",
		"connection reset",
		"eof",
		"broken pipe",
		"resource has been exhausted",
		"quota exceeded",
		"rate limit",
		"rate_limit",
	} {
		if strings.Contains(errStr, transient) {
			return true
		}
	}

	return false
}

func extractRetryStatusCode(err error) (int, bool) {
	for current := err; current != nil; current = errors.Unwrap(current) {
		if code, ok := retryStatusCodeFromValue(reflect.ValueOf(current)); ok {
			return code, true
		}
	}
	return 0, false
}

func retryStatusCodeFromValue(value reflect.Value) (int, bool) {
	if !value.IsValid() {
		return 0, false
	}
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return 0, false
		}
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		return 0, false
	}
	for _, fieldName := range []string{"StatusCode", "Code"} {
		field := value.FieldByName(fieldName)
		if !field.IsValid() {
			continue
		}
		switch field.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return int(field.Int()), true
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return int(field.Uint()), true
		}
	}
	return 0, false
}

func (p RetryPolicy) backoff(attempt int) time.Duration {
	p = p.normalized()

	var delay time.Duration
	switch p.Strategy {
	case RetryStrategyExponentialBackoff:
		delayMs := float64(p.InitialDelay.Milliseconds()) * math.Pow(p.Multiplier, float64(attempt))
		delay = time.Duration(int64(delayMs)) * time.Millisecond
	case RetryStrategyLinearBackoff:
		delay = p.InitialDelay * time.Duration(attempt+1)
	case RetryStrategyFixedDelay:
		delay = p.InitialDelay
	default:
		return 0
	}

	if delay > p.MaxDelay {
		delay = p.MaxDelay
	}

	// Apply jitter: randomize delay by ±Jitter fraction to avoid thundering herd
	if p.Jitter > 0 {
		jitterMs := float64(delay.Milliseconds()) * p.Jitter * (2*rand.Float64() - 1)
		delay = time.Duration(int64(float64(delay.Milliseconds())+jitterMs)) * time.Millisecond
		if delay < 0 {
			delay = 0
		}
	}
	return delay
}

func (p RetryPolicy) retryDelay(attempt int, err error) time.Duration {
	delay := p.backoff(attempt)
	if !p.RespectRetryAfter {
		return delay
	}
	if retryAfter, ok := extractRetryAfter(err); ok && retryAfter > delay {
		if p.MaxDelay > 0 && retryAfter > p.MaxDelay {
			return p.MaxDelay
		}
		return retryAfter
	}
	return delay
}

func extractRetryAfter(err error) (time.Duration, bool) {
	for current := err; current != nil; current = errors.Unwrap(current) {
		value := reflect.ValueOf(current)
		if !value.IsValid() {
			continue
		}
		if value.Kind() == reflect.Pointer {
			if value.IsNil() {
				continue
			}
			value = value.Elem()
		}
		if value.Kind() != reflect.Struct {
			continue
		}
		field := value.FieldByName("Response")
		if !field.IsValid() || field.IsNil() {
			continue
		}
		resp, ok := field.Interface().(*http.Response)
		if !ok || resp == nil {
			continue
		}
		text := strings.TrimSpace(resp.Header.Get("Retry-After"))
		if text == "" {
			continue
		}
		if seconds, convErr := strconv.Atoi(text); convErr == nil && seconds >= 0 {
			return time.Duration(seconds) * time.Second, true
		}
		if when, parseErr := http.ParseTime(text); parseErr == nil {
			delay := time.Until(when)
			if delay < 0 {
				delay = 0
			}
			return delay, true
		}
	}
	return 0, false
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (c *ClientWithRetry) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	if !c.policy.Enabled() {
		return c.client.CreateCompletion(ctx, messages, params)
	}

	var lastErr error
	for attempt := 0; attempt <= c.policy.MaxRetries; attempt++ {
		resp, err := c.client.CreateCompletion(ctx, messages, params)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if !isTransientError(err) || attempt >= c.policy.MaxRetries {
			return nil, err
		}
		if err := waitForRetry(ctx, c.policy.retryDelay(attempt, err)); err != nil {
			return nil, err
		}
	}

	return nil, lastErr
}

func (c *ClientWithRetry) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	if c.embedding == nil {
		return nil, ErrEmbeddingsNotSupported
	}
	if !c.policy.Enabled() {
		return c.embedding.CreateEmbeddings(ctx, req)
	}

	var lastErr error
	for attempt := 0; attempt <= c.policy.MaxRetries; attempt++ {
		resp, err := c.embedding.CreateEmbeddings(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if !isTransientError(err) || attempt >= c.policy.MaxRetries {
			return nil, err
		}
		if err := waitForRetry(ctx, c.policy.retryDelay(attempt, err)); err != nil {
			return nil, err
		}
	}

	return nil, lastErr
}

type retryStream struct {
	client   Client
	messages []Message
	params   InferenceParams
	policy   RetryPolicy
	ctx      context.Context

	innerStream Stream
	attempt     int
	hasYielded  bool
}

func (s *retryStream) Recv(ctx context.Context) (*StreamEvent, error) {
	for {
		if s.innerStream == nil {
			var err error
			for s.attempt <= s.policy.MaxRetries {
				s.innerStream, err = s.client.CreateCompletionStream(s.ctx, s.messages, s.params)
				if err == nil {
					break
				}
				if err == io.EOF {
					return nil, io.EOF
				}
				if !isTransientError(err) || s.attempt >= s.policy.MaxRetries {
					return nil, err
				}
				if err := waitForRetry(ctx, s.policy.retryDelay(s.attempt, err)); err != nil {
					return nil, err
				}
				s.attempt++
			}
			if err != nil {
				return nil, err
			}
		}

		event, err := s.innerStream.Recv(ctx)
		if err == nil {
			s.hasYielded = true
			return event, nil
		}
		if err == io.EOF {
			return nil, io.EOF
		}
		if s.hasYielded || !isTransientError(err) || s.attempt >= s.policy.MaxRetries {
			return nil, err
		}

		_ = s.innerStream.Close()
		s.innerStream = nil

		if err := waitForRetry(ctx, s.policy.retryDelay(s.attempt, err)); err != nil {
			return nil, err
		}
		s.attempt++
	}
}

func (s *retryStream) Current() *Message {
	if s.innerStream != nil {
		return s.innerStream.Current()
	}
	return nil
}

func (s *retryStream) Close() error {
	if s.innerStream != nil {
		return s.innerStream.Close()
	}
	return nil
}

func (c *ClientWithRetry) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	if !c.policy.Enabled() {
		return c.client.CreateCompletionStream(ctx, messages, params)
	}

	return &retryStream{
		client:   c.client,
		messages: messages,
		params:   params,
		policy:   c.policy,
		ctx:      ctx,
	}, nil
}

func (c *ClientWithRetry) SupportsStreaming() bool {
	capable, ok := c.client.(StreamingCapable)
	return ok && capable.SupportsStreaming()
}

type BaseStream struct {
	mu            sync.Mutex
	accumulated   Message
	pendingEvents []*StreamEvent
	blockOpen     bool
	started       bool
	finished      bool
}

func NewBaseStream() *BaseStream {
	return &BaseStream{
		accumulated: Message{
			Role:      MessageRoleAssistant,
			Timestamp: time.Now(),
			Content:   MessageContent{},
		},
	}
}

func (s *BaseStream) Accumulated() Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.accumulated
}

func (s *BaseStream) IsFinished() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.finished
}

func (s *BaseStream) HasPendingEvents() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.pendingEvents) > 0
}

func (s *BaseStream) PopEvent() *StreamEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.pendingEvents) == 0 {
		return nil
	}
	event := s.pendingEvents[0]
	s.pendingEvents = s.pendingEvents[1:]
	return event
}

func (s *BaseStream) EmitMessageStart() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		return
	}
	s.started = true
	s.pendingEvents = append(s.pendingEvents, &StreamEvent{
		Type:     EventTypeMessageStart,
		Delta:    MessageDelta{Content: MessageContent{}},
		Snapshot: s.accumulated,
	})
}

func (s *BaseStream) OpenBlock(block ContentBlock) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.blockOpen {
		s.closeBlockLocked(StopReason(""), nil)
	}
	s.accumulated.Content = append(s.accumulated.Content, block)
	s.pendingEvents = append(s.pendingEvents, &StreamEvent{
		Type:     EventTypeContentStart,
		Delta:    MessageDelta{Content: MessageContent{block}},
		Snapshot: s.accumulated,
	})
	s.blockOpen = true
}

func (s *BaseStream) AppendDelta(deltaContent ContentBlock) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.blockOpen || len(s.accumulated.Content) == 0 {
		return
	}

	lastIdx := len(s.accumulated.Content) - 1
	last := &s.accumulated.Content[lastIdx]

	switch deltaContent.Type {
	case ContentBlockTypeText:
		last.Text += deltaContent.Text

	case ContentBlockTypeThinking:
		last.Thinking += deltaContent.Thinking
		if deltaContent.Signature != "" {
			last.Signature = deltaContent.Signature
		}

	case ContentBlockTypeToolUse:
		if last.ToolUse != nil && deltaContent.ToolUse != nil {
			deltaContent.ToolUse.ID = last.ToolUse.ID
			if len(deltaContent.ToolUse.Input) > 0 {
				last.ToolUse.partialInput += string(deltaContent.ToolUse.Input)
			}
			if deltaContent.ToolUse.Signature != "" {
				last.ToolUse.Signature = deltaContent.ToolUse.Signature
			}
		}

	case ContentBlockTypeToolResult:
		if last.ToolOutput != nil && deltaContent.ToolOutput != nil {
			last.ToolOutput.Output = append(last.ToolOutput.Output, deltaContent.ToolOutput.Output...)
		}

	case ContentBlockTypeImage:
		if deltaContent.Image != nil {
			last.Image = deltaContent.Image
		}

	case ContentBlockTypeAudio:
		if deltaContent.Audio != nil {
			last.Audio = deltaContent.Audio
		}

	case ContentBlockTypeVideo:
		if deltaContent.Video != nil {
			last.Video = deltaContent.Video
		}

	case ContentBlockTypeDocument:
		if deltaContent.Document != nil {
			if last.Document != nil && last.Document.Type == DocumentSourceTypeText && deltaContent.Document.Type == DocumentSourceTypeText {
				last.Document.Text += deltaContent.Document.Text
			} else {
				last.Document = deltaContent.Document
			}
		}
	}
	s.pendingEvents = append(s.pendingEvents, &StreamEvent{
		Type:     EventTypeContentDelta,
		Delta:    MessageDelta{Content: MessageContent{deltaContent}},
		Snapshot: s.accumulated,
	})
}

func (s *BaseStream) CloseCurrentBlock(stopReason StopReason, usage *Usage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closeBlockLocked(stopReason, usage)
}

func (s *BaseStream) closeBlockLocked(stopReason StopReason, usage *Usage) {
	if !s.blockOpen || len(s.accumulated.Content) == 0 {
		return
	}

	lastBlock := s.accumulated.Content[len(s.accumulated.Content)-1]
	deltaBlock := ContentBlock{Type: lastBlock.Type}

	if lastBlock.Type == ContentBlockTypeToolUse && lastBlock.ToolUse != nil {
		lastBlock.ToolUse.Input = finalizeToolUseInput(lastBlock.ToolUse)
		deltaBlock.ToolUse = &ToolUse{
			ID:   lastBlock.ToolUse.ID,
			Name: lastBlock.ToolUse.Name,
		}
	}

	delta := MessageDelta{Content: MessageContent{deltaBlock}}
	if stopReason != "" {
		delta.StopReason = stopReason
	}
	if usage != nil {
		delta.Usage = usage
	}

	s.pendingEvents = append(s.pendingEvents, &StreamEvent{
		Type:     EventTypeContentEnd,
		Delta:    delta,
		Snapshot: s.accumulated,
	})
	s.blockOpen = false
}

func (s *BaseStream) Finish(stopReason StopReason, usage *Usage) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.finished {
		return
	}

	if stopReason != "" {
		s.accumulated.StopReason = stopReason
	}
	if usage != nil {
		s.accumulated.Usage = usage
	}

	if s.blockOpen {
		s.closeBlockLocked(stopReason, usage)
	}
	s.pendingEvents = append(s.pendingEvents, &StreamEvent{
		Type:     EventTypeMessageEnd,
		Delta:    MessageDelta{StopReason: stopReason, Usage: usage},
		Snapshot: s.accumulated,
	})

	s.finished = true
}

func finalizeToolUseInput(tool *ToolUse) json.RawMessage {
	if tool == nil {
		return nil
	}

	original := normalizeToolUseRawJSON(tool.Input)
	if tool.partialInput == "" {
		return original
	}

	candidates := []json.RawMessage{
		json.RawMessage(tool.partialInput),
	}

	originalText := strings.TrimSpace(string(original))
	partialText := tool.partialInput
	trimmedPartial := strings.TrimSpace(partialText)
	if originalText != "" && originalText != "{}" {
		if strings.HasPrefix(trimmedPartial, ",") {
			withoutClosing := trimTrailingJSONWhitespace(originalText)
			if strings.HasSuffix(withoutClosing, "}") {
				withoutClosing = trimTrailingJSONWhitespace(withoutClosing[:len(withoutClosing)-1])
				candidates = append([]json.RawMessage{json.RawMessage(withoutClosing + partialText)}, candidates...)
			}
		}
		candidates = append(candidates, json.RawMessage(originalText+partialText))
	}

	for _, candidate := range candidates {
		normalized := normalizeToolUseRawJSON(candidate)
		if len(normalized) > 0 && json.Valid(normalized) {
			return normalized
		}
	}

	return normalizeToolUseRawJSON(json.RawMessage(tool.partialInput))
}

// ToolInputRepairKind classifies how RepairToolInput coerced bytes into
// valid JSON. Callers should look at this when deciding whether to execute
// the tool: RepairKindLossless changes are safe to run, RepairKindTruncated
// means the value was cut off and the tool should probably NOT execute (or
// at least surface a warning).
type ToolInputRepairKind int

const (
	// RepairKindNone — input was already valid JSON. cleaned == raw.
	RepairKindNone ToolInputRepairKind = iota

	// RepairKindLossless — cleaned the bytes without losing data. Covers:
	//   • empty/whitespace input replaced with {}
	//   • stray control bytes inside strings escaped (\n, \t, …)
	// The semantic content is preserved.
	RepairKindLossless

	// RepairKindTruncated — closed a dangling JSON structure by appending
	// missing quotes / brackets / braces. Almost always means the model
	// hit its max_tokens cap mid-stream and the closing was synthesised
	// from scratch. The PAYLOAD inside the now-valid JSON is truncated:
	// strings end where the model stopped emitting, arrays/objects close
	// short. Tool runners should refuse to execute on this kind so the
	// model retries with a smaller payload rather than running with
	// silently-cut-off arguments (e.g. half a file write).
	RepairKindTruncated
)

// Repaired reports whether the repair changed the input at all.
func (k ToolInputRepairKind) Repaired() bool { return k != RepairKindNone }

// LostData reports whether the repair almost certainly dropped content
// (i.e. truncated mid-payload). Callers should treat this as a "do not
// execute" signal for stateful tools.
func (k ToolInputRepairKind) LostData() bool { return k == RepairKindTruncated }

// RepairToolInput tries to coerce a tool_use input — which may be malformed
// (typically because the model hit max_tokens mid-call, or because raw
// control characters from a JS template literal slipped through unescaped) —
// into valid JSON. It returns:
//
//   - cleaned: the bytes the caller should persist to history.
//   - kind:    classification of the repair (see ToolInputRepairKind).
//   - err:     non-nil when no straightforward repair produces valid JSON.
//     In that case `cleaned` is the original bytes and callers should fall
//     back to SafeToolInputFallback() before persisting, so history-replay
//     to the provider succeeds.
//
// This is the central place to evolve repair heuristics — both the streaming
// finaliser and the runtime tool-call dispatch path should funnel through it
// so failure modes stay consistent across runtimes (v1 actor loop, v2 run
// loop, future replays).
func RepairToolInput(raw json.RawMessage) (cleaned json.RawMessage, kind ToolInputRepairKind, err error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return []byte("{}"), RepairKindLossless, nil
	}
	if json.Valid(raw) {
		return raw, RepairKindNone, nil
	}
	// First pass: escape stray control bytes inside strings. Catches the
	// case where a model dumped a literal newline or tab into a string
	// without escaping it. No data is lost here — we just re-encode bytes
	// that should have been escaped at emission time.
	escaped := escapeToolUseControlChars(raw)
	if json.Valid(escaped) {
		return escaped, RepairKindLossless, nil
	}
	// Second pass: a max_tokens truncation usually leaves a dangling open
	// string and missing braces. Try to close the structure greedily.
	// THIS path loses data — the model wanted to emit more bytes for the
	// open string, but ran out and we forced it shut at the cutoff.
	if patched, ok := closeDanglingJSONObject(escaped); ok && json.Valid(patched) {
		return patched, RepairKindTruncated, nil
	}
	return raw, RepairKindNone, fmt.Errorf("tool input is not valid JSON and could not be repaired")
}

// SafeToolInputFallback returns the canonical "empty input" tool-call body.
// Callers persist this to history when RepairToolInput fails so the next
// model-call doesn't get rejected by the provider's "tool use input must be
// valid JSON" validator.
func SafeToolInputFallback() json.RawMessage {
	return json.RawMessage([]byte("{}"))
}

// closeDanglingJSONObject is a best-effort patch for truncated JSON objects.
// It walks the bytes tracking string/escape/brace state. If the walk ends
// while still inside an unterminated string, it closes the string. If the
// brace depth is positive at end-of-input, it appends enough `}` to balance.
// Trailing commas inside objects/arrays are stripped before closing.
//
// Returns ok=false when the input is structurally hopeless (e.g. ends inside
// an array literal we didn't track, or has unbalanced `]`).
func closeDanglingJSONObject(raw []byte) ([]byte, bool) {
	var (
		braceDepth   int
		bracketDepth int
		inString     bool
		escape       bool
	)
	for _, b := range raw {
		if escape {
			escape = false
			continue
		}
		if inString {
			switch b {
			case '\\':
				escape = true
			case '"':
				inString = false
			}
			continue
		}
		switch b {
		case '"':
			inString = true
		case '{':
			braceDepth++
		case '}':
			braceDepth--
			if braceDepth < 0 {
				return nil, false
			}
		case '[':
			bracketDepth++
		case ']':
			bracketDepth--
			if bracketDepth < 0 {
				return nil, false
			}
		}
	}
	if braceDepth == 0 && bracketDepth == 0 && !inString && !escape {
		return raw, true
	}
	out := make([]byte, 0, len(raw)+braceDepth+bracketDepth+1)
	out = append(out, raw...)
	if escape {
		// Trailing backslash — drop it; otherwise the closing quote we add
		// would itself be escaped.
		out = out[:len(out)-1]
	}
	if inString {
		out = append(out, '"')
	}
	// Strip any trailing comma/whitespace immediately before structural
	// closes — `{"a":1,}` is not valid JSON.
	for len(out) > 0 {
		last := out[len(out)-1]
		if last == ' ' || last == '\t' || last == '\n' || last == '\r' || last == ',' {
			out = out[:len(out)-1]
			continue
		}
		break
	}
	for i := 0; i < bracketDepth; i++ {
		out = append(out, ']')
	}
	for i := 0; i < braceDepth; i++ {
		out = append(out, '}')
	}
	return out, true
}

func normalizeToolUseRawJSON(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || json.Valid(raw) {
		return raw
	}
	repaired := escapeToolUseControlChars(raw)
	if json.Valid(repaired) {
		return repaired
	}
	return raw
}

func escapeToolUseControlChars(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}

	var builder strings.Builder
	builder.Grow(len(raw))

	inString := false
	escaped := false

	for i := 0; i < len(raw); i++ {
		ch := raw[i]
		if !inString {
			builder.WriteByte(ch)
			if ch == '"' {
				inString = true
			}
			continue
		}

		if escaped {
			builder.WriteByte(ch)
			escaped = false
			continue
		}

		switch ch {
		case '\\':
			builder.WriteByte(ch)
			escaped = true
		case '"':
			builder.WriteByte(ch)
			inString = false
		case '\b':
			builder.WriteString(`\b`)
		case '\f':
			builder.WriteString(`\f`)
		case '\n':
			builder.WriteString(`\n`)
		case '\r':
			builder.WriteString(`\r`)
		case '\t':
			builder.WriteString(`\t`)
		default:
			if ch < 0x20 {
				builder.WriteString(`\u00`)
				builder.WriteByte("0123456789abcdef"[ch>>4])
				builder.WriteByte("0123456789abcdef"[ch&0x0f])
				continue
			}
			builder.WriteByte(ch)
		}
	}

	return json.RawMessage(builder.String())
}

func trimTrailingJSONWhitespace(value string) string {
	return strings.TrimRight(value, " \t\r\n")
}

func (s *BaseStream) SetStopReason(reason StopReason) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.accumulated.StopReason = reason
}

func (s *BaseStream) SetUsage(usage *Usage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.accumulated.Usage = usage
}

func (s *BaseStream) IsBlockOpen() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.blockOpen
}

func (s *BaseStream) CurrentBlockType() ContentBlockType {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.blockOpen || len(s.accumulated.Content) == 0 {
		return ""
	}
	return s.accumulated.Content[len(s.accumulated.Content)-1].Type
}

type CompletionRequest struct {
	Messages []Message       `json:"messages"`
	Params   InferenceParams `json:"params"`
	Metadata map[string]any  `json:"metadata,omitempty"`
}

type ConcurrentConfig struct {
	MaxConcurrency int
	MinDelay       time.Duration
	StopOnError    bool
}

func RunConcurrentCompletions(ctx context.Context, client Client, reqs []CompletionRequest, cfg ConcurrentConfig) ([]*Message, error) {
	return runConcurrent(ctx, reqs, cfg, func(ctx context.Context, req CompletionRequest) (*Message, error) {
		return client.CreateCompletion(ctx, req.Messages, req.Params)
	})
}

func RunConcurrentEmbeddings(ctx context.Context, client EmbeddingCapable, reqs []EmbeddingRequest, cfg ConcurrentConfig) ([]*EmbeddingResponse, error) {
	if client == nil {
		return nil, ErrEmbeddingsNotSupported
	}
	return runConcurrent(ctx, reqs, cfg, client.CreateEmbeddings)
}

func runConcurrent[T any, R any](ctx context.Context, reqs []T, cfg ConcurrentConfig, fn func(context.Context, T) (R, error)) ([]R, error) {
	cfg = normalizeConcurrentConfig(cfg)
	if len(reqs) == 0 {
		return nil, nil
	}

	runCtx := ctx
	cancel := func() {}
	if cfg.StopOnError {
		var cancelFn context.CancelFunc
		runCtx, cancelFn = context.WithCancel(ctx)
		cancel = cancelFn
	}
	defer cancel()

	results := make([]R, len(reqs))
	errs := make([]error, len(reqs))

	var rateMu sync.Mutex
	var lastStart time.Time

	acquireStartSlot := func(ctx context.Context) error {
		if cfg.MinDelay <= 0 {
			return nil
		}
		rateMu.Lock()
		defer rateMu.Unlock()
		if lastStart.IsZero() {
			lastStart = time.Now()
			return nil
		}
		wait := cfg.MinDelay - time.Since(lastStart)
		if wait > 0 {
			timer := time.NewTimer(wait)
			defer timer.Stop()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-timer.C:
			}
		}
		lastStart = time.Now()
		return nil
	}

	type job struct {
		index int
		req   T
	}

	jobs := make(chan job)
	workerCount := min(cfg.MaxConcurrency, len(reqs))
	var wg sync.WaitGroup
	for range workerCount {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				if err := acquireStartSlot(runCtx); err != nil {
					errs[job.index] = err
					if cfg.StopOnError {
						cancel()
					}
					continue
				}

				result, err := fn(runCtx, job.req)
				results[job.index] = result
				errs[job.index] = err
				if err != nil && cfg.StopOnError {
					cancel()
				}
			}
		}()
	}

	for i, req := range reqs {
		if runCtx.Err() != nil {
			for j := i; j < len(reqs); j++ {
				errs[j] = runCtx.Err()
			}
			break
		}

		select {
		case <-runCtx.Done():
			for j := i; j < len(reqs); j++ {
				errs[j] = runCtx.Err()
			}
		case jobs <- job{index: i, req: req}:
		}
	}
	close(jobs)
	wg.Wait()

	var firstErr error
	for _, err := range errs {
		if err != nil && !errors.Is(err, context.Canceled) {
			firstErr = err
			break
		}
	}
	if firstErr != nil {
		return results, firstErr
	}
	if err := runCtx.Err(); err != nil && !errors.Is(err, context.Canceled) {
		return results, err
	}
	return results, nil
}

func normalizeConcurrentConfig(cfg ConcurrentConfig) ConcurrentConfig {
	if cfg.MaxConcurrency < 1 {
		cfg.MaxConcurrency = runtime.NumCPU()
	}
	return cfg
}
