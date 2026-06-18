package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"

	llm "github.com/proiceremo/ai-sdk"
)

func responsesInput(messages []llm.Message) ([]any, error) {
	var out []any
	for _, msg := range messages {
		for _, block := range msg.Content {
			if block.ToolOutput == nil {
				continue
			}
			callID, _ := splitResponsesToolID(block.ToolOutput.ToolUseID)
			out = append(out, map[string]any{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  llm.SanitizeSurrogates(block.ToolOutput.Output.ToString()),
			})
		}
		switch msg.Role {
		case llm.MessageRoleUser:
			out = append(out, map[string]any{
				"role":    "user",
				"content": responsesInputContent(msg.Content),
			})
		case llm.MessageRoleAssistant:
			for _, block := range msg.Content {
				switch block.Type {
				case llm.ContentBlockTypeText:
					out = append(out, map[string]any{
						"type":   "message",
						"role":   "assistant",
						"content": []any{
							map[string]any{
								"type":        "output_text",
								"text":        llm.SanitizeSurrogates(block.Text),
								"annotations": []any{},
							},
						},
						"status": "completed",
					})
				case llm.ContentBlockTypeToolUse:
					if block.ToolUse != nil {
						callID, itemID := splitResponsesToolID(block.ToolUse.ID)
						out = append(out, map[string]any{
							"type":      "function_call",
							"id":        itemID,
							"call_id":   callID,
							"name":      block.ToolUse.Name,
							"arguments": string(block.ToolUse.Input),
						})
					}
				}
			}
		}
	}
	return out, nil
}

func responsesInputContent(content llm.MessageContent) []any {
	parts := []any{}
	for _, block := range content {
		switch block.Type {
		case llm.ContentBlockTypeText:
			parts = append(parts, map[string]any{"type": "input_text", "text": llm.SanitizeSurrogates(block.Text)})
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

func responsesTools(tools []llm.Tool) []any {
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

func responsesReasoningEffort(level llm.ThinkingLevel) string {
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

func splitResponsesToolID(id string) (callID, itemID string) {
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

type responsesStream struct {
	base        *llm.BaseStream
	ch          chan *llm.StreamEvent
	done        chan struct{}
	once        sync.Once
	mu          sync.Mutex
	err         error
	toolArgLen  int
	requestTier *string
}

func newResponsesStream(requestTier *string) *responsesStream {
	s := &responsesStream{
		base:        llm.NewBaseStream(),
		ch:          make(chan *llm.StreamEvent, 64),
		done:        make(chan struct{}),
		requestTier: requestTier,
	}
	s.base.EmitMessageStart()
	s.flush()
	return s
}

func (s *responsesStream) Recv(ctx context.Context) (*llm.StreamEvent, error) {
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

func (s *responsesStream) Current() *llm.Message {
	msg := s.base.Accumulated()
	return &msg
}

func (s *responsesStream) Close() error {
	s.finish(nil)
	return nil
}

func (s *responsesStream) fail(err error) {
	s.mu.Lock()
	s.err = err
	s.mu.Unlock()
	s.finish(nil)
}

func (s *responsesStream) finish(usage *llm.Usage) {
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

func (s *responsesStream) flush() {
	for {
		event := s.base.PopEvent()
		if event == nil {
			return
		}
		s.ch <- event
	}
}

func (s *responsesStream) addDiagnostic(text string) {
	_ = text
}

func (s *responsesStream) handleEvent(event map[string]any) (bool, error) {
	eventType, _ := event["type"].(string)
	switch eventType {
	case "error":
		return false, fmt.Errorf("responses error: %v", event)
	case "response.failed":
		return false, fmt.Errorf("responses failed: %v", event)
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
		usage := s.responsesUsage(event)
		s.finish(usage)
		return true, nil
	}
	return false, nil
}

func (s *responsesStream) openItem(item map[string]any) {
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

func (s *responsesStream) doneItem(item map[string]any) {
	itemType, _ := item["type"].(string)
	switch itemType {
	case "message":
		s.base.CloseCurrentBlock("", nil)
	case "reasoning":
		s.base.CloseCurrentBlock("", nil)
	case "function_call":
		s.base.CloseCurrentBlock(llm.StopReasonToolUse, nil)
	case "image_generation_call":
		id, _ := item["id"].(string)
		revPrompt, _ := item["revised_prompt"].(string)
		prompt, _ := item["prompt"].(string)
		if prompt == "" {
			prompt = revPrompt
		}
		args := map[string]any{"prompt": prompt}
		argsBytes, _ := json.Marshal(args)
		s.base.OpenBlock(llm.ContentBlock{
			Type: llm.ContentBlockTypeToolUse,
			ToolUse: &llm.ToolUse{
				ID:    "call_" + id + "|" + id,
				Name:  "imagegen",
				Input: argsBytes,
			},
		})
		s.base.CloseCurrentBlock(llm.StopReasonToolUse, nil)
	case "web_search_call":
		id, _ := item["id"].(string)
		action := item["action"]
		if action == nil {
			action = item["results"]
		}
		actionBytes, _ := json.Marshal(action)
		s.base.OpenBlock(llm.ContentBlock{
			Type: llm.ContentBlockTypeToolUse,
			ToolUse: &llm.ToolUse{
				ID:    "call_" + id + "|" + id,
				Name:  "web_run",
				Input: actionBytes,
			},
		})
		s.base.CloseCurrentBlock(llm.StopReasonToolUse, nil)
	}
	s.flush()
}

func (s *responsesStream) scanSSE(body io.Reader) error {
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

func (s *responsesStream) responsesUsage(event map[string]any) *llm.Usage {
	response, _ := event["response"].(map[string]any)
	rawUsage, _ := response["usage"].(map[string]any)
	if rawUsage == nil {
		return nil
	}
	input := intFromAny(rawUsage["input_tokens"])
	output := intFromAny(rawUsage["output_tokens"])
	total := intFromAny(rawUsage["total_tokens"])
	var cacheRead, inputAudio int
	if details, ok := rawUsage["input_tokens_details"].(map[string]any); ok {
		cacheRead = intFromAny(details["cached_tokens"])
		inputAudio = intFromAny(details["audio_tokens"])
	}
	if cacheRead > input {
		cacheRead = input
	}
	input -= cacheRead
	if total == 0 {
		total = input + cacheRead + output
	}
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

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
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

