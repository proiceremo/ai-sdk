package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	. "ai-sdk"
)

func TestGoogleCreateCompletionStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify streaming endpoint is called
		if r.URL.Path != "/v1beta/models/gemini-test:streamGenerateContent" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.URL.Query().Get("alt") != "sse" {
			t.Fatalf("expected alt=sse query parameter")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Flush headers immediately
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}

		responses := []googleGenerateContentResponse{
			{
				Candidates: []googleCandidate{{
					Content: &googleContent{
						Role: "model",
						Parts: []googlePart{{
							Text: stringPtr("Hello"),
						}},
					},
				}},
			},
			{
				Candidates: []googleCandidate{{
					Content: &googleContent{
						Role: "model",
						Parts: []googlePart{{
							Text: stringPtr(" world"),
						}},
					},
				}},
			},
			{
				Candidates: []googleCandidate{{
					Content: &googleContent{
						Role: "model",
						Parts: []googlePart{{
							Text: stringPtr("!"),
						}},
					},
					FinishReason: stringPtr("STOP"),
				}},
				UsageMetadata: &googleUsageMetadata{
					PromptTokenCount:     intPtr(5),
					CandidatesTokenCount: intPtr(3),
					TotalTokenCount:      intPtr(8),
				},
			},
		}

		for _, resp := range responses {
			data, _ := json.Marshal(resp)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			time.Sleep(10 * time.Millisecond)
		}
	}))
	defer server.Close()

	client := newGoogleClient("test-key", server.URL, defaultGoogleAPIVersion, "", "", false, server.Client())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := client.CreateCompletionStream(ctx, []Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("Say hello")},
	}}, InferenceParams{
		Model: "gemini-test",
	})
	if err != nil {
		t.Fatalf("CreateCompletionStream returned error: %v", err)
	}
	defer stream.Close()

	var events []*StreamEvent
	eventTypes := make(map[StreamEventType]int)

	for {
		event, err := stream.Recv(ctx)
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			t.Fatalf("Recv returned error: %v", err)
		}
		if event == nil {
			t.Fatal("expected event, got nil")
		}
		events = append(events, event)
		eventTypes[event.Type]++
	}

	// Should have received events
	if len(events) == 0 {
		t.Fatal("expected events, got none")
	}

	// Should have at least: message_start, content_start, content_delta (3x), content_end, message_end
	if eventTypes[EventTypeMessageStart] != 1 {
		t.Errorf("expected 1 message_start event, got %d", eventTypes[EventTypeMessageStart])
	}
	if eventTypes[EventTypeMessageEnd] != 1 {
		t.Errorf("expected 1 message_end event, got %d", eventTypes[EventTypeMessageEnd])
	}

	// Check accumulated message
	accumulated := stream.Current()
	if accumulated == nil {
		t.Fatal("expected accumulated message")
	}
	if accumulated.Content.Text() != "Hello world!" {
		t.Fatalf("expected 'Hello world!', got %q", accumulated.Content.Text())
	}
	if accumulated.StopReason != StopReasonEndTurn {
		t.Fatalf("expected stop reason end_turn, got %q", accumulated.StopReason)
	}
}

func TestGoogleCreateCompletionStreamHandlesEmptyCandidates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}

		// Send an empty candidates response (heartbeat/keepalive)
		resp := googleGenerateContentResponse{
			Candidates: []googleCandidate{},
		}
		data, _ := json.Marshal(resp)
		fmt.Fprintf(w, "data: %s\n\n", data)

		// Then send actual content
		resp2 := googleGenerateContentResponse{
			Candidates: []googleCandidate{{
				Content: &googleContent{
					Role: "model",
					Parts: []googlePart{{
						Text: stringPtr("Hello"),
					}},
				},
				FinishReason: stringPtr("STOP"),
			}},
		}
		data2, _ := json.Marshal(resp2)
		fmt.Fprintf(w, "data: %s\n\n", data2)
	}))
	defer server.Close()

	client := newGoogleClient("test-key", server.URL, defaultGoogleAPIVersion, "", "", false, server.Client())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := client.CreateCompletionStream(ctx, []Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("Test")},
	}}, InferenceParams{
		Model: "gemini-test",
	})
	if err != nil {
		t.Fatalf("CreateCompletionStream returned error: %v", err)
	}
	defer stream.Close()

	eventCount := 0
	for {
		event, err := stream.Recv(ctx)
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			t.Fatalf("Recv returned error: %v", err)
		}
		if event != nil {
			eventCount++
		}
	}

	if eventCount == 0 {
		t.Fatal("expected some events, got none")
	}
}

func TestGoogleCreateCompletionStreamWithToolUse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}

		responses := []googleGenerateContentResponse{
			{
				Candidates: []googleCandidate{{
					Content: &googleContent{
						Role: "model",
						Parts: []googlePart{{
							FunctionCall: &googleFunctionCall{
								ID:   stringPtr("call-1"),
								Name: stringPtr("get_weather"),
								Args: json.RawMessage(`{"location": "London"}`),
							},
						}},
					},
					FinishReason: stringPtr("STOP"),
				}},
			},
		}

		for _, resp := range responses {
			data, _ := json.Marshal(resp)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}
	}))
	defer server.Close()

	client := newGoogleClient("test-key", server.URL, defaultGoogleAPIVersion, "", "", false, server.Client())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := client.CreateCompletionStream(ctx, []Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("What's the weather?")},
	}}, InferenceParams{
		Model: "gemini-test",
		Tools: []Tool{&mockTool{name: "get_weather"}},
	})
	if err != nil {
		t.Fatalf("CreateCompletionStream returned error: %v", err)
	}
	defer stream.Close()

	var gotToolUse bool
	for {
		event, err := stream.Recv(ctx)
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			t.Fatalf("Recv returned error: %v", err)
		}
		if event != nil && len(event.Delta.Content) > 0 {
			for _, block := range event.Delta.Content {
				if block.Type == ContentBlockTypeToolUse {
					gotToolUse = true
					if block.ToolUse == nil || block.ToolUse.Name != "get_weather" {
						t.Fatalf("unexpected tool use: %#v", block.ToolUse)
					}
				}
			}
		}
	}

	if !gotToolUse {
		t.Fatal("expected to receive tool use event")
	}

	accumulated := stream.Current()
	toolCalls := accumulated.Content.ToolCalls()
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].Name != "get_weather" {
		t.Fatalf("expected tool name get_weather, got %s", toolCalls[0].Name)
	}
	if accumulated.StopReason != StopReasonToolUse {
		t.Fatalf("expected stop reason tool_use, got %q", accumulated.StopReason)
	}
}

type mockTool struct {
	name        string
	description string
	schema      map[string]any
}

func (m *mockTool) Schema() ToolSchema {
	return ToolSchema{
		Name:        m.name,
		Description: m.description,
		InputSchema: m.schema,
	}
}

func (m *mockTool) Execute(ctx ToolContext, input map[string]any) ToolResult {
	return ToolResult{}
}
