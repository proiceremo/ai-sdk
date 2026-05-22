package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	. "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/tools"
)

func TestOpenAIBuildCompletionParamsIncludesResponseFormat(t *testing.T) {
	client := &OpenAIClient{}
	schema := tools.ReflectValue(struct {
		Value string `json:"value" jsonschema:"required"`
	}{})

	params, err := client.buildCompletionParams([]Message{
		{Role: MessageRoleUser, Content: MessageContent{NewTextContentBlock("hello")}},
	}, InferenceParams{
		Model: "test-model",
		ResponseFormat: &ResponseFormat{
			Type:       ResponseFormatTypeJSONObject,
			JSONSchema: schema,
		},
	})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}

	if params.ResponseFormat == nil || params.ResponseFormat.Type != "json_schema" {
		t.Fatalf("expected json schema response format, got %#v", params.ResponseFormat)
	}
	if params.ResponseFormat.JSONSchema == nil || params.ResponseFormat.JSONSchema.Name != "structured_output" {
		t.Fatalf("unexpected schema name: %#v", params.ResponseFormat)
	}
	if params.ResponseFormat.JSONSchema.Schema == nil {
		t.Fatal("expected response schema payload to be set")
	}
}

func TestOpenAIBuildStreamingCompletionParamsIncludesUsage(t *testing.T) {
	client := &OpenAIClient{}
	params, err := client.buildStreamingCompletionParams([]Message{
		{Role: MessageRoleUser, Content: MessageContent{NewTextContentBlock("hello")}},
	}, InferenceParams{
		Model: "test-model",
	})
	if err != nil {
		t.Fatalf("buildStreamingCompletionParams returned error: %v", err)
	}

	if !params.Stream {
		t.Fatalf("expected streaming params to enable stream, got %#v", params)
	}
	if params.StreamOptions == nil || !params.StreamOptions.IncludeUsage {
		t.Fatalf("expected streaming params to request usage, got %#v", params.StreamOptions)
	}
}

func TestOpenAIBuildCompletionParamsPreservesStrictToolSchemas(t *testing.T) {
	client := &OpenAIClient{}
	tool := tools.NewGenericTool(
		"lookup_weather",
		"Look up the weather for a location",
		func(ctx ToolContext, input struct {
			Location string `json:"location" jsonschema:"required"`
		}) ToolResult {
			return ToolResult{}
		},
	).WithStrict(true)

	params, err := client.buildCompletionParams([]Message{
		{Role: MessageRoleUser, Content: MessageContent{NewTextContentBlock("hello")}},
	}, InferenceParams{
		Model: "test-model",
		Tools: []Tool{tool},
	})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}

	if len(params.Tools) != 1 {
		t.Fatalf("expected one tool, got %#v", params.Tools)
	}
	if params.Tools[0].Function == nil || params.Tools[0].Function.Strict == nil || !*params.Tools[0].Function.Strict {
		t.Fatalf("expected strict tool definition, got %#v", params.Tools[0])
	}
	if got := params.Tools[0].Function.Parameters["additionalProperties"]; got != false {
		t.Fatalf("expected additionalProperties=false, got %#v", params.Tools[0].Function.Parameters)
	}
}

func TestOpenAIBuildCompletionParamsPreservesDocumentTextAndFiles(t *testing.T) {
	client := &OpenAIClient{}
	params, err := client.buildCompletionParams([]Message{
		{
			Role: MessageRoleUser,
			Content: MessageContent{{
				Type: ContentBlockTypeDocument,
				Document: &DocumentSource{
					Type:      DocumentSourceTypeBase64,
					Data:      "data:application/pdf;base64,ZW1iZWRkZWQ=",
					MediaType: "application/pdf",
					Name:      "doc.pdf",
					Title:     "Doc Title",
					Context:   "Doc Context",
					Text:      "Doc Body",
				},
			}},
		},
	}, InferenceParams{Model: "test-model"})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}

	if len(params.Messages) != 1 {
		t.Fatalf("expected one message, got %#v", params.Messages)
	}

	parts, ok := params.Messages[0].Content.([]any)
	if !ok {
		t.Fatalf("expected user content parts slice, got %#v", params.Messages[0].Content)
	}
	if len(parts) != 2 {
		t.Fatalf("expected text + file parts, got %#v", parts)
	}

	textPart, ok := parts[0].(openAIChatCompletionTextPart)
	if !ok {
		t.Fatalf("expected first part to be text, got %#v", parts[0])
	}
	if textPart.Text != "Doc Title\nDoc Context\nDoc Body" {
		t.Fatalf("unexpected document text part: %#v", textPart)
	}

	filePart, ok := parts[1].(openAIChatCompletionFilePart)
	if !ok {
		t.Fatalf("expected second part to be file, got %#v", parts[1])
	}
	if filePart.File.FileData == nil || *filePart.File.FileData != "ZW1iZWRkZWQ=" {
		t.Fatalf("expected stripped base64 payload, got %#v", filePart.File)
	}
	if filePart.File.Filename == nil || *filePart.File.Filename != "doc.pdf" {
		t.Fatalf("unexpected filename: %#v", filePart.File)
	}
}

func TestOpenAIBuildCompletionParamsIncludesNamedToolChoice(t *testing.T) {
	client := &OpenAIClient{}
	tool := tools.NewGenericTool(
		"submit_structured_output",
		"Submit structured output",
		func(ctx ToolContext, input struct {
			Value string `json:"value"`
		}) ToolResult {
			return ToolResult{}
		},
	).WithStrict(true)

	params, err := client.buildCompletionParams([]Message{
		{Role: MessageRoleUser, Content: MessageContent{NewTextContentBlock("hello")}},
	}, InferenceParams{
		Model: "test-model",
		Tools: []Tool{tool},
		ToolChoice: &ToolChoice{
			Type: ToolChoiceTypeTool,
			Name: "submit_structured_output",
		},
	})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}
	if params.ToolChoice == nil || params.ToolChoice.Type != "function" || params.ToolChoice.Function == nil || params.ToolChoice.Function.Name != "submit_structured_output" {
		t.Fatalf("expected named tool choice, got %#v", params.ToolChoice)
	}
}

func TestOpenAIBuildCompletionParamsTurnsToolResultsIntoToolMessages(t *testing.T) {
	client := &OpenAIClient{}
	params, err := client.buildCompletionParams([]Message{
		{
			Role: MessageRoleUser,
			Content: MessageContent{
				NewToolResultContentBlock(ToolOutput{
					ToolUseID: "tool-1",
					Output: MessageContent{{
						Type: ContentBlockTypeDocument,
						Document: &DocumentSource{
							Type:    DocumentSourceTypeText,
							Title:   "Doc Title",
							Context: "Doc Context",
							Text:    "Doc Body",
						},
					}},
				}),
			},
		},
	}, InferenceParams{Model: "test-model"})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}

	if len(params.Messages) != 1 {
		t.Fatalf("expected one tool message, got %#v", params.Messages)
	}
	if params.Messages[0].Role != "tool" {
		t.Fatalf("expected tool role, got %#v", params.Messages[0])
	}
	if got, ok := params.Messages[0].Content.(string); !ok || got != "Doc Title\nDoc Context\nDoc Body" {
		t.Fatalf("unexpected tool content: %#v", params.Messages[0].Content)
	}
}

func TestOpenAIBuildCompletionParamsAddsRichToolResultSidecar(t *testing.T) {
	client := &OpenAIClient{}
	params, err := client.buildCompletionParams([]Message{
		{
			Role: MessageRoleUser,
			Content: MessageContent{
				NewToolResultContentBlock(ToolOutput{
					ToolUseID: "tool-1",
					Name:      "js_execute",
					Output: MessageContent{
						NewTextContentBlock("screenshot captured"),
						NewImageContentBlockFromBase64("iVBORw0KGgo=", "image/png"),
					},
				}),
			},
		},
	}, InferenceParams{Model: "test-model"})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}

	if len(params.Messages) != 2 {
		t.Fatalf("expected tool message plus rich sidecar, got %#v", params.Messages)
	}
	if params.Messages[0].Role != "tool" || params.Messages[0].ToolCallID != "tool-1" {
		t.Fatalf("unexpected tool message: %#v", params.Messages[0])
	}
	if params.Messages[1].Role != "user" {
		t.Fatalf("expected rich sidecar to be a user message, got %#v", params.Messages[1])
	}
	parts, ok := params.Messages[1].Content.([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("expected text + image sidecar parts, got %#v", params.Messages[1].Content)
	}
	if image, ok := parts[1].(openAIChatCompletionImagePart); !ok || image.Type != "image_url" || !strings.Contains(image.ImageURL.URL, "data:image/png;base64,") {
		t.Fatalf("expected image_url sidecar part, got %#v", parts[1])
	}
}

func TestOpenAIBuildCompletionParamsSplitsMixedUserAndToolResultContent(t *testing.T) {
	client := &OpenAIClient{}
	params, err := client.buildCompletionParams([]Message{
		{
			Role: MessageRoleUser,
			Content: MessageContent{
				NewTextContentBlock("before"),
				NewToolResultContentBlock(ToolOutput{
					ToolUseID: "tool-1",
					Output:    MessageContent{NewTextContentBlock("tool-output")},
				}),
				NewTextContentBlock("after"),
			},
		},
	}, InferenceParams{Model: "test-model"})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}

	if len(params.Messages) != 3 {
		t.Fatalf("expected three messages, got %#v", params.Messages)
	}
	if params.Messages[0].Role != "user" {
		t.Fatalf("expected first message to be user, got %#v", params.Messages[0])
	}
	if got, ok := params.Messages[0].Content.(string); !ok || got != "before" {
		t.Fatalf("unexpected first message content: %#v", params.Messages[0].Content)
	}
	if params.Messages[1].Role != "tool" || params.Messages[1].ToolCallID != "tool-1" {
		t.Fatalf("unexpected tool message: %#v", params.Messages[1])
	}
	if got, ok := params.Messages[1].Content.(string); !ok || got != "tool-output" {
		t.Fatalf("unexpected tool message content: %#v", params.Messages[1].Content)
	}
	if params.Messages[2].Role != "user" {
		t.Fatalf("expected third message to be user, got %#v", params.Messages[2])
	}
	if got, ok := params.Messages[2].Content.(string); !ok || got != "after" {
		t.Fatalf("unexpected third message content: %#v", params.Messages[2].Content)
	}
}

func TestOpenAIBuildCompletionParamsIgnoresAssistantThinkingBlocks(t *testing.T) {
	client := &OpenAIClient{}
	params, err := client.buildCompletionParams([]Message{
		{
			Role: MessageRoleAssistant,
			Content: MessageContent{
				NewThinkingContentBlock("internal reasoning", ""),
				NewTextContentBlock("visible answer"),
			},
		},
	}, InferenceParams{Model: "test-model"})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}

	if len(params.Messages) != 1 {
		t.Fatalf("expected one assistant message, got %#v", params.Messages)
	}
	if got, ok := params.Messages[0].Content.(string); !ok || got != "visible answer" {
		t.Fatalf("unexpected assistant content: %#v", params.Messages[0].Content)
	}
}

func TestOpenAIFromOpenAIResponseMapsToolCallsAndUsage(t *testing.T) {
	client := &OpenAIClient{}
	message := client.fromOpenAIResponse(&openAIChatCompletionResponse{
		Choices: []openAIChatCompletionChoice{{
			FinishReason: "tool_calls",
			Message: openAIChatCompletionResponseMessage{
				Content: "hello",
				ToolCalls: []openAIChatCompletionToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: openAIChatCompletionFunctionCall{
						Name:      "lookup",
						Arguments: `{"city":"London"}`,
					},
				}},
			},
		}},
		Usage: &openAICompletionUsage{
			PromptTokens:     7,
			CompletionTokens: 3,
			TotalTokens:      10,
			PromptTokensDetails: &openAIPromptTokensDetails{
				CachedTokens: 2,
			},
		},
	}, nil)

	if message == nil {
		t.Fatal("expected message")
	}
	if message.StopReason != StopReasonToolUse {
		t.Fatalf("expected tool use stop reason, got %q", message.StopReason)
	}
	if message.Usage == nil || message.Usage.Totals.CacheReadInputTokens != 2 {
		t.Fatalf("unexpected usage: %#v", message.Usage)
	}
	if len(message.Content) != 2 || message.Content[1].ToolUse == nil {
		t.Fatalf("unexpected content: %#v", message.Content)
	}
	if got := string(message.Content[1].ToolUse.Input); got != `{"city":"London"}` {
		t.Fatalf("unexpected tool input: %q", got)
	}
}

func TestOpenAIFromOpenAIResponseMapsReasoningBlocks(t *testing.T) {
	client := &OpenAIClient{}
	message := client.fromOpenAIResponse(&openAIChatCompletionResponse{
		Choices: []openAIChatCompletionChoice{{
			FinishReason: "stop",
			Message: openAIChatCompletionResponseMessage{
				Reasoning: json.RawMessage(`{"text":"thinking..."}`),
				Content:   "answer",
			},
		}},
	}, nil)

	if message == nil {
		t.Fatal("expected message")
	}
	if len(message.Content) != 2 {
		t.Fatalf("expected thinking + text content, got %#v", message.Content)
	}
	if message.Content[0].Type != ContentBlockTypeThinking || message.Content[0].Thinking != "thinking..." {
		t.Fatalf("unexpected thinking block: %#v", message.Content[0])
	}
	if message.Content[1].Text != "answer" {
		t.Fatalf("unexpected text block: %#v", message.Content[1])
	}
}

func TestOpenAICreateEmbeddingsUsesSingleInput(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("unexpected authorization header: %q", got)
		}

		var request openAIEmbeddingRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if len(request.Input) != 1 {
			t.Fatalf("expected a single embedding input, got %#v", request.Input)
		}
		if request.Input[0] != "hello\nworld" {
			t.Fatalf("unexpected embedding input: %q", request.Input[0])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbeddingResponse{
			Model: "text-embedding-3-small",
			Data: []openAIEmbedding{{
				Embedding: []float64{0.25, 0.75},
			}},
			Usage: openAIEmbeddingUsage{
				PromptTokens: 5,
				TotalTokens:  5,
			},
		})
	}))
	defer server.Close()

	client := &OpenAIClient{
		client:  server.Client(),
		apiKey:  "test-key",
		baseURL: server.URL,
	}
	response, err := client.CreateEmbeddings(context.Background(), EmbeddingRequest{
		Model: "text-embedding-3-small",
		Inputs: MessageContent{
			NewTextContentBlock("hello"),
			NewDocumentContentBlockFromText("world", "text/plain", ""),
		},
	})
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %v", err)
	}

	if len(response.Embeddings) != 2 {
		t.Fatalf("expected embedding dimension 2, got %d", len(response.Embeddings))
	}
	if response.Embeddings[0] != 0.25 || response.Embeddings[1] != 0.75 {
		t.Fatalf("unexpected embedding values: %#v", response.Embeddings)
	}
	if response.Usage == nil || response.Usage.Totals.InputTokens != 5 || response.Usage.Totals.TotalTokens != 5 {
		t.Fatalf("unexpected usage: %#v", response.Usage)
	}
	if response.Usage.Totals.InputTokensDetails == nil || response.Usage.Totals.InputTokensDetails.TextTokens != 5 {
		t.Fatalf("unexpected multimodal details: %#v", response.Usage.Totals.InputTokensDetails)
	}
}

// TestOpenAIBuildCompletionParamsTextOnlyUserContentIsString verifies that a
// user message whose content contains only text blocks (e.g. the original user
// message plus a <plan>…</plan> block injected by the plan policy) is
// serialized as a plain string rather than a structured array. Some
// OpenAI-compatible providers (e.g. StepFun via OpenRouter) reject messages
// whose content is a single-element or multi-element text array.
func TestOpenAIBuildCompletionParamsTextOnlyUserContentIsString(t *testing.T) {
	client := &OpenAIClient{}

	// Single text block → should produce a string.
	params, err := client.buildCompletionParams([]Message{
		{Role: MessageRoleUser, Content: MessageContent{
			NewTextContentBlock("What day is it today"),
		}},
	}, InferenceParams{Model: "test-model"})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}
	if got, ok := params.Messages[0].Content.(string); !ok || got != "What day is it today" {
		t.Fatalf("expected plain string content for single text block, got %#v", params.Messages[0].Content)
	}

	// Multiple text blocks (e.g. user message + plan inject) → concatenated
	// string, not an array.
	params, err = client.buildCompletionParams([]Message{
		{Role: MessageRoleUser, Content: MessageContent{
			NewTextContentBlock("What day is it today"),
			NewTextContentBlock("<plan>\n1. ○ [HIGH] answer question\n</plan>"),
		}},
	}, InferenceParams{Model: "test-model"})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}
	want := "What day is it today<plan>\n1. ○ [HIGH] answer question\n</plan>"
	if got, ok := params.Messages[0].Content.(string); !ok || got != want {
		t.Fatalf("expected concatenated string for multi-text block, got %#v", params.Messages[0].Content)
	}
}

// TestOpenAIBuildCompletionParamsMultiTurnToolCall verifies that a multi-turn
// conversation (user → assistant with tool_call → tool result) is serialized
// correctly: assistant message carries tool_calls + no content, tool result
// becomes a "tool" role message with string content.
func TestOpenAIBuildCompletionParamsMultiTurnToolCall(t *testing.T) {
	client := &OpenAIClient{}

	params, err := client.buildCompletionParams([]Message{
		{Role: MessageRoleUser, Content: MessageContent{
			NewTextContentBlock("What day is it today"),
		}},
		{Role: MessageRoleAssistant, Content: MessageContent{
			NewThinkingContentBlock("I should search for this", ""),
			{Type: ContentBlockTypeToolUse, ToolUse: &ToolUse{
				ID:    "call_1",
				Name:  "grounded_search",
				Input: []byte(`{"query":"what day is it today"}`),
			}},
		}},
		{Role: MessageRoleUser, Content: MessageContent{
			NewToolResultContentBlock(ToolOutput{
				ToolUseID: "call_1",
				Output:    MessageContent{NewTextContentBlock("No grounded context found.")},
			}),
		}},
	}, InferenceParams{Model: "test-model"})
	if err != nil {
		t.Fatalf("buildCompletionParams returned error: %v", err)
	}

	// system prompt omitted → 3 messages total
	if len(params.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d: %#v", len(params.Messages), params.Messages)
	}

	// user message must be a plain string
	if got, ok := params.Messages[0].Content.(string); !ok || got != "What day is it today" {
		t.Fatalf("expected user message to be a string, got %#v", params.Messages[0].Content)
	}

	// assistant message must carry tool_calls and no content
	asst := params.Messages[1]
	if asst.Role != "assistant" {
		t.Fatalf("expected assistant role, got %q", asst.Role)
	}
	if len(asst.ToolCalls) != 1 || asst.ToolCalls[0].ID != "call_1" {
		t.Fatalf("expected one tool call, got %#v", asst.ToolCalls)
	}
	if asst.Content != nil {
		t.Fatalf("expected no content on assistant tool-call message, got %#v", asst.Content)
	}

	// tool result must become a "tool" role message with string content
	tool := params.Messages[2]
	if tool.Role != "tool" {
		t.Fatalf("expected tool role, got %q", tool.Role)
	}
	if tool.ToolCallID != "call_1" {
		t.Fatalf("expected tool_call_id call_1, got %q", tool.ToolCallID)
	}
	if got, ok := tool.Content.(string); !ok || got != "No grounded context found." {
		t.Fatalf("unexpected tool content: %#v", tool.Content)
	}
}

func TestOpenAIStreamAccumulatesToolCallsAndUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		var request openAIChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if !request.Stream {
			t.Fatalf("expected stream request, got %#v", request)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http flusher")
		}

		encode := func(v any) string {
			data, err := json.Marshal(v)
			if err != nil {
				t.Fatalf("failed to marshal stream chunk: %v", err)
			}
			return string(data)
		}

		chunks := []string{
			encode(openAIChatCompletionChunk{
				ID: "chatcmpl-1",
				Choices: []openAIChatCompletionChunkChoice{{
					Index: 0,
					Delta: openAIChatCompletionChunkDelta{
						ReasoningContent: json.RawMessage(`"Thinking: "`),
					},
				}},
			}),
			encode(openAIChatCompletionChunk{
				ID: "chatcmpl-1",
				Choices: []openAIChatCompletionChunkChoice{{
					Index: 0,
					Delta: openAIChatCompletionChunkDelta{
						ReasoningContent: json.RawMessage(`"step 1"`),
					},
				}},
			}),
			encode(openAIChatCompletionChunk{
				ID: "chatcmpl-1",
				Choices: []openAIChatCompletionChunkChoice{{
					Index: 0,
					Delta: openAIChatCompletionChunkDelta{
						Content: Ptr("Hel"),
					},
				}},
			}),
			encode(openAIChatCompletionChunk{
				ID: "chatcmpl-1",
				Choices: []openAIChatCompletionChunkChoice{{
					Index: 0,
					Delta: openAIChatCompletionChunkDelta{
						Content: Ptr("lo"),
					},
				}},
			}),
			encode(openAIChatCompletionChunk{
				ID: "chatcmpl-1",
				Choices: []openAIChatCompletionChunkChoice{{
					Index: 0,
					Delta: openAIChatCompletionChunkDelta{
						ToolCalls: []openAIChatCompletionChunkToolCall{{
							Index: 0,
							ID:    Ptr("call_1"),
							Type:  Ptr("function"),
							Function: &openAIChatCompletionFunctionCallDelta{
								Name:      Ptr("lookup"),
								Arguments: Ptr("{\"city\":\"Lon"),
							},
						}},
					},
				}},
			}),
			encode(openAIChatCompletionChunk{
				ID: "chatcmpl-1",
				Choices: []openAIChatCompletionChunkChoice{{
					Index:        0,
					FinishReason: Ptr("tool_calls"),
					Delta: openAIChatCompletionChunkDelta{
						ToolCalls: []openAIChatCompletionChunkToolCall{{
							Index: 0,
							Function: &openAIChatCompletionFunctionCallDelta{
								Arguments: Ptr("don\"}"),
							},
						}},
					},
				}},
			}),
			encode(openAIChatCompletionChunk{
				ID:      "chatcmpl-1",
				Choices: []openAIChatCompletionChunkChoice{},
				Usage: &openAICompletionUsage{
					PromptTokens:     7,
					CompletionTokens: 3,
					TotalTokens:      10,
					PromptTokensDetails: &openAIPromptTokensDetails{
						CachedTokens: 2,
					},
				},
			}),
			"[DONE]",
		}

		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := &OpenAIClient{
		client:  server.Client(),
		apiKey:  "test-key",
		baseURL: server.URL,
	}
	stream, err := client.CreateCompletionStream(context.Background(), []Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("hello")},
	}}, InferenceParams{
		Model: "test-model",
	})
	if err != nil {
		t.Fatalf("CreateCompletionStream returned error: %v", err)
	}
	defer stream.Close()

	var eventTypes []StreamEventType
	for {
		event, err := stream.Recv(context.Background())
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("Recv returned error: %v", err)
		}
		eventTypes = append(eventTypes, event.Type)
	}

	current := stream.Current()
	if current.StopReason != StopReasonToolUse {
		t.Fatalf("expected tool_use stop reason, got %q", current.StopReason)
	}
	if current.Usage == nil || current.Usage.Totals.InputTokens != 7 || current.Usage.Totals.OutputTokens != 3 || current.Usage.Totals.CacheReadInputTokens != 2 {
		t.Fatalf("unexpected usage: %#v", current.Usage)
	}
	if len(current.Content) != 3 {
		t.Fatalf("expected thinking, text and tool call content, got %#v", current.Content)
	}
	if current.Content[0].Type != ContentBlockTypeThinking || current.Content[0].Thinking != "Thinking: step 1" {
		t.Fatalf("unexpected accumulated thinking: %#v", current.Content[0])
	}
	if current.Content[1].Text != "Hello" {
		t.Fatalf("unexpected accumulated text: %#v", current.Content[1])
	}
	if current.Content[2].ToolUse == nil || current.Content[2].ToolUse.Name != "lookup" {
		t.Fatalf("unexpected tool call content: %#v", current.Content[2])
	}
	if got := string(current.Content[2].ToolUse.Input); got != `{"city":"London"}` {
		t.Fatalf("unexpected accumulated tool input: %q", got)
	}

	if len(eventTypes) == 0 || eventTypes[len(eventTypes)-1] != EventTypeMessageEnd {
		t.Fatalf("expected message end event, got %#v", eventTypes)
	}
}

// mapOpenAIUsage must surface the OpenRouter-style cache_write_tokens
// extension that the pi reference also reads (openai-completions.ts:1014).
// Real OpenAI never sets it; providers proxying through OpenRouter do,
// and missing it understates cache cost on those runs.
func TestMapOpenAIUsageReadsCacheWriteTokens(t *testing.T) {
	u := mapOpenAIUsage(&openAICompletionUsage{
		PromptTokens:     1000,
		CompletionTokens: 200,
		TotalTokens:      1200,
		PromptTokensDetails: &openAIPromptTokensDetails{
			CachedTokens:     700,
			CacheWriteTokens: 50,
		},
	}, nil, nil)
	if u == nil {
		t.Fatal("nil usage")
	}
	if u.Totals.CacheReadInputTokens != 700 {
		t.Errorf("CacheReadInputTokens=%d want 700", u.Totals.CacheReadInputTokens)
	}
	if u.Totals.CacheCreationInputTokens != 50 {
		t.Errorf("CacheCreationInputTokens=%d want 50 (from cache_write_tokens)", u.Totals.CacheCreationInputTokens)
	}
}

// completion_tokens_details.reasoning_tokens is REAL on OpenAI for
// o-series and gpt-5/codex models. Sub-count of completion_tokens; must
// NOT be added on top.
func TestMapOpenAIUsageReadsReasoningTokens(t *testing.T) {
	u := mapOpenAIUsage(&openAICompletionUsage{
		PromptTokens:     500,
		CompletionTokens: 3000,
		TotalTokens:      3500,
		CompletionTokensDetails: &openAICompletionTokensDetails{
			ReasoningTokens: 2500,
		},
	}, nil, nil)
	if u == nil {
		t.Fatal("nil usage")
	}
	if u.Totals.OutputTokens != 3000 {
		t.Errorf("OutputTokens=%d want 3000 (reasoning must NOT be added)", u.Totals.OutputTokens)
	}
	if u.Totals.ReasoningOutputTokens != 2500 {
		t.Errorf("ReasoningOutputTokens=%d want 2500", u.Totals.ReasoningOutputTokens)
	}
}

// gpt-4o-audio reports audio_tokens in both prompt and completion
// details — they land in the multimodal sub-detail slots rather than
// being collapsed into top-level counts.
func TestMapOpenAIUsageReadsAudioTokens(t *testing.T) {
	u := mapOpenAIUsage(&openAICompletionUsage{
		PromptTokens:     400,
		CompletionTokens: 100,
		TotalTokens:      500,
		PromptTokensDetails:     &openAIPromptTokensDetails{AudioTokens: 80},
		CompletionTokensDetails: &openAICompletionTokensDetails{AudioTokens: 25},
	}, nil, nil)
	if u == nil {
		t.Fatal("nil usage")
	}
	if u.Totals.InputTokensDetails == nil || u.Totals.InputTokensDetails.AudioTokens != 80 {
		t.Errorf("input AudioTokens missing: %+v", u.Totals.InputTokensDetails)
	}
	if u.Totals.OutputTokensDetails == nil || u.Totals.OutputTokensDetails.AudioTokens != 25 {
		t.Errorf("output AudioTokens missing: %+v", u.Totals.OutputTokensDetails)
	}
}

func TestResolveOpenAIServiceTier(t *testing.T) {
	str := func(s string) *string { return &s }

	tests := []struct {
		name     string
		resp     *string
		req      *string
		expected string
	}{
		{
			name:     "both nil",
			resp:     nil,
			req:      nil,
			expected: "",
		},
		{
			name:     "resp set, req nil",
			resp:     str("flex"),
			req:      nil,
			expected: "flex",
		},
		{
			name:     "resp nil, req set",
			resp:     nil,
			req:      str("priority"),
			expected: "priority",
		},
		{
			name:     "both set, resp wins",
			resp:     str("priority"),
			req:      str("flex"),
			expected: "priority",
		},
		{
			name:     "both set to default",
			resp:     str("default"),
			req:      str("default"),
			expected: "default",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := resolveOpenAIServiceTier(tc.resp, tc.req)
			if got != tc.expected {
				t.Errorf("resolveOpenAIServiceTier(%v, %v) = %q; want %q", tc.resp, tc.req, got, tc.expected)
			}
		})
	}
}

func TestMapOpenAIUsagePropagatesServiceTier(t *testing.T) {
	str := func(s string) *string { return &s }
	u := mapOpenAIUsage(&openAICompletionUsage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}, str("flex"), str("priority"))
	if u == nil {
		t.Fatal("nil usage")
	}
	if u.Totals.ServiceTier != "flex" {
		t.Errorf("expected ServiceTier flex, got %q", u.Totals.ServiceTier)
	}
}

