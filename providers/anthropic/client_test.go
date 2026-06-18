package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	. "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/tools"
	"golang.org/x/oauth2"
)

func TestAnthropicOAuthHeadersUseBearerAndClaudeIdentity(t *testing.T) {
	client, err := NewAnthropicClientWithTokenSource(context.Background(), oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "sk-ant-oat-test"}), "")
	if err != nil {
		t.Fatal(err)
	}
	// No tools in this request, so the fine-grained-tool-streaming
	// beta should not appear; existing OAuth betas should pass through.
	headers, err := client.requestHeaders(context.Background(), &anthropicMessageRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if headers["authorization"] != "Bearer sk-ant-oat-test" {
		t.Fatalf("authorization = %q", headers["authorization"])
	}
	if headers["x-api-key"] != "" {
		t.Fatalf("x-api-key should be absent, got %q", headers["x-api-key"])
	}
	if headers["x-app"] != "cli" || headers["user-agent"] == "" {
		t.Fatalf("missing Claude identity headers: %#v", headers)
	}
	if headers["anthropic-beta"] != "claude-code-20250219,oauth-2025-04-20" {
		t.Fatalf("anthropic-beta = %q", headers["anthropic-beta"])
	}
}

func TestAnthropicCreateCompletionUsesOutputConfigAndDocumentContext(t *testing.T) {
	schema := tools.ReflectValue(struct {
		Value string `json:"value" jsonschema:"required"`
	}{})
	tool := tools.NewGenericTool(
		"lookup_weather",
		"Look up the weather for a location",
		func(ctx ToolContext, input struct {
			Location string `json:"location" jsonschema:"required"`
		}) ToolResult {
			return ToolResult{}
		},
	).WithStrict(true)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != "test-key" {
			t.Fatalf("unexpected api key header: %q", got)
		}
		if got := r.Header.Get("anthropic-version"); got != defaultAnthropicVersion {
			t.Fatalf("unexpected anthropic version: %q", got)
		}

		var request anthropicMessageRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if request.OutputConfig == nil || request.OutputConfig.Format.Type != "json_schema" {
			t.Fatalf("expected output_config format, got %#v", request.OutputConfig)
		}
		if len(request.Messages) != 1 || len(request.Messages[0].Content) != 1 {
			t.Fatalf("unexpected messages payload: %#v", request.Messages)
		}
		doc := request.Messages[0].Content[0]
		// Non-native endpoints convert documents to text blocks.
		if doc.Type != "text" || doc.Text != "Doc Title\nDoc Context\nDoc Body" {
			t.Fatalf("expected document to be converted to text for non-native endpoint, got %#v", doc)
		}
		if len(request.Tools) != 1 {
			t.Fatalf("expected one tool, got %#v", request.Tools)
		}
		toolDef, ok := request.Tools[0].(map[string]any)
		if !ok {
			t.Fatalf("expected tool definition map, got %#v", request.Tools[0])
		}
		if strict, ok := toolDef["strict"].(bool); !ok || !strict {
			t.Fatalf("expected strict tool definition, got %#v", toolDef)
		}
		inputSchema, ok := toolDef["input_schema"].(map[string]any)
		if !ok {
			t.Fatalf("expected input schema map, got %#v", toolDef["input_schema"])
		}
		if additionalProperties, ok := inputSchema["additionalProperties"].(bool); !ok || additionalProperties {
			t.Fatalf("expected additionalProperties=false, got %#v", inputSchema)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(anthropicMessage{
			Role:       "assistant",
			StopReason: "end_turn",
			Content: []anthropicContentBlock{{
				Type: "text",
				Text: "ok",
			}},
			Usage: anthropicUsage{
				InputTokens:  10,
				OutputTokens: 5,
			},
		})
	}))
	defer server.Close()

	client := &AnthropicClient{
		client:     server.Client(),
		apiKey:     "test-key",
		baseURL:    server.URL,
		apiVersion: defaultAnthropicVersion,
	}
	msg, err := client.CreateCompletion(context.Background(), []Message{{
		Role: MessageRoleUser,
		Content: MessageContent{{
			Type: ContentBlockTypeDocument,
			Document: &DocumentSource{
				Type:    DocumentSourceTypeText,
				Title:   "Doc Title",
				Context: "Doc Context",
				Text:    "Doc Body",
			},
		}},
	}}, InferenceParams{
		Model: "claude-test",
		Tools: []Tool{tool},
		ResponseFormat: &ResponseFormat{
			Type:       ResponseFormatTypeJSONObject,
			JSONSchema: schema,
		},
	})
	if err != nil {
		t.Fatalf("CreateCompletion returned error: %v", err)
	}
	if msg == nil || msg.Content.Text() != "ok" {
		t.Fatalf("unexpected completion message: %#v", msg)
	}
}

func TestAnthropicBuildCompletionRequestRequiresSchemaForStructuredOutputs(t *testing.T) {
	client := &AnthropicClient{}
	_, err := client.buildCompletionRequest([]Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("hello")},
	}}, InferenceParams{
		Model: "claude-test",
		ResponseFormat: &ResponseFormat{
			Type: ResponseFormatTypeJSONObject,
		},
	}, false)
	if err == nil {
		t.Fatal("expected structured output request without schema to fail")
	}
}

func TestAnthropicStreamAccumulatesToolUseJSONDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http flusher")
		}

		events := []anthropicStreamEvent{
			{
				Type: "message_start",
				Message: &anthropicMessage{
					Usage: anthropicUsage{InputTokens: 11},
				},
			},
			{
				Type:  "content_block_start",
				Index: 0,
				ContentBlock: &anthropicContentBlock{
					Type:  "tool_use",
					ID:    "tool-1",
					Name:  "lookup",
					Input: json.RawMessage(`{}`),
				},
			},
			{
				Type:  "content_block_delta",
				Index: 0,
				Delta: json.RawMessage(`{"type":"input_json_delta","partial_json":"{\"city\":\"London\"}"}`),
			},
			{
				Type:  "content_block_stop",
				Index: 0,
			},
			{
				Type:  "message_delta",
				Delta: json.RawMessage(`{"stop_reason":"tool_use"}`),
				Usage: &anthropicMessageDeltaUsage{OutputTokens: 3},
			},
			{Type: "message_stop"},
		}

		for _, event := range events {
			payload, err := json.Marshal(event)
			if err != nil {
				t.Fatalf("failed to marshal sse event: %v", err)
			}
			fmt.Fprintf(w, "event: %s\n", event.Type)
			fmt.Fprintf(w, "data: %s\n\n", payload)
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := &AnthropicClient{
		client:     server.Client(),
		apiKey:     "test-key",
		baseURL:    server.URL,
		apiVersion: defaultAnthropicVersion,
	}
	stream, err := client.CreateCompletionStream(context.Background(), []Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("hello")},
	}}, InferenceParams{
		Model: "claude-test",
	})
	if err != nil {
		t.Fatalf("CreateCompletionStream returned error: %v", err)
	}
	defer stream.Close()

	for {
		if _, err := stream.Recv(context.Background()); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("Recv returned error: %v", err)
		}
	}

	current := stream.Current()
	if current.StopReason != StopReasonToolUse {
		t.Fatalf("expected tool_use stop reason, got %q", current.StopReason)
	}
	if current.Usage == nil || current.Usage.Totals.InputTokens != 11 || current.Usage.Totals.OutputTokens != 3 {
		t.Fatalf("unexpected usage: %#v", current.Usage)
	}
	if len(current.Content) != 1 || current.Content[0].ToolUse == nil {
		t.Fatalf("unexpected stream content: %#v", current.Content)
	}
	if got := string(current.Content[0].ToolUse.Input); got != "{\"city\":\"London\"}" {
		t.Fatalf("unexpected accumulated tool input: %q", got)
	}
}

func TestAnthropicBuildCompletionRequestIncludesNamedToolChoice(t *testing.T) {
	client := &AnthropicClient{}
	tool := tools.NewGenericTool(
		"submit_structured_output",
		"Submit structured output",
		func(ctx ToolContext, input struct {
			Value string `json:"value"`
		}) ToolResult {
			return ToolResult{}
		},
	).WithStrict(true)

	request, err := client.buildCompletionRequest([]Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("hello")},
	}}, InferenceParams{
		Model: "claude-test",
		Tools: []Tool{tool},
		ToolChoice: &ToolChoice{
			Type: ToolChoiceTypeTool,
			Name: "submit_structured_output",
		},
	}, false)
	if err != nil {
		t.Fatalf("buildCompletionRequest returned error: %v", err)
	}
	if request.ToolChoice == nil || request.ToolChoice.Type != "tool" || request.ToolChoice.Name != "submit_structured_output" {
		t.Fatalf("expected named tool choice, got %#v", request.ToolChoice)
	}
	if request.ToolChoice.DisableParallelToolUse == nil || !*request.ToolChoice.DisableParallelToolUse {
		t.Fatalf("expected parallel tool use to be disabled, got %#v", request.ToolChoice)
	}
}

func TestAnthropicStreamMergesSeededToolUseJSONDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http flusher")
		}

		events := []anthropicStreamEvent{
			{
				Type: "message_start",
				Message: &anthropicMessage{
					Usage: anthropicUsage{InputTokens: 9},
				},
			},
			{
				Type:  "content_block_start",
				Index: 0,
				ContentBlock: &anthropicContentBlock{
					Type:  "tool_use",
					ID:    "tool-2",
					Name:  "fs_write",
					Input: json.RawMessage(`{"path":"README.md"}`),
				},
			},
			{
				Type:  "content_block_delta",
				Index: 0,
				Delta: json.RawMessage(`{"type":"input_json_delta","partial_json":",\"content\":\"updated\"}"}`),
			},
			{
				Type:  "content_block_stop",
				Index: 0,
			},
			{
				Type:  "message_delta",
				Delta: json.RawMessage(`{"stop_reason":"tool_use"}`),
				Usage: &anthropicMessageDeltaUsage{OutputTokens: 4},
			},
			{Type: "message_stop"},
		}

		for _, event := range events {
			payload, err := json.Marshal(event)
			if err != nil {
				t.Fatalf("failed to marshal sse event: %v", err)
			}
			fmt.Fprintf(w, "event: %s\n", event.Type)
			fmt.Fprintf(w, "data: %s\n\n", payload)
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := &AnthropicClient{
		client:     server.Client(),
		apiKey:     "test-key",
		baseURL:    server.URL,
		apiVersion: defaultAnthropicVersion,
	}
	stream, err := client.CreateCompletionStream(context.Background(), []Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("hello")},
	}}, InferenceParams{
		Model: "claude-test",
	})
	if err != nil {
		t.Fatalf("CreateCompletionStream returned error: %v", err)
	}
	defer stream.Close()

	for {
		if _, err := stream.Recv(context.Background()); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("Recv returned error: %v", err)
		}
	}

	current := stream.Current()
	if len(current.Content) != 1 || current.Content[0].ToolUse == nil {
		t.Fatalf("unexpected stream content: %#v", current.Content)
	}
	if got := string(current.Content[0].ToolUse.Input); got != `{"path":"README.md","content":"updated"}` {
		t.Fatalf("unexpected accumulated tool input: %q", got)
	}
}

func TestAnthropicFromAnthropicMessagePreservesServerToolBlocks(t *testing.T) {
	client := &AnthropicClient{}
	message := client.fromAnthropicMessage(&anthropicMessage{
		Role:       "assistant",
		StopReason: "end_turn",
		Content: []anthropicContentBlock{
			{
				Type:  "server_tool_use",
				ID:    "srvtoolu_1",
				Name:  "web_search",
				Input: json.RawMessage(`{"query":"weather in London"}`),
			},
			{
				Type:      "web_fetch_tool_result",
				ToolUseID: "srvtoolu_1",
				Content: json.RawMessage(`{
					"type":"web_fetch_result",
					"url":"https://example.com/article",
					"content":{
						"type":"document",
						"title":"Article Title",
						"source":{"type":"text","media_type":"text/plain","data":"Article body"}
					}
				}`),
			},
		},
		Usage: anthropicUsage{
			InputTokens:  10,
			OutputTokens: 5,
			ServerToolUse: map[string]int{
				"web_fetch_requests": 1,
			},
		},
	}, nil)

	if message == nil {
		t.Fatal("expected message")
	}
	if len(message.Content) != 2 {
		t.Fatalf("expected server tool use + result blocks, got %#v", message.Content)
	}
	if message.Content[0].ToolUse == nil || message.Content[0].ToolUse.Execution != ToolExecutionModeServer {
		t.Fatalf("expected first block to be a server tool use, got %#v", message.Content[0])
	}
	if message.Content[1].ToolOutput == nil || message.Content[1].ToolOutput.Execution != ToolExecutionModeServer {
		t.Fatalf("expected second block to be a server tool result, got %#v", message.Content[1])
	}
	if message.Content[1].ToolOutput.ProviderType != "web_fetch_tool_result" {
		t.Fatalf("unexpected server tool result type: %#v", message.Content[1].ToolOutput)
	}
	if message.Content[1].ToolOutput.Output.Text() != "Article Title\nArticle body" {
		t.Fatalf("expected fetched document text to be preserved, got %#v", message.Content[1].ToolOutput.Output)
	}
	if message.Usage == nil || message.Usage.Totals.ServerToolUse["web_fetch_requests"] != 1 {
		t.Fatalf("expected server tool usage counts to be preserved, got %#v", message.Usage)
	}
}
