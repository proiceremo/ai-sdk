package google

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	. "github.com/proiceremo/ai-sdk"

	"cloud.google.com/go/auth"
)

func TestGoogleEmbeddingContentsSupportMultimodalBlocks(t *testing.T) {
	client := &GoogleClient{}
	contents, err := client.toGoogleEmbeddingContents(
		MessageContent{
			NewTextContentBlock("describe this"),
			NewImageContentBlockFromURL("gs://bucket/cat.png", "image/png"),
		},
	)
	if err != nil {
		t.Fatalf("toGoogleEmbeddingContents returned error: %v", err)
	}
	if len(contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(contents))
	}
	if len(contents[0].Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(contents[0].Parts))
	}
	if contents[0].Parts[0].Text == nil || *contents[0].Parts[0].Text != "describe this" {
		t.Fatalf("unexpected text part: %#v", contents[0].Parts[0])
	}
	if contents[0].Parts[1].FileData == nil || contents[0].Parts[1].FileData.FileURI == nil || *contents[0].Parts[1].FileData.FileURI != "gs://bucket/cat.png" {
		t.Fatalf("unexpected image part: %#v", contents[0].Parts[1])
	}
}

func TestGoogleEmbeddingContentsRejectToolingBlocks(t *testing.T) {
	client := &GoogleClient{}
	_, err := client.toGoogleEmbeddingContents(MessageContent{
		NewToolUseContentBlock("tool-1", "lookup", nil, nil),
	})
	if err == nil {
		t.Fatal("expected unsupported tooling block to be rejected")
	}
}

func TestGoogleFromGoogleMessageHandlesEmptyCandidates(t *testing.T) {
	client := &GoogleClient{}
	if msg := client.fromGoogleMessage(&googleGenerateContentResponse{}); msg != nil {
		t.Fatalf("expected nil message for empty candidates, got %#v", msg)
	}
	if msg := client.fromGoogleMessage(nil); msg != nil {
		t.Fatalf("expected nil message for nil response, got %#v", msg)
	}
}

func TestGoogleFromGoogleMessageMapsModelRoleToAssistant(t *testing.T) {
	client := &GoogleClient{}
	response := &googleGenerateContentResponse{
		Candidates: []googleCandidate{{
			Content: &googleContent{
				Role: "model",
				Parts: []googlePart{{
					Text: stringPtr("hello"),
				}},
			},
		}},
	}

	msg := client.fromGoogleMessage(response)
	if msg == nil {
		t.Fatal("expected message")
	}
	if msg.Role != MessageRoleAssistant {
		t.Fatalf("expected assistant role, got %q", msg.Role)
	}
}

func TestGoogleFromGoogleMessagePreservesGroundingMetadata(t *testing.T) {
	client := &GoogleClient{}
	response := &googleGenerateContentResponse{
		Candidates: []googleCandidate{{
			Content: &googleContent{
				Role: "model",
				Parts: []googlePart{{
					Text: stringPtr("hello"),
				}},
			},
			GroundingMetadata:  json.RawMessage(`{"groundingChunks":[{"web":{"uri":"https://example.com"}}]}`),
			URLContextMetadata: json.RawMessage(`{"urlMetadata":[{"retrievedUrl":"https://example.com"}]}`),
		}},
	}

	msg := client.fromGoogleMessage(response)
	if msg == nil {
		t.Fatal("expected message")
	}
	if msg.Metadata == nil {
		t.Fatalf("expected grounding metadata to be preserved, got %#v", msg)
	}
	if _, ok := msg.Metadata["grounding_metadata"]; !ok {
		t.Fatalf("expected grounding_metadata entry, got %#v", msg.Metadata)
	}
	if _, ok := msg.Metadata["url_context_metadata"]; !ok {
		t.Fatalf("expected url_context_metadata entry, got %#v", msg.Metadata)
	}
}

func TestGoogleToolResultConversionPreservesDocumentTextAndMedia(t *testing.T) {
	client := &GoogleClient{}
	parts, err := client.toGoogleParts(MessageContent{
		NewToolResultContentBlock(ToolOutput{
			ToolUseID: "tool-1",
			Name:      "lookup",
			Output: MessageContent{
				NewTextContentBlock("plain"),
				NewDocumentContentBlockFromText("document body", "text/plain", ""),
				NewImageContentBlockFromURL("gs://bucket/cat.png", "image/png"),
			},
		}),
	})
	if err != nil {
		t.Fatalf("toGoogleParts returned error: %v", err)
	}
	if len(parts) != 1 || parts[0].FunctionResponse == nil {
		t.Fatalf("unexpected tool result parts: %#v", parts)
	}

	output, ok := parts[0].FunctionResponse.Response["output"].(map[string]any)
	if !ok {
		t.Fatalf("expected structured output map, got %#v", parts[0].FunctionResponse.Response)
	}
	if got := output["part-1"]; got != "document body" {
		t.Fatalf("expected document text to be preserved, got %#v", got)
	}
	if len(parts[0].FunctionResponse.Parts) != 1 {
		t.Fatalf("expected one media response part, got %#v", parts[0].FunctionResponse.Parts)
	}
	if parts[0].FunctionResponse.Parts[0].FileData == nil || parts[0].FunctionResponse.Parts[0].FileData.FileURI == nil || *parts[0].FunctionResponse.Parts[0].FileData.FileURI != "gs://bucket/cat.png" {
		t.Fatalf("unexpected media response part: %#v", parts[0].FunctionResponse.Parts[0])
	}
}

func TestGoogleFromGoogleMessageMapsFunctionCallStopToToolUse(t *testing.T) {
	client := &GoogleClient{}
	response := &googleGenerateContentResponse{
		Candidates: []googleCandidate{{
			Content: &googleContent{
				Role: "model",
				Parts: []googlePart{{
					FunctionCall: &googleFunctionCall{
						ID:   stringPtr("call-1"),
						Name: stringPtr("lookup"),
						Args: json.RawMessage(`{"q":"status"}`),
					},
				}},
			},
			FinishReason: stringPtr("STOP"),
		}},
	}

	msg := client.fromGoogleMessage(response)
	if msg == nil {
		t.Fatal("expected message")
	}
	if msg.StopReason != StopReasonToolUse {
		t.Fatalf("expected stop reason tool_use, got %q", msg.StopReason)
	}
	if len(msg.Content.ToolCalls()) != 1 {
		t.Fatalf("expected 1 tool call, got %#v", msg.Content)
	}
}

func TestGoogleCreateEmbeddingsUsesSingleContentRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1beta/models/test-model:embedContent" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Fatalf("unexpected api key header: %q", got)
		}

		var request googleEmbedContentRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if request.Content == nil || len(request.Content.Parts) != 2 {
			t.Fatalf("unexpected embedding content: %#v", request.Content)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(googleEmbedContentResponse{
			Embedding: &googleContentEmbedding{Values: []float32{3, 4}},
		})
	}))
	defer server.Close()

	client := newGoogleClient("test-key", server.URL, defaultGoogleAPIVersion, "", "", false, server.Client())
	resp, err := client.CreateEmbeddings(context.Background(), EmbeddingRequest{
		Model: "test-model",
		Inputs: MessageContent{
			NewTextContentBlock("hello"),
			NewDocumentContentBlockFromText("world", "text/plain", ""),
		},
	})
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %v", err)
	}
	if len(resp.Embeddings) != 2 {
		t.Fatalf("expected embedding dimension 2, got %d", len(resp.Embeddings))
	}
	if resp.Embeddings[0] != 0.6 || resp.Embeddings[1] != 0.8 {
		t.Fatalf("unexpected normalized embedding: %#v", resp.Embeddings)
	}
}

func TestGoogleCreateEmbeddingsWithUsageMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(googleEmbedContentResponse{
			Embedding: &googleContentEmbedding{Values: []float32{1, 0}},
			UsageMetadata: &googleUsageMetadata{
				PromptTokenCount: intPtr(10),
				PromptTokensDetails: []googleModalityTokenCount{
					{Modality: stringPtr("TEXT"), TokenCount: intPtr(4)},
					{Modality: stringPtr("IMAGE"), TokenCount: intPtr(6)},
				},
			},
		})
	}))
	defer server.Close()

	client := newGoogleClient("test-key", server.URL, defaultGoogleAPIVersion, "", "", false, server.Client())
	resp, err := client.CreateEmbeddings(context.Background(), EmbeddingRequest{
		Model: "test-model",
		Inputs: MessageContent{
			NewTextContentBlock("hello"),
		},
	})
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %v", err)
	}

	if resp.Usage == nil {
		t.Fatal("expected usage")
	}
	if resp.Usage.Totals.InputTokens != 10 {
		t.Fatalf("expected 10 prompt tokens, got %d", resp.Usage.Totals.InputTokens)
	}
	if resp.Usage.Totals.InputTokensDetails == nil || resp.Usage.Totals.InputTokensDetails.TextTokens != 4 || resp.Usage.Totals.InputTokensDetails.ImageTokens != 6 {
		t.Fatalf("unexpected multimodal details: %#v", resp.Usage.Totals.InputTokensDetails)
	}
	if len(resp.Usage.Entries) != 1 || resp.Usage.Entries[0].Operation != UsageOperationEmbedding {
		t.Fatalf("expected embedding usage operation, got %#v", resp.Usage.Entries)
	}
}

func TestVertexCreateEmbeddingsUsesModelOnlyInURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		expectedPath := "/v1beta1/projects/test-project/locations/global/publishers/google/models/gemini-embedding-2-preview:embedContent"
		if r.URL.Path != expectedPath {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		var raw map[string]any
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if _, exists := raw["model"]; exists {
			t.Fatalf("expected embedContent request body to omit model, got %#v", raw)
		}

		content, ok := raw["content"].(map[string]any)
		if !ok {
			t.Fatalf("expected content object, got %#v", raw["content"])
		}
		parts, ok := content["parts"].([]any)
		if !ok || len(parts) != 1 {
			t.Fatalf("expected one content part, got %#v", content["parts"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(googleEmbedContentResponse{
			Embedding: &googleContentEmbedding{Values: []float32{1, 2, 2}},
		})
	}))
	defer server.Close()

	client := newGoogleClient("", server.URL, defaultVertexAPIVersion, "test-project", "global", true, server.Client())
	resp, err := client.CreateEmbeddings(context.Background(), EmbeddingRequest{
		Model: "gemini-embedding-2-preview",
		Inputs: MessageContent{
			NewTextContentBlock("hello"),
		},
	})
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %v", err)
	}
	if len(resp.Embeddings) != 3 {
		t.Fatalf("expected embedding dimension 3, got %d", len(resp.Embeddings))
	}
	if !strings.HasPrefix(resp.Model, "gemini-embedding-2-preview") && resp.Model != "gemini-embedding-2-preview" {
		t.Fatalf("unexpected model echo: %q", resp.Model)
	}
}

func TestMapGoogleUsageMetadataIncludesModalityDetails(t *testing.T) {
	usage := mapGoogleUsageMetadata(googleUsageMetadata{
		PromptTokenCount:        intPtr(11),
		CandidatesTokenCount:    intPtr(7),
		CachedContentTokenCount: intPtr(3),
		ToolUsePromptTokenCount: intPtr(5),
		PromptTokensDetails: []googleModalityTokenCount{
			{Modality: stringPtr("TEXT"), TokenCount: intPtr(6)},
			{Modality: stringPtr("IMAGE"), TokenCount: intPtr(5)},
		},
		CandidatesTokensDetails: []googleModalityTokenCount{
			{Modality: stringPtr("AUDIO"), TokenCount: intPtr(2)},
			{Modality: stringPtr("TEXT"), TokenCount: intPtr(5)},
		},
		CacheTokensDetails: []googleModalityTokenCount{
			{Modality: stringPtr("IMAGE"), TokenCount: intPtr(3)},
		},
		ToolUsePromptTokensDetails: []googleModalityTokenCount{
			{Modality: stringPtr("DOCUMENT"), TokenCount: intPtr(4)},
			{Modality: stringPtr("TEXT"), TokenCount: intPtr(1)},
		},
	})

	if usage == nil {
		t.Fatal("expected usage")
	}
	if usage.Totals.InputTokens != 11 || usage.Totals.OutputTokens != 7 || usage.Totals.CacheReadInputTokens != 3 || usage.Totals.ToolUseInputTokens != 5 {
		t.Fatalf("unexpected usage totals: %#v", usage)
	}
	if usage.Totals.TotalTokens != 23 {
		t.Fatalf("expected total tokens to include tool-use prompts, got %#v", usage)
	}
	if usage.Totals.InputTokensDetails == nil || usage.Totals.InputTokensDetails.TextTokens != 6 || usage.Totals.InputTokensDetails.ImageTokens != 5 {
		t.Fatalf("unexpected prompt details: %#v", usage.Totals.InputTokensDetails)
	}
	if usage.Totals.OutputTokensDetails == nil || usage.Totals.OutputTokensDetails.AudioTokens != 2 || usage.Totals.OutputTokensDetails.TextTokens != 5 {
		t.Fatalf("unexpected completion details: %#v", usage.Totals.OutputTokensDetails)
	}
	if usage.Totals.CacheReadInputTokensDetails == nil || usage.Totals.CacheReadInputTokensDetails.ImageTokens != 3 {
		t.Fatalf("unexpected cache read details: %#v", usage.Totals.CacheReadInputTokensDetails)
	}
	if usage.Totals.ToolUseInputTokensDetails == nil || usage.Totals.ToolUseInputTokensDetails.DocumentTokens != 4 || usage.Totals.ToolUseInputTokensDetails.TextTokens != 1 {
		t.Fatalf("unexpected tool-use details: %#v", usage.Totals.ToolUseInputTokensDetails)
	}
}

func TestNewVertexGoogleClientWithOptionsUsesAuthMiddleware(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer vertex-token" {
			t.Fatalf("unexpected authorization header: %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(googleGenerateContentResponse{
			Candidates: []googleCandidate{{
				Content: &googleContent{
					Role: "model",
					Parts: []googlePart{{
						Text: stringPtr("ok"),
					}},
				},
			}},
		})
	}))
	defer server.Close()

	creds := auth.NewCredentials(&auth.CredentialsOptions{
		TokenProvider: staticTokenProvider{token: "vertex-token"},
		ProjectIDProvider: auth.CredentialsPropertyFunc(func(ctx context.Context) (string, error) {
			return "project-from-creds", nil
		}),
	})

	client, err := NewVertexGoogleClientWithOptions(context.Background(), "", "us-central1", VertexGoogleClientOptions{
		Credentials: creds,
		HTTPClient:  server.Client(),
		BaseURL:     server.URL,
		APIVersion:  defaultVertexAPIVersion,
	})
	if err != nil {
		t.Fatalf("NewVertexGoogleClientWithOptions returned error: %v", err)
	}
	if client.project != "project-from-creds" {
		t.Fatalf("expected project to be inferred from credentials, got %q", client.project)
	}

	msg, err := client.CreateCompletion(context.Background(), []Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("hello")},
	}}, InferenceParams{
		Model: "gemini-test",
	})
	if err != nil {
		t.Fatalf("CreateCompletion returned error: %v", err)
	}
	if msg == nil || msg.Content.Text() != "ok" {
		t.Fatalf("unexpected completion response: %#v", msg)
	}
}

func intPtr(value int) *int {
	return &value
}

func TestGoogleBuildCompletionRequestIncludesNamedToolChoice(t *testing.T) {
	client := &GoogleClient{}
	tool := &mockTool{name: "submit_structured_output"}

	request, err := client.buildCompletionRequest([]Message{{
		Role:    MessageRoleUser,
		Content: MessageContent{NewTextContentBlock("hello")},
	}}, InferenceParams{
		Model: "gemini-test",
		Tools: []Tool{tool},
		ToolChoice: &ToolChoice{
			Type: ToolChoiceTypeTool,
			Name: "submit_structured_output",
		},
	})
	if err != nil {
		t.Fatalf("buildCompletionRequest returned error: %v", err)
	}
	if request.ToolConfig == nil || request.ToolConfig.FunctionCallingConfig == nil {
		t.Fatalf("expected tool config, got %#v", request.ToolConfig)
	}
	if request.ToolConfig.FunctionCallingConfig.Mode == nil || *request.ToolConfig.FunctionCallingConfig.Mode != "ANY" {
		t.Fatalf("expected ANY function calling mode, got %#v", request.ToolConfig.FunctionCallingConfig)
	}
	if len(request.ToolConfig.FunctionCallingConfig.AllowedFunctionNames) != 1 || request.ToolConfig.FunctionCallingConfig.AllowedFunctionNames[0] != "submit_structured_output" {
		t.Fatalf("expected named allowed function, got %#v", request.ToolConfig.FunctionCallingConfig.AllowedFunctionNames)
	}
}
