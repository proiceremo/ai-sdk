package google

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
	"github.com/proiceremo/ai-sdk/contentio"
	"github.com/proiceremo/ai-sdk/providers/internal/httpx"

	"cloud.google.com/go/auth"
	"cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/auth/httptransport"
)

const (
	defaultGoogleBaseURL     = "https://generativelanguage.googleapis.com"
	defaultGoogleAPIVersion  = "v1beta"
	defaultVertexBaseURL     = "https://aiplatform.googleapis.com"
	defaultVertexAPIVersion  = "v1beta1"
	googleCloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
)

type GoogleClient struct {
	client     *http.Client
	apiKey     string
	baseURL    string
	apiVersion string
	project    string
	location   string
	vertex     bool
}

func (c *GoogleClient) SupportsStreaming() bool { return true }

type VertexGoogleClientOptions struct {
	AccessToken      string
	Credentials      *auth.Credentials
	TokenProvider    auth.TokenProvider
	CredentialsFile  string
	CredentialsJSON  []byte
	CredentialType   credentials.CredType
	Scopes           []string
	UseSelfSignedJWT bool
	UniverseDomain   string
	HTTPClient       *http.Client
	BaseURL          string
	APIVersion       string
}

func NewGoogleClient(ctx context.Context, apiKey string) (*GoogleClient, error) {
	_ = ctx
	return newGoogleClient(apiKey, defaultGoogleBaseURL, defaultGoogleAPIVersion, "", "", false, nil), nil
}

func NewVertexGoogleClient(ctx context.Context, apiKey, project, location string) (*GoogleClient, error) {
	return NewVertexGoogleClientWithOptions(ctx, project, location, VertexGoogleClientOptions{
		AccessToken: apiKey,
	})
}

func NewVertexGoogleClientWithOptions(ctx context.Context, project, location string, options VertexGoogleClientOptions) (*GoogleClient, error) {
	if location == "" {
		location = "global"
	}
	baseURL := options.BaseURL
	if baseURL == "" {
		baseURL = defaultVertexBaseURL
		if location != "global" {
			baseURL = fmt.Sprintf("https://%s-aiplatform.googleapis.com", location)
		}
	}

	apiVersion := options.APIVersion
	if apiVersion == "" {
		apiVersion = defaultVertexAPIVersion
	}

	client := options.HTTPClient
	if client == nil {
		client = &http.Client{}
	}

	creds, err := resolveVertexCredentials(ctx, options, client)
	if err != nil {
		return nil, err
	}
	if creds != nil {
		if err := httptransport.AddAuthorizationMiddleware(client, creds); err != nil {
			return nil, err
		}
		if project == "" {
			if inferredProject, err := creds.ProjectID(ctx); err == nil && inferredProject != "" {
				project = inferredProject
			}
		}
	}

	return newGoogleClient("", baseURL, apiVersion, project, location, true, client), nil
}

func newGoogleClient(apiKey, baseURL, apiVersion, project, location string, vertex bool, client *http.Client) *GoogleClient {
	if client == nil {
		client = defaultHTTPClient()
	}
	return &GoogleClient{
		client:     client,
		apiKey:     apiKey,
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiVersion: apiVersion,
		project:    project,
		location:   location,
		vertex:     vertex,
	}
}

func defaultHTTPClient() *http.Client {
	return &http.Client{
		Timeout: 10 * time.Minute, // Timeout to prevent indefinite network hangs; streaming is supported within this window
	}
}

func resolveVertexCredentials(ctx context.Context, options VertexGoogleClientOptions, client *http.Client) (*auth.Credentials, error) {
	_ = ctx

	authSources := 0
	if options.Credentials != nil {
		authSources++
	}
	if options.TokenProvider != nil {
		authSources++
	}
	if strings.TrimSpace(options.AccessToken) != "" {
		authSources++
	}
	if strings.TrimSpace(options.CredentialsFile) != "" {
		authSources++
	}
	if len(options.CredentialsJSON) > 0 {
		authSources++
	}
	if authSources > 1 {
		return nil, fmt.Errorf("vertex authentication options are mutually exclusive")
	}

	if options.Credentials != nil {
		return options.Credentials, nil
	}
	if options.TokenProvider != nil {
		return auth.NewCredentials(&auth.CredentialsOptions{
			TokenProvider: auth.NewCachedTokenProvider(options.TokenProvider, nil),
		}), nil
	}
	if strings.TrimSpace(options.AccessToken) != "" {
		return auth.NewCredentials(&auth.CredentialsOptions{
			TokenProvider: auth.NewCachedTokenProvider(staticTokenProvider{
				token: options.AccessToken,
			}, nil),
		}), nil
	}

	detectClient := &http.Client{
		Timeout: 2 * time.Second, // Prevent hanging on metadata server
	}
	if client != nil && client.Transport != nil {
		detectClient.Transport = client.Transport
	}

	detectOptions := &credentials.DetectOptions{
		Scopes:           vertexScopes(options.Scopes),
		CredentialsFile:  options.CredentialsFile,
		CredentialsJSON:  options.CredentialsJSON,
		UseSelfSignedJWT: options.UseSelfSignedJWT,
		Client:           detectClient,
		UniverseDomain:   options.UniverseDomain,
	}

	return credentials.DetectDefault(detectOptions)
}

func vertexScopes(scopes []string) []string {
	if len(scopes) == 0 {
		return []string{googleCloudPlatformScope}
	}
	return append([]string(nil), scopes...)
}

type staticTokenProvider struct {
	token string
}

func (p staticTokenProvider) Token(ctx context.Context) (*auth.Token, error) {
	_ = ctx
	return &auth.Token{
		Value: strings.TrimSpace(strings.TrimPrefix(p.token, "Bearer ")),
	}, nil
}

func normalizeEmbeddingIfNeeded(values []float32) []float32 {
	if IsNormalized(values) {
		return values
	}
	return NormalizeVector(values)
}

func (c *GoogleClient) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	contents, err := c.toGoogleEmbeddingContents(req.Inputs)
	if err != nil {
		return nil, err
	}
	if len(contents) != 1 {
		return nil, fmt.Errorf("embedding input must resolve to exactly one Google content, got %d", len(contents))
	}

	request := googleEmbedContentRequest{
		Content:              &contents[0],
		TaskType:             req.TaskType,
		OutputDimensionality: req.Dimensions,
	}
	if req.Metadata != nil && req.Metadata["title"] != "" && req.TaskType == "RETRIEVAL_DOCUMENT" {
		request.Title = req.Metadata["title"]
	}

	var response googleEmbedContentResponse
	if err := httpx.DoJSON(ctx, c.client, http.MethodPost, c.endpointURL(req.Model, ":embedContent"), c.requestHeaders(), request, &response); err != nil {
		return nil, fmt.Errorf("failed to create embeddings: %w", err)
	}
	if response.Embedding == nil {
		return nil, fmt.Errorf("embedding response is missing embedding data")
	}

	var usage *Usage
	if response.UsageMetadata != nil {
		usage = mapGoogleUsageMetadataForOperation(*response.UsageMetadata, UsageOperationEmbedding)
	} else if stats := response.Embedding.Statistics; stats != nil && stats.TokenCount != nil {
		usage = NewUsage(UsageOperationEmbedding, TokenUsage{InputTokens: *stats.TokenCount, TotalTokens: *stats.TokenCount})
	}

	return &EmbeddingResponse{
		Model:      req.Model,
		Embeddings: normalizeEmbeddingIfNeeded(response.Embedding.Values),
		Usage:      usage,
	}, nil
}

func (c *GoogleClient) toGoogleEmbeddingContents(inputs MessageContent) ([]googleContent, error) {
	parts, err := c.toGoogleEmbeddingParts(inputs)
	if err != nil {
		return nil, err
	}
	return []googleContent{{
		Role:  "user",
		Parts: parts,
	}}, nil
}

func (c *GoogleClient) toGoogleEmbeddingParts(contentBlocks []ContentBlock) ([]googlePart, error) {
	parts := make([]googlePart, 0, len(contentBlocks))
	for _, content := range contentBlocks {
		switch content.Type {
		case ContentBlockTypeText:
			text := content.Text
			parts = append(parts, googlePart{Text: &text})
		case ContentBlockTypeImage:
			part, err := googlePartFromImage(content.Image)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		case ContentBlockTypeVideo:
			part, err := googlePartFromVideo(content.Video)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		case ContentBlockTypeAudio:
			part, err := googlePartFromAudio(content.Audio)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		case ContentBlockTypeDocument:
			part, err := googlePartFromDocument(content.Document)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		default:
			return nil, fmt.Errorf("content block type %q is not supported for embeddings", content.Type)
		}
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("embedding input must contain at least one supported content block")
	}
	return parts, nil
}

func (c *GoogleClient) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	if err := ValidateMessages(messages); err != nil {
		return nil, err
	}
	request, err := c.buildCompletionRequest(messages, params)
	if err != nil {
		return nil, err
	}

	var response googleGenerateContentResponse
	if err := httpx.DoJSON(ctx, c.client, http.MethodPost, c.endpointURL(params.Model, ":generateContent"), c.requestHeaders(), request, &response); err != nil {
		return nil, err
	}

	msg := c.fromGoogleMessage(&response)
	if msg == nil {
		return nil, fmt.Errorf("google completion returned no candidates")
	}
	return msg, nil
}

func (c *GoogleClient) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	if err := ValidateMessages(messages); err != nil {
		return nil, err
	}
	request, err := c.buildCompletionRequest(messages, params)
	if err != nil {
		return nil, err
	}

	// Debug: Log the request being sent
	if os.Getenv("BENCH_DEBUG") != "" {
		if requestJSON, err := json.MarshalIndent(request, "", "  "); err == nil {
			fmt.Fprintf(os.Stderr, "[GoogleClient DEBUG] Request to %s:\n%s\n", c.endpointURL(params.Model, ":streamGenerateContent"), string(requestJSON))
		}
	}

	endpointURL := c.endpointURL(params.Model, ":streamGenerateContent?alt=sse")
	if os.Getenv("BENCH_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[GoogleClient DEBUG] Endpoint URL: %s\n", endpointURL)
	}

	stream, err := httpx.DoSSE(ctx, c.client, http.MethodPost, endpointURL, c.requestHeaders(), request)
	if err != nil {
		if os.Getenv("BENCH_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[GoogleClient ERROR] DoSSE failed: %v\n", err)
		}
		return nil, err
	}

	if os.Getenv("BENCH_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[GoogleClient DEBUG] Stream created successfully\n")
	}

	return &googleStream{
		BaseStream: NewBaseStream(),
		stream:     stream,
	}, nil
}

func (c *GoogleClient) buildCompletionRequest(messages []Message, params InferenceParams) (*googleGenerateContentParameters, error) {
	googleMessages, err := c.toGoogleMessages(messages)
	if err != nil {
		return nil, err
	}

	request := &googleGenerateContentParameters{
		Contents: googleMessages,
		GenerationConfig: &googleGenerationConfig{
			Temperature:     params.Temperature,
			TopP:            params.TopP,
			MaxOutputTokens: params.MaxTokens,
		},
	}
	if params.TopK != nil {
		topK := int32(*params.TopK)
		request.GenerationConfig.TopK = &topK
	}
	if params.SystemPrompt != "" {
		systemText := params.SystemPrompt
		request.SystemInstruction = &googleContent{
			Parts: []googlePart{{Text: &systemText}},
		}
	}
	if params.WebSearch != nil && params.WebSearch.Enabled {
		request.Tools = []googleTool{
			{GoogleSearch: &struct{}{}},
			{URLContext: &struct{}{}},
		}
	} else if len(params.Tools) > 0 {
		functions := make([]googleFunctionDeclaration, 0, len(params.Tools))
		for _, tool := range params.Tools {
			schema := tool.Schema()
			functions = append(functions, googleFunctionDeclaration{
				Name:                 &schema.Name,
				Description:          &schema.Description,
				ParametersJsonSchema: schema.InputSchema,
			})
		}
		request.Tools = append(request.Tools, googleTool{FunctionDeclarations: functions})
		if params.ToolChoice != nil {
			toolConfig, err := toGoogleToolConfig(*params.ToolChoice)
			if err != nil {
				return nil, err
			}
			request.ToolConfig = toolConfig
		}
	}
	if params.Thinking != nil && params.Thinking.Enabled {
		request.GenerationConfig.ThinkingConfig = convertToGoogleThinkingConfig(*params.Thinking)
	}
	if params.ResponseFormat != nil {
		switch params.ResponseFormat.Type {
		case ResponseFormatTypeJSONObject:
			mimeType := "application/json"
			request.GenerationConfig.ResponseMimeType = &mimeType
			if params.ResponseFormat.JSONSchema != nil {
				request.GenerationConfig.ResponseJsonSchema = params.ResponseFormat.JSONSchema
			}
		case ResponseFormatTypeText:
			mimeType := "text/plain"
			request.GenerationConfig.ResponseMimeType = &mimeType
		}
	}
	return request, nil
}

func toGoogleToolConfig(choice ToolChoice) (*googleToolConfig, error) {
	switch choice.Type {
	case "", ToolChoiceTypeAuto:
		mode := "AUTO"
		return &googleToolConfig{
			FunctionCallingConfig: &googleFunctionCallingConfig{Mode: &mode},
		}, nil
	case ToolChoiceTypeRequired:
		mode := "ANY"
		return &googleToolConfig{
			FunctionCallingConfig: &googleFunctionCallingConfig{Mode: &mode},
		}, nil
	case ToolChoiceTypeTool:
		if strings.TrimSpace(choice.Name) == "" {
			return nil, fmt.Errorf("google named tool choice requires a tool name")
		}
		mode := "ANY"
		return &googleToolConfig{
			FunctionCallingConfig: &googleFunctionCallingConfig{
				Mode:                 &mode,
				AllowedFunctionNames: []string{choice.Name},
			},
		}, nil
	default:
		return nil, fmt.Errorf("unsupported google tool choice type %q", choice.Type)
	}
}

func googlePartFromImage(source *ImageSource) (*googlePart, error) {
	return googleMediaPart(string(source.Type), source.URL, source.Data, source.MediaType)
}

func googlePartFromVideo(source *VideoSource) (*googlePart, error) {
	return googleMediaPart(string(source.Type), source.URL, source.Data, source.MediaType)
}

func googlePartFromAudio(source *AudioSource) (*googlePart, error) {
	return googleMediaPart(string(source.Type), source.URL, source.Data, source.MediaType)
}

func googlePartFromDocument(source *DocumentSource) (*googlePart, error) {
	if source == nil {
		return nil, fmt.Errorf("document content is missing payload")
	}
	switch source.Type {
	case DocumentSourceTypeText:
		text := strings.TrimSpace(googleDocumentText(source))
		if text == "" {
			text = source.Text
		}
		return &googlePart{Text: &text}, nil
	case DocumentSourceTypeURL:
		return &googlePart{FileData: &googleFileData{
			FileURI:  &source.URL,
			MimeType: stringPtr(source.MediaType),
		}}, nil
	case DocumentSourceTypeBase64:
		blob, err := googleBlobFromBase64(source.Data, source.MediaType)
		if err != nil {
			return nil, err
		}
		return &googlePart{InlineData: blob}, nil
	default:
		return nil, fmt.Errorf("unsupported document source type %q", source.Type)
	}
}

func (c *GoogleClient) toGoogleMessages(messages []Message) ([]googleContent, error) {
	googleMessages := make([]googleContent, 0, len(messages))
	for _, message := range messages {
		parts, err := c.toGoogleParts(message.Content)
		if err != nil {
			return nil, err
		}
		if len(parts) == 0 {
			continue
		}
		role := string(message.Role)
		if message.Role == MessageRoleAssistant {
			role = "model"
		}
		googleMessages = append(googleMessages, googleContent{
			Role:  role,
			Parts: parts,
		})
	}
	return googleMessages, nil
}

func (c *GoogleClient) toGoogleParts(contentBlocks []ContentBlock) ([]googlePart, error) {
	parts := make([]googlePart, 0, len(contentBlocks))
	for _, content := range contentBlocks {
		switch content.Type {
		case ContentBlockTypeThinking:
			part := googlePart{
				Thought: ptrBool(true),
				Text:    stringPtr(content.Thinking),
			}
			if content.Signature != "" {
				part.ThoughtSignature = &content.Signature
			}
			parts = append(parts, part)
		case ContentBlockTypeText:
			parts = append(parts, googlePart{Text: &content.Text})
		case ContentBlockTypeToolResult:
			part, err := googleFunctionResponseFromToolOutput(content.ToolOutput)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		case ContentBlockTypeToolUse:
			part, err := googleFunctionCallPart(content.ToolUse)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		case ContentBlockTypeImage:
			part, err := googlePartFromImage(content.Image)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		case ContentBlockTypeVideo:
			part, err := googlePartFromVideo(content.Video)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		case ContentBlockTypeAudio:
			part, err := googlePartFromAudio(content.Audio)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		case ContentBlockTypeDocument:
			part, err := googlePartFromDocument(content.Document)
			if err != nil {
				return nil, err
			}
			parts = append(parts, *part)
		default:
			return nil, fmt.Errorf("unsupported content block type %q", content.Type)
		}
	}
	return parts, nil
}

func googleFunctionResponseFromToolOutput(output *ToolOutput) (*googlePart, error) {
	if output == nil {
		return nil, fmt.Errorf("tool result content is missing payload")
	}

	responseMap := map[string]any{}
	responseParts := make([]googleFunctionResponsePart, 0, len(output.Output))

	for index, part := range output.Output {
		key := fmt.Sprintf("part-%d", index)
		switch part.Type {
		case ContentBlockTypeText:
			responseMap[key] = part.Text
		case ContentBlockTypeThinking:
			responseMap[key] = part.Thinking
		case ContentBlockTypeDocument:
			if docText := strings.TrimSpace(googleDocumentText(part.Document)); docText != "" {
				responseMap[key] = docText
			}
			functionPart, err := googleFunctionResponsePartFromContentBlock(part)
			if err != nil {
				return nil, err
			}
			if functionPart != nil {
				responseParts = append(responseParts, *functionPart)
			}
		default:
			functionPart, err := googleFunctionResponsePartFromContentBlock(part)
			if err != nil {
				return nil, err
			}
			if functionPart != nil {
				responseParts = append(responseParts, *functionPart)
				continue
			}
			responseMap[key] = googleContentBlockFallbackText(part)
		}
	}

	key := "output"
	if output.IsError {
		key = "error"
	}

	// Get the function name - try output.Name first, then extract from tool use ID
	funcName := output.Name
	if funcName == "" && output.ToolUseID != "" {
		// Try to extract function name from context if available
		// The ToolUseID should correspond to a previous function call
		funcName = "unknown_function"
	}

	functionResponse := &googleFunctionResponse{
		ID:       stringPtr(output.ToolUseID),
		Name:     stringPtr(funcName),
		Response: map[string]any{key: normalizeGoogleToolResponsePayload(responseMap)},
	}
	if len(responseParts) > 0 {
		functionResponse.Parts = responseParts
	}

	return &googlePart{FunctionResponse: functionResponse}, nil
}

func googleFunctionCallPart(toolUse *ToolUse) (*googlePart, error) {
	if toolUse == nil {
		return nil, fmt.Errorf("tool use content is missing payload")
	}
	call := &googleFunctionCall{
		ID:   stringPtr(toolUse.ID),
		Name: stringPtr(toolUse.Name),
	}
	if len(toolUse.Input) > 0 {
		call.Args = toolUse.Input
	} else {
		call.Args = json.RawMessage(`{}`)
	}
	part := &googlePart{FunctionCall: call}
	if toolUse.Signature != "" {
		part.ThoughtSignature = stringPtr(toolUse.Signature)
	}
	return part, nil
}

func (c *GoogleClient) fromGoogleMessage(response *googleGenerateContentResponse) *Message {
	if response == nil || len(response.Candidates) == 0 {
		return nil
	}
	firstCandidate := response.Candidates[0]
	if firstCandidate.Content == nil || len(firstCandidate.Content.Parts) == 0 {
		return nil
	}

	content := make([]ContentBlock, 0, len(firstCandidate.Content.Parts))
	for _, block := range firstCandidate.Content.Parts {
		part, err := c.googlePartToContentBlock(block)
		if err != nil {
			return nil
		}
		if part != nil {
			content = append(content, *part)
		}
	}

	var usage *Usage
	if response.UsageMetadata != nil {
		usage = mapGoogleUsageMetadata(*response.UsageMetadata)
	}

	metadata := googleCandidateMetadata(firstCandidate)
	stopReason := googleStopReason(content, stringValue(firstCandidate.FinishReason))

	return &Message{
		Role:       googleRoleToMessageRole(firstCandidate.Content.Role),
		StopReason: stopReason,
		Content:    content,
		Timestamp:  time.Now(),
		Usage:      usage,
		Metadata:   metadata,
	}
}

func googleCandidateMetadata(candidate googleCandidate) map[string]any {
	metadata := map[string]any{}
	if len(candidate.GroundingMetadata) > 0 {
		if decoded, err := rawJSONMetadata(candidate.GroundingMetadata); err == nil {
			metadata["grounding_metadata"] = decoded
		}
	}
	if len(candidate.URLContextMetadata) > 0 {
		if decoded, err := rawJSONMetadata(candidate.URLContextMetadata); err == nil {
			metadata["url_context_metadata"] = decoded
		}
	}
	if len(metadata) == 0 {
		return nil
	}
	return metadata
}

func rawJSONMetadata(value json.RawMessage) (any, error) {
	var decoded any
	if err := json.Unmarshal(value, &decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

type googleStream struct {
	*BaseStream
	stream                *httpx.SSEStream
	err                   error
	lastBlockWillContinue bool
}

func (s *googleStream) Recv(ctx context.Context) (*StreamEvent, error) {
	_ = ctx
	if event := s.PopEvent(); event != nil {
		return event, nil
	}
	if s.IsFinished() {
		if os.Getenv("BENCH_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[GoogleStream DEBUG] Stream is finished, returning EOF\n")
		}
		return nil, io.EOF
	}

	eventCount := 0
	for s.stream.Next() {
		eventCount++
		streamEvent := s.stream.Current()

		if len(streamEvent.Data) == 0 {
			if os.Getenv("BENCH_DEBUG") != "" {
				fmt.Fprintf(os.Stderr, "[GoogleStream DEBUG] Empty event data, skipping (event #%d)\n", eventCount)
			}
			continue
		}

		if os.Getenv("BENCH_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[GoogleStream DEBUG] Received event #%d, data length: %d\n", eventCount, len(streamEvent.Data))
		}

		var response googleGenerateContentResponse
		if err := json.Unmarshal(streamEvent.Data, &response); err != nil {
			if os.Getenv("BENCH_DEBUG") != "" {
				fmt.Fprintf(os.Stderr, "[GoogleStream ERROR] Failed to unmarshal event: %v, data: %s\n", err, string(streamEvent.Data))
			}
			s.err = err
			s.Finish("", nil)
			return nil, err
		}

		if err := s.processResponse(&response); err != nil {
			s.err = err
			s.Finish("", nil)
			return nil, err
		}

		if event := s.PopEvent(); event != nil {
			return event, nil
		}
	}

	streamErr := s.stream.Err()
	if streamErr != nil {
		if os.Getenv("BENCH_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[GoogleStream ERROR] Stream error after %d events: %v\n", eventCount, streamErr)
		}
		s.err = streamErr
		s.Finish("", nil)
		return nil, streamErr
	}

	if os.Getenv("BENCH_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[GoogleStream DEBUG] Stream ended normally after %d events\n", eventCount)
	}
	accumulated := s.Accumulated()
	if os.Getenv("BENCH_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[GoogleStream DEBUG] Accumulated stop reason: %s, usage: %+v\n", accumulated.StopReason, accumulated.Usage)
	}
	s.Finish(accumulated.StopReason, accumulated.Usage)
	if event := s.PopEvent(); event != nil {
		return event, nil
	}
	return nil, io.EOF
}

func (s *googleStream) processResponse(response *googleGenerateContentResponse) error {
	s.EmitMessageStart()

	if len(response.Candidates) > 0 {
		candidate := response.Candidates[0]
		candidateBlocks := make([]ContentBlock, 0)
		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				newBlock, err := s.googlePartToContentBlock(part)
				if err != nil {
					return err
				}
				if newBlock == nil {
					continue
				}

				willContinue := false
				if part.FunctionCall != nil && part.FunctionCall.WillContinue != nil {
					willContinue = *part.FunctionCall.WillContinue
				}

				if s.shouldAppendToLast(newBlock, s.CurrentBlockType()) {
					s.AppendDelta(*newBlock)
				} else {
					empty := ContentBlock{Type: newBlock.Type}
					if newBlock.Type == ContentBlockTypeToolUse && newBlock.ToolUse != nil {
						empty.ToolUse = &ToolUse{
							ID:   newBlock.ToolUse.ID,
							Name: newBlock.ToolUse.Name,
						}
					}
					s.OpenBlock(empty)
					s.AppendDelta(*newBlock)
				}
				s.lastBlockWillContinue = willContinue
				candidateBlocks = append(candidateBlocks, *newBlock)
			}
		}

		if candidate.FinishReason != nil {
			stopReason := googleStopReason(candidateBlocks, *candidate.FinishReason)
			s.SetStopReason(stopReason)
			if s.IsBlockOpen() {
				s.CloseCurrentBlock(stopReason, nil)
			}
		}
	}

	if response.UsageMetadata != nil {
		s.SetUsage(mapGoogleUsageMetadata(*response.UsageMetadata))
	}
	return nil
}

func (c *GoogleClient) googlePartToContentBlock(part googlePart) (*ContentBlock, error) {
	return googlePartToContentBlock(part)
}

func (s *googleStream) googlePartToContentBlock(part googlePart) (*ContentBlock, error) {
	return googlePartToContentBlock(part)
}

func googlePartToContentBlock(part googlePart) (*ContentBlock, error) {
	if part.Thought != nil && *part.Thought {
		signature := ""
		if part.ThoughtSignature != nil {
			signature = *part.ThoughtSignature
		}
		block := NewThinkingContentBlock(stringValue(part.Text), signature)
		return &block, nil
	}

	if part.Text != nil {
		block := NewTextContentBlock(*part.Text)
		return &block, nil
	}

	if part.FunctionCall != nil {
		name := stringValue(part.FunctionCall.Name)
		id := stringValue(part.FunctionCall.ID)
		args := part.FunctionCall.Args
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		block := NewToolUseContentBlock(id, name, args, nil)
		if part.ThoughtSignature != nil {
			block.ToolUse.Signature = *part.ThoughtSignature
		}
		return &block, nil
	}

	if part.InlineData != nil && part.InlineData.Data != nil {
		block := contentio.ContentBlockFromMimeTypeBase64(stringValue(part.InlineData.MimeType), *part.InlineData.Data)
		return &block, nil
	}

	if part.FileData != nil && part.FileData.FileURI != nil {
		block := contentio.ContentBlockFromMimeTypeURL(stringValue(part.FileData.MimeType), *part.FileData.FileURI)
		return &block, nil
	}

	return nil, nil
}

func googleStopReason(content []ContentBlock, finishReason string) StopReason {
	stopReason := NormaliseStopReason(APIFormatGoogle, finishReason)
	if len(content) == 0 || stopReason != StopReasonEndTurn {
		return stopReason
	}
	for _, block := range content {
		if block.Type == ContentBlockTypeToolUse && block.ToolUse != nil {
			return StopReasonToolUse
		}
	}
	return stopReason
}

func (s *googleStream) shouldAppendToLast(newBlock *ContentBlock, currentType ContentBlockType) bool {
	if currentType == "" {
		return false
	}
	if currentType == ContentBlockTypeText && newBlock.Type == ContentBlockTypeText {
		return true
	}
	if currentType == ContentBlockTypeThinking && newBlock.Type == ContentBlockTypeThinking {
		return true
	}
	if currentType == ContentBlockTypeToolUse && newBlock.Type == ContentBlockTypeToolUse {
		return s.lastBlockWillContinue
	}
	return false
}

func (s *googleStream) Current() *Message {
	accumulated := s.Accumulated()
	return &accumulated
}

func (s *googleStream) Close() error {
	s.Finish("", nil)
	if s.stream == nil {
		return nil
	}
	return s.stream.Close()
}

func googleMediaPart(sourceType, url, data, mediaType string) (*googlePart, error) {
	switch sourceType {
	case "url":
		return &googlePart{FileData: &googleFileData{
			FileURI:  &url,
			MimeType: stringPtr(mediaType),
		}}, nil
	case "base64":
		blob, err := googleBlobFromBase64(data, mediaType)
		if err != nil {
			return nil, err
		}
		return &googlePart{InlineData: blob}, nil
	default:
		return nil, fmt.Errorf("unsupported media source type %q", sourceType)
	}
}

func googleBlobFromBase64(data, mediaType string) (*googleBlob, error) {
	if data == "" {
		return nil, fmt.Errorf("inline data is required")
	}
	if i := strings.Index(data, ","); i != -1 {
		data = data[i+1:]
	}
	if mediaType == "" {
		_, detected, err := DecodeBase64File(data)
		if err != nil {
			return nil, err
		}
		mediaType = detected
	}
	return &googleBlob{
		Data:     &data,
		MimeType: &mediaType,
	}, nil
}

func googleFunctionResponsePartFromContentBlock(block ContentBlock) (*googleFunctionResponsePart, error) {
	switch block.Type {
	case ContentBlockTypeImage:
		if block.Image == nil {
			return nil, nil
		}
		return googleFunctionResponsePartFromMedia(string(block.Image.Type), block.Image.URL, block.Image.Data, block.Image.MediaType)
	case ContentBlockTypeVideo:
		if block.Video == nil {
			return nil, nil
		}
		return googleFunctionResponsePartFromMedia(string(block.Video.Type), block.Video.URL, block.Video.Data, block.Video.MediaType)
	case ContentBlockTypeAudio:
		if block.Audio == nil {
			return nil, nil
		}
		return googleFunctionResponsePartFromMedia(string(block.Audio.Type), block.Audio.URL, block.Audio.Data, block.Audio.MediaType)
	case ContentBlockTypeDocument:
		if block.Document == nil {
			return nil, nil
		}
		switch block.Document.Type {
		case DocumentSourceTypeURL:
			return &googleFunctionResponsePart{FileData: &googleFileData{
				FileURI:  &block.Document.URL,
				MimeType: stringPtr(block.Document.MediaType),
			}}, nil
		case DocumentSourceTypeBase64:
			blob, err := googleBlobFromBase64(block.Document.Data, block.Document.MediaType)
			if err != nil {
				return nil, err
			}
			return &googleFunctionResponsePart{InlineData: blob}, nil
		case DocumentSourceTypeText:
			return nil, nil
		}
	}
	return nil, nil
}

func googleFunctionResponsePartFromMedia(sourceType, url, data, mediaType string) (*googleFunctionResponsePart, error) {
	switch sourceType {
	case "url":
		return &googleFunctionResponsePart{FileData: &googleFileData{
			FileURI:  &url,
			MimeType: stringPtr(mediaType),
		}}, nil
	case "base64":
		blob, err := googleBlobFromBase64(data, mediaType)
		if err != nil {
			return nil, err
		}
		return &googleFunctionResponsePart{InlineData: blob}, nil
	default:
		return nil, nil
	}
}

func googleDocumentText(doc *DocumentSource) string {
	if doc == nil {
		return ""
	}
	parts := make([]string, 0, 3)
	for _, part := range []string{doc.Title, doc.Context, doc.Text} {
		part = strings.TrimSpace(part)
		if part != "" {
			parts = append(parts, part)
		}
	}
	return strings.Join(parts, "\n")
}

func googleContentBlockFallbackText(block ContentBlock) string {
	switch block.Type {
	case ContentBlockTypeImage:
		if block.Image != nil {
			if block.Image.Type == ImageSourceTypeURL {
				return "[IMAGE] " + block.Image.URL
			}
			return "[IMAGE] <base64 data>"
		}
	case ContentBlockTypeVideo:
		if block.Video != nil {
			if block.Video.Type == VideoSourceTypeURL {
				return "[VIDEO] " + block.Video.URL
			}
			return "[VIDEO] <base64 data>"
		}
	case ContentBlockTypeAudio:
		if block.Audio != nil {
			if block.Audio.Type == AudioSourceTypeURL {
				return "[AUDIO] " + block.Audio.URL
			}
			return "[AUDIO] <base64 data>"
		}
	case ContentBlockTypeDocument:
		if block.Document != nil {
			if text := googleDocumentText(block.Document); text != "" {
				return text
			}
			if block.Document.Type == DocumentSourceTypeURL {
				return "[DOCUMENT] " + block.Document.URL
			}
			return "[DOCUMENT] <base64 data>"
		}
	}
	return "[CONTENT] " + string(block.Type)
}

func normalizeGoogleToolResponsePayload(responseMap map[string]any) any {
	if len(responseMap) == 0 {
		return map[string]any{}
	}
	if len(responseMap) == 1 {
		for _, value := range responseMap {
			return value
		}
	}
	return responseMap
}

func convertToGoogleThinkingConfig(thinking ThinkingParams) *googleThinkingConfig {
	config := &googleThinkingConfig{
		IncludeThoughts: ptrBool(thinking.Enabled),
	}
	switch thinking.Level {
	case ThinkingLevelMinimal:
		config.ThinkingLevel = "MINIMAL"
	case ThinkingLevelLow:
		config.ThinkingLevel = "LOW"
	case ThinkingLevelMedium:
		config.ThinkingLevel = "MEDIUM"
	case ThinkingLevelHigh, ThinkingLevelXHigh:
		config.ThinkingLevel = "HIGH"
	default:
		config.ThinkingLevel = "MINIMAL"
	}
	return config
}

func mapGoogleUsageMetadata(usageMetadata googleUsageMetadata) *Usage {
	return mapGoogleUsageMetadataForOperation(usageMetadata, UsageOperationCompletion)
}

func mapGoogleUsageMetadataForOperation(usageMetadata googleUsageMetadata, operation UsageOperation) *Usage {
	tokens := TokenUsage{}
	if usageMetadata.PromptTokenCount != nil {
		tokens.InputTokens = *usageMetadata.PromptTokenCount
	}
	if usageMetadata.CandidatesTokenCount != nil {
		tokens.OutputTokens = *usageMetadata.CandidatesTokenCount
	}
	if usageMetadata.TotalTokenCount != nil {
		tokens.TotalTokens = *usageMetadata.TotalTokenCount
	} else {
		tokens.TotalTokens = tokens.InputTokens + tokens.OutputTokens
		if usageMetadata.ToolUsePromptTokenCount != nil {
			tokens.TotalTokens += *usageMetadata.ToolUsePromptTokenCount
		}
	}
	if usageMetadata.CachedContentTokenCount != nil {
		tokens.CacheReadInputTokens = *usageMetadata.CachedContentTokenCount
	}
	tokens.InputTokensDetails = mapGoogleTokenDetails(usageMetadata.PromptTokensDetails)
	tokens.OutputTokensDetails = mapGoogleTokenDetails(usageMetadata.CandidatesTokensDetails)
	tokens.CacheReadInputTokensDetails = mapGoogleTokenDetails(usageMetadata.CacheTokensDetails)
	if usageMetadata.ToolUsePromptTokenCount != nil {
		tokens.ToolUseInputTokens = *usageMetadata.ToolUsePromptTokenCount
	}
	tokens.ToolUseInputTokensDetails = mapGoogleTokenDetails(usageMetadata.ToolUsePromptTokensDetails)
	tokens.CacheBilledSeparately = true
	return NewUsage(operation, tokens)
}

func mapGoogleTokenDetails(modalities []googleModalityTokenCount) *UsageTokenDetails {
	if len(modalities) == 0 {
		return nil
	}

	details := &UsageTokenDetails{}
	for _, modality := range modalities {
		if modality.Modality == nil || modality.TokenCount == nil {
			continue
		}
		switch strings.ToUpper(*modality.Modality) {
		case "TEXT":
			details.TextTokens += *modality.TokenCount
		case "AUDIO":
			details.AudioTokens += *modality.TokenCount
		case "IMAGE":
			details.ImageTokens += *modality.TokenCount
		case "VIDEO":
			details.VideoTokens += *modality.TokenCount
		case "DOCUMENT":
			details.DocumentTokens += *modality.TokenCount
		}
	}

	if details.Empty() {
		return nil
	}
	return details
}

func googleRoleToMessageRole(role string) MessageRole {
	if role == "model" || role == "" {
		return MessageRoleAssistant
	}
	return MessageRole(role)
}

func (c *GoogleClient) requestHeaders() map[string]string {
	headers := map[string]string{}
	if c.vertex {
		if c.apiKey != "" {
			headers["Authorization"] = "Bearer " + c.apiKey
		}
		return headers
	}
	headers["x-goog-api-key"] = c.apiKey
	return headers
}

func (c *GoogleClient) endpointURL(model, action string) string {
	return fmt.Sprintf("%s/%s/%s%s", c.baseURL, c.apiVersion, c.normalizeModel(model), action)
}

func (c *GoogleClient) normalizeModel(model string) string {
	if model == "" {
		return model
	}
	if c.vertex {
		switch {
		case strings.HasPrefix(model, "projects/"), strings.HasPrefix(model, "publishers/"), strings.HasPrefix(model, "models/"):
			return model
		default:
			return fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", c.project, c.location, model)
		}
	}
	if strings.HasPrefix(model, "models/") || strings.HasPrefix(model, "tunedModels/") {
		return model
	}
	return "models/" + model
}

func ptrBool(value bool) *bool {
	return &value
}

func stringPtr(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}
