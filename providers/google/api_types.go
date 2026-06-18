package google

import "encoding/json"

type googleGenerateContentParameters struct {
	Contents          []googleContent         `json:"contents"`
	Tools             []googleTool            `json:"tools,omitempty"`
	ToolConfig        *googleToolConfig       `json:"toolConfig,omitempty"`
	SystemInstruction *googleContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *googleGenerationConfig `json:"generationConfig,omitempty"`
	Stream            bool                    `json:"-"` // Used internally to track streaming mode, not sent to API
}

type googleContent struct {
	Parts []googlePart `json:"parts,omitempty"`
	Role  string       `json:"role,omitempty"`
}

type googlePart struct {
	Thought          *bool                   `json:"thought,omitempty"`
	InlineData       *googleBlob             `json:"inlineData,omitempty"`
	FileData         *googleFileData         `json:"fileData,omitempty"`
	ThoughtSignature *string                 `json:"thoughtSignature,omitempty"`
	FunctionCall     *googleFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *googleFunctionResponse `json:"functionResponse,omitempty"`
	Text             *string                 `json:"text,omitempty"`
}

type googleBlob struct {
	DisplayName *string `json:"displayName,omitempty"`
	Data        *string `json:"data,omitempty"`
	MimeType    *string `json:"mimeType,omitempty"`
}

type googleFileData struct {
	DisplayName *string `json:"displayName,omitempty"`
	FileURI     *string `json:"fileUri,omitempty"`
	MimeType    *string `json:"mimeType,omitempty"`
}

type googleFunctionCall struct {
	ID           *string         `json:"id,omitempty"`
	Args         json.RawMessage `json:"args,omitempty"`
	Name         *string         `json:"name,omitempty"`
	WillContinue *bool           `json:"willContinue,omitempty"`
}

type googleFunctionResponse struct {
	ID           *string                      `json:"id,omitempty"`
	Name         *string                      `json:"name,omitempty"`
	Response     map[string]any               `json:"response,omitempty"`
	Parts        []googleFunctionResponsePart `json:"parts,omitempty"`
	WillContinue *bool                        `json:"willContinue,omitempty"`
}

type googleFunctionResponsePart struct {
	InlineData *googleBlob     `json:"inlineData,omitempty"`
	FileData   *googleFileData `json:"fileData,omitempty"`
}

type googleGenerationConfig struct {
	Temperature        *float64              `json:"temperature,omitempty"`
	TopP               *float64              `json:"topP,omitempty"`
	TopK               *int32                `json:"topK,omitempty"`
	CandidateCount     int                   `json:"candidateCount,omitempty"`
	MaxOutputTokens    *int                  `json:"maxOutputTokens,omitempty"`
	ResponseMimeType   *string               `json:"responseMimeType,omitempty"`
	ResponseJsonSchema any                   `json:"responseJsonSchema,omitempty"`
	ThinkingConfig     *googleThinkingConfig `json:"thinkingConfig,omitempty"`
}

type googleThinkingConfig struct {
	IncludeThoughts *bool  `json:"includeThoughts,omitempty"`
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
}

type googleTool struct {
	FunctionDeclarations []googleFunctionDeclaration `json:"functionDeclarations,omitempty"`
	GoogleSearch         *struct{}                   `json:"googleSearch,omitempty"`
	URLContext           *struct{}                   `json:"urlContext,omitempty"`
}

type googleFunctionDeclaration struct {
	Description          *string `json:"description,omitempty"`
	Name                 *string `json:"name,omitempty"`
	ParametersJsonSchema any     `json:"parametersJsonSchema,omitempty"`
	Parameters           any     `json:"parameters,omitempty"`
}

type googleToolConfig struct {
	FunctionCallingConfig *googleFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type googleFunctionCallingConfig struct {
	Mode                 *string  `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type googleGenerateContentResponse struct {
	Candidates    []googleCandidate    `json:"candidates,omitempty"`
	ModelVersion  *string              `json:"modelVersion,omitempty"`
	ResponseID    *string              `json:"responseId,omitempty"`
	UsageMetadata *googleUsageMetadata `json:"usageMetadata,omitempty"`
}

type googleCandidate struct {
	Content            *googleContent  `json:"content,omitempty"`
	FinishReason       *string         `json:"finishReason,omitempty"`
	FinishMessage      *string         `json:"finishMessage,omitempty"`
	TokenCount         *int            `json:"tokenCount,omitempty"`
	Index              *int            `json:"index,omitempty"`
	GroundingMetadata  json.RawMessage `json:"groundingMetadata,omitempty"`
	URLContextMetadata json.RawMessage `json:"urlContextMetadata,omitempty"`
}

type googleUsageMetadata struct {
	CacheTokensDetails         []googleModalityTokenCount `json:"cacheTokensDetails,omitempty"`
	CachedContentTokenCount    *int                       `json:"cachedContentTokenCount,omitempty"`
	CandidatesTokenCount       *int                       `json:"candidatesTokenCount,omitempty"`
	CandidatesTokensDetails    []googleModalityTokenCount `json:"candidatesTokensDetails,omitempty"`
	PromptTokenCount           *int                       `json:"promptTokenCount,omitempty"`
	PromptTokensDetails        []googleModalityTokenCount `json:"promptTokensDetails,omitempty"`
	ThoughtsTokenCount         *int                       `json:"thoughtsTokenCount,omitempty"`
	ToolUsePromptTokenCount    *int                       `json:"toolUsePromptTokenCount,omitempty"`
	ToolUsePromptTokensDetails []googleModalityTokenCount `json:"toolUsePromptTokensDetails,omitempty"`
	TotalTokenCount            *int                       `json:"totalTokenCount,omitempty"`
}

type googleModalityTokenCount struct {
	Modality   *string `json:"modality,omitempty"`
	TokenCount *int    `json:"tokenCount,omitempty"`
}

type googleEmbedContentRequest struct {
	Content              *googleContent `json:"content"`
	TaskType             string         `json:"taskType,omitempty"`
	Title                string         `json:"title,omitempty"`
	OutputDimensionality *int32         `json:"outputDimensionality,omitempty"`
}

type googleEmbedContentResponse struct {
	Embedding     *googleContentEmbedding     `json:"embedding,omitempty"`
	Metadata      *googleEmbedContentMetadata `json:"metadata,omitempty"`
	UsageMetadata *googleUsageMetadata        `json:"usageMetadata,omitempty"`
}

type googleEmbedContentMetadata struct {
	BillableCharacterCount int `json:"billableCharacterCount,omitempty"`
}

type googleContentEmbedding struct {
	Values     []float32                         `json:"values,omitempty"`
	Statistics *googleContentEmbeddingStatistics `json:"statistics,omitempty"`
}

type googleContentEmbeddingStatistics struct {
	TokenCount *int `json:"tokenCount,omitempty"`
}
