package llm

import (
	"context"
	"errors"
	"math"
)

var ErrEmbeddingsNotSupported = errors.New("provider does not support embeddings")

type EmbeddingRequest struct {
	Model      string            `json:"model"`
	Inputs     MessageContent    `json:"inputs"`
	TaskType   string            `json:"task_type,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Dimensions *int32            `json:"dimensions,omitempty"`
}

type EmbeddingResponse struct {
	Model      string    `json:"model,omitempty"`
	Embeddings []float32 `json:"embeddings"`
	Usage      *Usage    `json:"usage,omitempty"`
}

type EmbeddingCapable interface {
	CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error)
}

func (r EmbeddingRequest) Validate() error {
	if r.Model == "" {
		return errors.New("embedding model is required")
	}
	if len(r.Inputs) == 0 {
		return errors.New("at least one embedding input is required")
	}
	if err := r.Inputs.Validate(); err != nil {
		return errors.New("embedding input : " + err.Error())
	}
	return nil
}

func NewEmbeddingRequest(model string, inputs ...ContentBlock) EmbeddingRequest {
	return EmbeddingRequest{
		Model:  model,
		Inputs: inputs,
	}
}

func NewTextEmbeddingRequest(model string, inputs ...string) EmbeddingRequest {
	content := MessageContent{}
	for _, input := range inputs {
		content = append(content, NewTextContentBlock(input))
	}
	return EmbeddingRequest{
		Model:  model,
		Inputs: content,
	}
}

const isNormalizedPrecisionTolerance = 1e-6

func NormalizeVector(v []float32) []float32 {
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	res := make([]float32, len(v))
	if norm == 0 {
		copy(res, v)
		return res
	}
	for i, val := range v {
		res[i] = val / norm
	}

	return res
}

func IsNormalized(v []float32) bool {
	if len(v) == 0 {
		return false
	}
	var sqSum float64
	for _, val := range v {
		sqSum += float64(val) * float64(val)
	}
	magnitude := math.Sqrt(sqSum)
	return math.Abs(magnitude-1) < isNormalizedPrecisionTolerance
}

func NormalizeEmbeddingIfNeeded(values []float32) []float32 {
	if IsNormalized(values) {
		return values
	}
	return NormalizeVector(values)
}
