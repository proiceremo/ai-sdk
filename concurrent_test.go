package llm

import (
	"context"
	"fmt"
	"testing"
	"time"
)

type concurrentEmbeddingClient struct{}

func (c *concurrentEmbeddingClient) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	text := req.Inputs.Text()
	switch text {
	case "slow":
		time.Sleep(20 * time.Millisecond)
		return &EmbeddingResponse{Embeddings: []float32{1}}, nil
	case "fast":
		return &EmbeddingResponse{Embeddings: []float32{2}}, nil
	default:
		return &EmbeddingResponse{Embeddings: []float32{3}}, nil
	}
}

func TestRunConcurrentEmbeddingsPreservesOrder(t *testing.T) {
	results, err := RunConcurrentEmbeddings(context.Background(), &concurrentEmbeddingClient{}, []EmbeddingRequest{
		NewTextEmbeddingRequest("test-model", "slow"),
		NewTextEmbeddingRequest("test-model", "fast"),
		NewTextEmbeddingRequest("test-model", "other"),
	}, ConcurrentConfig{MaxConcurrency: 2})
	if err != nil {
		t.Fatalf("batch returned error: %v", err)
	}

	flattened := [][]float32{
		results[0].Embeddings,
		results[1].Embeddings,
		results[2].Embeddings,
	}
	if got := fmt.Sprint(flattened); got != "[[1] [2] [3]]" {
		t.Fatalf("unexpected batch order: %s", got)
	}
}

func TestRunConcurrentEmbeddingsRequiresClient(t *testing.T) {
	if _, err := RunConcurrentEmbeddings(context.Background(), nil, nil, ConcurrentConfig{}); err != ErrEmbeddingsNotSupported {
		t.Fatalf("expected ErrEmbeddingsNotSupported, got %v", err)
	}
}
