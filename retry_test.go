package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"
)

type retryTestClient struct {
	attempts          int
	failures          int
	embeddingAttempts int
	embeddingFailures int
}

func (c *retryTestClient) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	c.attempts++
	if c.attempts <= c.failures {
		return nil, errors.New("429 rate limit")
	}
	return &Message{Role: MessageRoleAssistant}, nil
}

func (c *retryTestClient) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	return nil, errors.New("not implemented")
}

func (c *retryTestClient) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	c.embeddingAttempts++
	if c.embeddingAttempts <= c.embeddingFailures {
		return nil, errors.New("429 rate limit")
	}
	return &EmbeddingResponse{Embeddings: []float32{1}}, nil
}

func TestWrapClientWithRetryNoRetryReturnsOriginal(t *testing.T) {
	client := &retryTestClient{}
	wrapped := WrapClientWithRetry(client, NoRetryPolicy())
	if wrapped != client {
		t.Fatalf("expected original client when retry is disabled")
	}
}

func TestClientWithRetryRetriesTransientErrors(t *testing.T) {
	client := &retryTestClient{failures: 2}
	wrapped := WrapClientWithRetry(client, RetryPolicy{
		Strategy:     RetryStrategyFixedDelay,
		MaxRetries:   3,
		InitialDelay: 1,
		MaxDelay:     1,
	})

	msg, err := wrapped.CreateCompletion(context.Background(), nil, InferenceParams{})
	if err != nil {
		t.Fatalf("CreateCompletion returned error: %v", err)
	}
	if msg == nil {
		t.Fatal("expected completion message")
	}
	if client.attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", client.attempts)
	}
}

func TestWrapClientWithRetryPreservesEmbeddingCapability(t *testing.T) {
	client := &retryTestClient{embeddingFailures: 1}
	wrapped := WrapClientWithRetry(client, RetryPolicy{
		Strategy:     RetryStrategyFixedDelay,
		MaxRetries:   2,
		InitialDelay: 1,
		MaxDelay:     1,
	})

	embeddingClient, ok := wrapped.(EmbeddingCapable)
	if !ok {
		t.Fatal("expected wrapped client to preserve embedding capability")
	}

	resp, err := embeddingClient.CreateEmbeddings(context.Background(), NewTextEmbeddingRequest("test-model", "hello"))
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %v", err)
	}
	if got := resp.Embeddings; len(got) != 1 || got[0] != 1 {
		t.Fatalf("unexpected embedding response: %#v", got)
	}
	if client.embeddingAttempts != 2 {
		t.Fatalf("expected 2 embedding attempts, got %d", client.embeddingAttempts)
	}
}

func TestRetryPolicyJitter(t *testing.T) {
	policy := RetryPolicy{
		Strategy:     RetryStrategyExponentialBackoff,
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   2,
		Jitter:       0.5,
	}

	var minDelay, maxDelay time.Duration = time.Hour, 0
	for i := 0; i < 100; i++ {
		d := policy.backoff(0)
		if d < minDelay {
			minDelay = d
		}
		if d > maxDelay {
			maxDelay = d
		}
	}

	base := policy.InitialDelay
	if minDelay == base && maxDelay == base {
		t.Fatal("jitter did not vary the delay")
	}
	if minDelay < 0 {
		t.Fatalf("jitter produced negative delay: %v", minDelay)
	}
	if maxDelay > policy.MaxDelay {
		t.Fatalf("jitter exceeded max delay: %v > %v", maxDelay, policy.MaxDelay)
	}
}

func TestRetryPolicyNoJitter(t *testing.T) {
	policy := RetryPolicy{
		Strategy:     RetryStrategyExponentialBackoff,
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   2,
		Jitter:       0,
	}

	for i := 0; i < 10; i++ {
		d := policy.backoff(1)
		expected := time.Duration(float64(policy.InitialDelay.Milliseconds())*policy.Multiplier) * time.Millisecond
		if d != expected {
			t.Fatalf("no jitter: expected %v, got %v", expected, d)
		}
	}
}

type retryAfterError struct {
	Response *http.Response
}

func (e *retryAfterError) Error() string {
	return fmt.Sprintf("HTTP %d", e.Response.StatusCode)
}

func TestRetryDelayHonorsRetryAfterHeader(t *testing.T) {
	err := &retryAfterError{Response: &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": []string{"7"}}}}
	policy := RetryPolicy{
		Strategy:          RetryStrategyFixedDelay,
		MaxRetries:        1,
		InitialDelay:      time.Second,
		MaxDelay:          10 * time.Second,
		RespectRetryAfter: true,
	}
	if got := policy.retryDelay(0, err); got != 7*time.Second {
		t.Fatalf("retry delay = %v, want 7s", got)
	}
}
