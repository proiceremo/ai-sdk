package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

type dummyClient struct {
	transientErr bool
	callCount    int
}

func (c *dummyClient) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	c.callCount++
	if c.transientErr {
		return nil, errors.New("429 rate limit")
	}
	return &Message{Role: MessageRoleAssistant}, nil
}

func (c *dummyClient) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	c.callCount++
	if c.transientErr {
		return nil, errors.New("429 rate limit")
	}
	return nil, errors.New("stream not implemented")
}

func (c *dummyClient) SupportsStreaming() bool { return false }

func TestProviderRateLimiterConcurrency(t *testing.T) {
	limiter := NewProviderRateLimiter("test", 2, 0)
	defer limiter.Stop()

	ctx := context.Background()
	if err := limiter.Acquire(ctx); err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if err := limiter.Acquire(ctx); err != nil {
		t.Fatalf("second acquire: %v", err)
	}

	// Third acquire should block until we release
	done := make(chan struct{})
	go func() {
		if err := limiter.Acquire(ctx); err != nil {
			t.Errorf("third acquire: %v", err)
		}
		close(done)
	}()

	select {
	case <-done:
		t.Fatal("third acquire should have blocked")
	case <-time.After(50 * time.Millisecond):
		// expected
	}

	limiter.Release()
	select {
	case <-done:
		// expected
	case <-time.After(time.Second):
		t.Fatal("third acquire did not unblock after release")
	}
	limiter.Release()
}

func TestProviderCircuitBreakerTripsAfterFailures(t *testing.T) {
	cb := NewProviderCircuitBreaker("test", 3, 100*time.Millisecond)

	if cb.IsTripped() {
		t.Fatal("breaker should not be tripped initially")
	}

	for i := 0; i < 2; i++ {
		if cb.RecordFailure() {
			t.Fatalf("breaker should not trip on failure %d", i+1)
		}
	}

	if !cb.RecordFailure() {
		t.Fatal("breaker should trip on third failure")
	}

	if !cb.IsTripped() {
		t.Fatal("breaker should be tripped")
	}

	// Wait for reset
	time.Sleep(150 * time.Millisecond)
	if cb.IsTripped() {
		t.Fatal("breaker should reset after timeout")
	}
}

func TestProviderCircuitBreakerSuccessResets(t *testing.T) {
	cb := NewProviderCircuitBreaker("test", 3, time.Hour)
	cb.RecordFailure()
	cb.RecordFailure()
	cb.RecordSuccess()

	if cb.IsTripped() {
		t.Fatal("success should reset failures")
	}

	if cb.RecordFailure() {
		t.Fatal("should not trip after reset on first failure")
	}
}

func TestClientWithRateLimitBlocksOnBreaker(t *testing.T) {
	client := &dummyClient{transientErr: true}
	limiter := NewProviderRateLimiter("test", 10, 0)
	defer limiter.Stop()
	breaker := NewProviderCircuitBreaker("test", 2, time.Hour)

	wrapped := newClientWithRateLimit(client, limiter, breaker)
	ctx := context.Background()

	// First failure
	_, err := wrapped.CreateCompletion(ctx, nil, InferenceParams{})
	if err == nil {
		t.Fatal("expected error")
	}

	// Second failure trips breaker
	_, err = wrapped.CreateCompletion(ctx, nil, InferenceParams{})
	if err == nil {
		t.Fatal("expected error")
	}

	// Third call should be blocked by breaker
	_, err = wrapped.CreateCompletion(ctx, nil, InferenceParams{})
	if err == nil {
		t.Fatal("expected error from breaker")
	}
	if !errors.Is(err, context.Canceled) && !containsStr(err.Error(), "circuit breaker is open") {
		t.Fatalf("expected circuit breaker error, got: %v", err)
	}
}

func containsStr(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || (len(s) > 0 && len(substr) > 0 && s[0:len(substr)] == substr || (len(s) > len(substr) && containsSub(s, substr))))
}

func containsSub(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestClientWithRateLimitSuccessClearsBreaker(t *testing.T) {
	client := &dummyClient{transientErr: false}
	limiter := NewProviderRateLimiter("test", 10, 0)
	defer limiter.Stop()
	breaker := NewProviderCircuitBreaker("test", 2, time.Hour)

	wrapped := newClientWithRateLimit(client, limiter, breaker)
	ctx := context.Background()

	_, err := wrapped.CreateCompletion(ctx, nil, InferenceParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if breaker.failures != 0 {
		t.Fatalf("expected failures reset, got %d", breaker.failures)
	}
}
