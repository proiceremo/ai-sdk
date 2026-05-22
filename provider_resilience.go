package llm

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"
)

// ProviderRateLimiter enforces per-provider concurrency and RPM limits.
type ProviderRateLimiter struct {
	providerID     string
	maxConcurrency int
	maxRPM         int
	sem            chan struct{}
	tokens         chan struct{}
	stopTicker     chan struct{}
	mu             sync.Mutex
}

// NewProviderRateLimiter creates a rate limiter for a provider.
// maxConcurrency: maximum concurrent requests (0 = unlimited).
// maxRPM: maximum requests per minute (0 = unlimited).
func NewProviderRateLimiter(providerID string, maxConcurrency, maxRPM int) *ProviderRateLimiter {
	prl := &ProviderRateLimiter{
		providerID:     providerID,
		maxConcurrency: maxConcurrency,
		maxRPM:         maxRPM,
		stopTicker:     make(chan struct{}),
	}
	if maxConcurrency > 0 {
		prl.sem = make(chan struct{}, maxConcurrency)
	}
	if maxRPM > 0 {
		prl.tokens = make(chan struct{}, maxRPM)
		// Fill the bucket
		for i := 0; i < maxRPM; i++ {
			prl.tokens <- struct{}{}
		}
		// Replenish one token every 60/maxRPM seconds
		interval := time.Duration(60_000_000_000/maxRPM) * time.Nanosecond
		go func() {
			ticker := time.NewTicker(interval)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					select {
					case prl.tokens <- struct{}{}:
					default:
					}
				case <-prl.stopTicker:
					return
				}
			}
		}()
	}
	return prl
}

// Stop halts the token replenishment goroutine.
func (prl *ProviderRateLimiter) Stop() {
	close(prl.stopTicker)
}

// Acquire blocks until a slot is available for the provider.
// Returns an error if the context is cancelled.
func (prl *ProviderRateLimiter) Acquire(ctx context.Context) error {
	if prl.maxConcurrency > 0 {
		select {
		case prl.sem <- struct{}{}:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if prl.maxRPM > 0 {
		select {
		case <-prl.tokens:
		case <-ctx.Done():
			if prl.maxConcurrency > 0 {
				<-prl.sem
			}
			return ctx.Err()
		}
	}
	return nil
}

// Release returns a slot to the limiter.
func (prl *ProviderRateLimiter) Release() {
	if prl.maxConcurrency > 0 {
		select {
		case <-prl.sem:
		default:
		}
	}
}

// ProviderCircuitBreaker is a global per-provider circuit breaker.
type ProviderCircuitBreaker struct {
	providerID  string
	maxFailures int
	resetAfter  time.Duration
	mu          sync.RWMutex
	failures    int
	lastFailure time.Time
	paused      bool
}

// NewProviderCircuitBreaker creates a circuit breaker for a provider.
func NewProviderCircuitBreaker(providerID string, maxFailures int, resetAfter time.Duration) *ProviderCircuitBreaker {
	return &ProviderCircuitBreaker{
		providerID:  providerID,
		maxFailures: maxFailures,
		resetAfter:  resetAfter,
	}
}

// RecordSuccess resets the failure count.
func (pcb *ProviderCircuitBreaker) RecordSuccess() {
	pcb.mu.Lock()
	defer pcb.mu.Unlock()
	pcb.failures = 0
	pcb.paused = false
}

// RecordFailure increments the failure count and returns true if tripped.
func (pcb *ProviderCircuitBreaker) RecordFailure() bool {
	pcb.mu.Lock()
	defer pcb.mu.Unlock()
	pcb.failures++
	pcb.lastFailure = time.Now()
	if pcb.failures >= pcb.maxFailures {
		pcb.paused = true
		return true
	}
	return false
}

// IsTripped returns true if the breaker is open.
func (pcb *ProviderCircuitBreaker) IsTripped() bool {
	pcb.mu.RLock()
	if !pcb.paused {
		pcb.mu.RUnlock()
		return false
	}
	last := pcb.lastFailure
	pcb.mu.RUnlock()
	if time.Since(last) > pcb.resetAfter {
		pcb.mu.Lock()
		pcb.paused = false
		pcb.failures = 0
		pcb.mu.Unlock()
		return false
	}
	return true
}

// ClientWithRateLimit wraps a Client with provider-level rate limiting.
type ClientWithRateLimit struct {
	client    Client
	limiter   *ProviderRateLimiter
	breaker   *ProviderCircuitBreaker
	embedding EmbeddingCapable
}

func newClientWithRateLimit(client Client, limiter *ProviderRateLimiter, breaker *ProviderCircuitBreaker) *ClientWithRateLimit {
	wrapped := &ClientWithRateLimit{
		client:  client,
		limiter: limiter,
		breaker: breaker,
	}
	if embedding, ok := client.(EmbeddingCapable); ok {
		wrapped.embedding = embedding
	}
	return wrapped
}

func (c *ClientWithRateLimit) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	if c.breaker != nil && c.breaker.IsTripped() {
		return nil, fmt.Errorf("provider circuit breaker is open")
	}
	if c.limiter != nil {
		if err := c.limiter.Acquire(ctx); err != nil {
			return nil, err
		}
		defer c.limiter.Release()
	}
	msg, err := c.client.CreateCompletion(ctx, messages, params)
	if err != nil {
		if c.breaker != nil && isTransientError(err) {
			c.breaker.RecordFailure()
		}
		return nil, err
	}
	if c.breaker != nil {
		c.breaker.RecordSuccess()
	}
	return msg, nil
}

func (c *ClientWithRateLimit) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	if c.breaker != nil && c.breaker.IsTripped() {
		return nil, fmt.Errorf("provider circuit breaker is open")
	}
	if c.limiter != nil {
		if err := c.limiter.Acquire(ctx); err != nil {
			return nil, err
		}
	}
	stream, err := c.client.CreateCompletionStream(ctx, messages, params)
	if err != nil {
		if c.limiter != nil {
			c.limiter.Release()
		}
		if c.breaker != nil && isTransientError(err) {
			c.breaker.RecordFailure()
		}
		return nil, err
	}
	return &rateLimitedStream{inner: stream, limiter: c.limiter, breaker: c.breaker}, nil
}

func (c *ClientWithRateLimit) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	if c.breaker != nil && c.breaker.IsTripped() {
		return nil, fmt.Errorf("provider circuit breaker is open")
	}
	if c.limiter != nil {
		if err := c.limiter.Acquire(ctx); err != nil {
			return nil, err
		}
		defer c.limiter.Release()
	}
	resp, err := c.embedding.CreateEmbeddings(ctx, req)
	if err != nil {
		if c.breaker != nil && isTransientError(err) {
			c.breaker.RecordFailure()
		}
		return nil, err
	}
	if c.breaker != nil {
		c.breaker.RecordSuccess()
	}
	return resp, nil
}

func (c *ClientWithRateLimit) SupportsStreaming() bool {
	if capable, ok := c.client.(StreamingCapable); ok {
		return capable.SupportsStreaming()
	}
	return false
}

type rateLimitedStream struct {
	inner   Stream
	limiter *ProviderRateLimiter
	breaker *ProviderCircuitBreaker
	closed  bool
}

func (s *rateLimitedStream) Recv(ctx context.Context) (*StreamEvent, error) {
	event, err := s.inner.Recv(ctx)
	if err != nil {
		if s.breaker != nil && isTransientError(err) {
			s.breaker.RecordFailure()
		}
		if err == io.EOF {
			_ = s.Close()
		}
		return nil, err
	}
	return event, nil
}

func (s *rateLimitedStream) Current() *Message {
	return s.inner.Current()
}

func (s *rateLimitedStream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	if s.limiter != nil {
		s.limiter.Release()
	}
	return s.inner.Close()
}
