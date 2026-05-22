package llm

import "testing"

func TestTokenUsageContextOccupiedIncludesSeparateCacheTokens(t *testing.T) {
	tokens := TokenUsage{
		InputTokens:              100,
		OutputTokens:             20,
		CacheReadInputTokens:     900,
		CacheCreationInputTokens: 10,
		CacheBilledSeparately:    true,
	}
	if got, want := tokens.ContextOccupied(), 1030; got != want {
		t.Fatalf("ContextOccupied() = %d, want %d", got, want)
	}
}

func TestTokenUsageContextOccupiedDoesNotDoubleCountIncludedCacheTokens(t *testing.T) {
	tokens := TokenUsage{
		InputTokens:          1000,
		OutputTokens:         20,
		CacheReadInputTokens: 900,
	}
	if got, want := tokens.ContextOccupied(), 1020; got != want {
		t.Fatalf("ContextOccupied() = %d, want %d", got, want)
	}
}
