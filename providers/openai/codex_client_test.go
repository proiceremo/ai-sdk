package openai

import (
	llm "github.com/proiceremo/ai-sdk"
	"testing"
)

// codexUsage must surface cache hits — the OpenAI Responses API reports
// them under input_tokens_details.cached_tokens and bundles the count
// INTO input_tokens. Our TokenUsage convention keeps the two separate,
// so the parser has to subtract.
func TestCodexUsageExtractsCacheReadFromInputTokensDetails(t *testing.T) {
	event := map[string]any{
		"response": map[string]any{
			"usage": map[string]any{
				"input_tokens":  float64(10000),
				"output_tokens": float64(2000),
				"total_tokens":  float64(12000),
				"input_tokens_details": map[string]any{
					"cached_tokens": float64(7500),
				},
			},
		},
	}
	usage := codexUsage(event)
	if usage == nil {
		t.Fatalf("codexUsage returned nil")
	}
	tokens := usage.Totals
	if tokens.InputTokens != 2500 {
		t.Errorf("InputTokens=%d want 2500 (10000 raw - 7500 cached)", tokens.InputTokens)
	}
	if tokens.CacheReadInputTokens != 7500 {
		t.Errorf("CacheReadInputTokens=%d want 7500", tokens.CacheReadInputTokens)
	}
	if tokens.OutputTokens != 2000 {
		t.Errorf("OutputTokens=%d want 2000", tokens.OutputTokens)
	}
	if tokens.TotalTokens != 12000 {
		t.Errorf("TotalTokens=%d want 12000 (provider-reported, not recomputed)", tokens.TotalTokens)
	}
}

// When the provider doesn't include input_tokens_details (or sets it to
// zero), cache_read stays zero and InputTokens carries the full count —
// nothing regresses for non-cached responses.
func TestCodexUsageNoCacheDetails(t *testing.T) {
	event := map[string]any{
		"response": map[string]any{
			"usage": map[string]any{
				"input_tokens":  float64(1234),
				"output_tokens": float64(567),
				"total_tokens":  float64(1801),
			},
		},
	}
	usage := codexUsage(event)
	if usage == nil {
		t.Fatalf("codexUsage returned nil")
	}
	if usage.Totals.InputTokens != 1234 {
		t.Errorf("InputTokens=%d want 1234", usage.Totals.InputTokens)
	}
	if usage.Totals.CacheReadInputTokens != 0 {
		t.Errorf("CacheReadInputTokens=%d want 0", usage.Totals.CacheReadInputTokens)
	}
}

// Reasoning models report reasoning_tokens as a SUB-count of
// output_tokens. We surface it on its own field so dashboards can show
// what fraction of output went to hidden chain-of-thought — but we must
// NOT add it to OutputTokens (it's already in there).
func TestCodexUsageExtractsReasoningTokens(t *testing.T) {
	event := map[string]any{
		"response": map[string]any{
			"usage": map[string]any{
				"input_tokens":  float64(500),
				"output_tokens": float64(2000),
				"total_tokens":  float64(2500),
				"output_tokens_details": map[string]any{
					"reasoning_tokens": float64(1500),
				},
			},
		},
	}
	usage := codexUsage(event)
	if usage == nil {
		t.Fatalf("nil usage")
	}
	if usage.Totals.OutputTokens != 2000 {
		t.Errorf("OutputTokens=%d want 2000 (reasoning must NOT be added)", usage.Totals.OutputTokens)
	}
	if usage.Totals.ReasoningOutputTokens != 1500 {
		t.Errorf("ReasoningOutputTokens=%d want 1500", usage.Totals.ReasoningOutputTokens)
	}
}

// Audio token sub-counts land in InputTokensDetails / OutputTokensDetails
// — the existing breakdown slots — rather than getting flattened into
// the top-level counts.
func TestCodexUsageExtractsAudioTokenDetails(t *testing.T) {
	event := map[string]any{
		"response": map[string]any{
			"usage": map[string]any{
				"input_tokens":  float64(800),
				"output_tokens": float64(400),
				"input_tokens_details": map[string]any{
					"audio_tokens": float64(120),
				},
				"output_tokens_details": map[string]any{
					"audio_tokens": float64(50),
				},
			},
		},
	}
	usage := codexUsage(event)
	if usage == nil {
		t.Fatal("nil usage")
	}
	if usage.Totals.InputTokensDetails == nil || usage.Totals.InputTokensDetails.AudioTokens != 120 {
		t.Errorf("InputTokensDetails.AudioTokens missing or wrong: %+v", usage.Totals.InputTokensDetails)
	}
	if usage.Totals.OutputTokensDetails == nil || usage.Totals.OutputTokensDetails.AudioTokens != 50 {
		t.Errorf("OutputTokensDetails.AudioTokens missing or wrong: %+v", usage.Totals.OutputTokensDetails)
	}
}

// If total_tokens is missing we recompute it from the parts. The pre-fix
// version returned input+output only; the fix needs to also add cache_read
// or the running totals downstream go inconsistent.
func TestCodexUsageRecomputesTotalIncludingCache(t *testing.T) {
	event := map[string]any{
		"response": map[string]any{
			"usage": map[string]any{
				"input_tokens":  float64(1000),
				"output_tokens": float64(200),
				"input_tokens_details": map[string]any{
					"cached_tokens": float64(600),
				},
			},
		},
	}
	usage := codexUsage(event)
	if usage == nil {
		t.Fatal("nil usage")
	}
	// 400 fresh + 600 cached + 200 output = 1200
	if usage.Totals.TotalTokens != 1200 {
		t.Errorf("TotalTokens=%d want 1200 (input %d + cache %d + output %d)",
			usage.Totals.TotalTokens, usage.Totals.InputTokens, usage.Totals.CacheReadInputTokens, usage.Totals.OutputTokens)
	}
}

func TestResolveCodexServiceTier(t *testing.T) {
	str := func(s string) *string { return &s }

	tests := []struct {
		name     string
		resp     *string
		req      *string
		expected string
	}{
		{
			name:     "both nil",
			resp:     nil,
			req:      nil,
			expected: "",
		},
		{
			name:     "resp default, req flex -> flex",
			resp:     str("default"),
			req:      str("flex"),
			expected: "flex",
		},
		{
			name:     "resp default, req priority -> priority",
			resp:     str("default"),
			req:      str("priority"),
			expected: "priority",
		},
		{
			name:     "resp default, req other -> default",
			resp:     str("default"),
			req:      str("other"),
			expected: "default",
		},
		{
			name:     "resp set non-default, req set -> resp wins",
			resp:     str("flex"),
			req:      str("priority"),
			expected: "flex",
		},
		{
			name:     "resp nil, req set -> req wins",
			resp:     nil,
			req:      str("flex"),
			expected: "flex",
		},
		{
			name:     "resp set, req nil -> resp wins",
			resp:     str("priority"),
			req:      nil,
			expected: "priority",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := resolveCodexServiceTier(tc.resp, tc.req)
			if got != tc.expected {
				t.Errorf("resolveCodexServiceTier(%v, %v) = %q; want %q", tc.resp, tc.req, got, tc.expected)
			}
		})
	}
}

func codexUsage(event map[string]any) *llm.Usage {
	return newCodexStream(nil).codexUsage(event)
}


