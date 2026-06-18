package llm_test

import (
	"context"
	"strings"
	"testing"

	llm "github.com/proiceremo/ai-sdk"
	_ "github.com/proiceremo/ai-sdk/providers/anthropic"
	_ "github.com/proiceremo/ai-sdk/providers/google"
	_ "github.com/proiceremo/ai-sdk/providers/openai"
)

func TestSanitizeSurrogates(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    "Hello World!",
			expected: "Hello World!",
		},
		{
			input:    "Hello \xed\xa0\x80 World!", // surrogate code point 0xD800 encoded as UTF-8 bytes
			expected: "Hello  World!",
		},
		{
			input:    "Hello \xff World!", // invalid UTF-8 byte
			expected: "Hello  World!",
		},
		{
			input:    "Valid emoji: 🌟 and text.",
			expected: "Valid emoji: 🌟 and text.",
		},
	}

	for _, tc := range tests {
		got := llm.SanitizeSurrogates(tc.input)
		if got != tc.expected {
			t.Errorf("SanitizeSurrogates(%q) = %q; want %q", tc.input, got, tc.expected)
		}
	}
}

func TestDefaultRegistryResolvesFactories(t *testing.T) {
	formats := []llm.APIFormat{
		llm.APIFormatOpenAI,
		llm.APIFormatOpenAIResponses,
		llm.APIFormatOpenAICodex,
		llm.APIFormatAnthropic,
		llm.APIFormatGoogle,
	}

	for _, format := range formats {
		registry := llm.NewRegistry().WithProviders(llm.ProviderConfig{
			ID:     "test-prov",
			Format: format,
		}).WithModels(llm.ModelConfig{
			ID:         "test-model",
			ProviderID: "test-prov",
		})
		_, _, err := registry.Resolve(context.Background(), "test-model")
		if err != nil && strings.Contains(err.Error(), "no provider factory registered") {
			t.Errorf("expected factory registered for format %q, but Resolve failed with: %v", format, err)
		}
	}
}
