package llm

import (
	"os"

	"github.com/proiceremo/ai-sdk/oauthx"
)

var (
	textOnly        = []Modality{ModalityText}
	multimodalInput = []Modality{ModalityText, ModalityImage, ModalityVideo, ModalityAudio}
	textAndImage    = []Modality{ModalityText, ModalityImage}
	textImageVideo  = []Modality{ModalityText, ModalityImage, ModalityVideo}

	gemini3FlashPricing = &ModelCosts{
		ModelPricing: ModelPricing{
			InputTokensPer1M:     0.5,
			OutputTokensPer1M:    3.0,
			CacheReadTokensPer1M: 0.05,
			InputCostByModality: &ModalityPricing{
				AudioTokensPer1M: Ptr(1.0),
			},
			CacheReadCostByModality: &ModalityPricing{
				AudioTokensPer1M: Ptr(0.1),
			},
		},
	}
	gemini35FlashPricing = &ModelCosts{
		ModelPricing: ModelPricing{
			InputTokensPer1M:      1.5,
			OutputTokensPer1M:     9.0,
			CacheReadTokensPer1M:  0.15,
			CacheWriteTokensPer1M: 0.083333,
		},
	}
	gemini31FlashLitePricing = &ModelCosts{
		ModelPricing: ModelPricing{
			InputTokensPer1M:     0.25,
			OutputTokensPer1M:    1.5,
			CacheReadTokensPer1M: 0.03,
			InputCostByModality: &ModalityPricing{
				AudioTokensPer1M: Ptr(0.5),
			},
			CacheReadCostByModality: &ModalityPricing{
				AudioTokensPer1M: Ptr(0.05),
			},
		},
	}
	gemini3ProPricing = &ModelCosts{
		Tiers: []PriceTier{
			{Boundary: 200000, ModelPricing: ModelPricing{InputTokensPer1M: 2.0, OutputTokensPer1M: 12.0, CacheReadTokensPer1M: 0.2}},
			{Boundary: -1, ModelPricing: ModelPricing{InputTokensPer1M: 4.0, OutputTokensPer1M: 18.0, CacheReadTokensPer1M: 0.4}},
		},
	}

	gpt55Pricing = &ModelCosts{
		ModelPricing: ModelPricing{
			InputTokensPer1M:     5.0,
			OutputTokensPer1M:    30.0,
			CacheReadTokensPer1M: 0.5,
		},
		Tiers: []PriceTier{
			{
				Boundary: 272000,
				ModelPricing: ModelPricing{
					InputTokensPer1M:     5.0,
					OutputTokensPer1M:    30.0,
					CacheReadTokensPer1M: 0.5,
				},
			},
			{
				Boundary: -1,
				ModelPricing: ModelPricing{
					InputTokensPer1M:     10.0,
					OutputTokensPer1M:    45.0,
					CacheReadTokensPer1M: 1.0,
				},
			},
		},
	}
	gpt54Pricing = &ModelCosts{
		ModelPricing: ModelPricing{
			InputTokensPer1M:     2.5,
			OutputTokensPer1M:    15.0,
			CacheReadTokensPer1M: 0.25,
		},
		Tiers: []PriceTier{
			{
				Boundary: 272000,
				ModelPricing: ModelPricing{
					InputTokensPer1M:     2.5,
					OutputTokensPer1M:    15.0,
					CacheReadTokensPer1M: 0.25,
				},
			},
			{
				Boundary: -1,
				ModelPricing: ModelPricing{
					InputTokensPer1M:     5.0,
					OutputTokensPer1M:    22.5,
					CacheReadTokensPer1M: 0.5,
				},
			},
		},
	}
	gpt54MiniPricing = &ModelCosts{
		ModelPricing: ModelPricing{
			InputTokensPer1M:     0.75,
			OutputTokensPer1M:    4.5,
			CacheReadTokensPer1M: 0.075,
		},
	}
	umansPricing = &ModelCosts{
		ModelPricing: ModelPricing{
			InputTokensPer1M:  0.0,
			OutputTokensPer1M: 0.0,
		},
	}

	DefaultModels = []ModelConfig{
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.5", ModelID: "gpt-5.5", Name: "OpenAI Codex GPT-5.5",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt55Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelMedium},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.5-none", ModelID: "gpt-5.5", Name: "OpenAI Codex GPT-5.5 (Thinking: None)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt55Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: false},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.5-low", ModelID: "gpt-5.5", Name: "OpenAI Codex GPT-5.5 (Thinking: Low)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt55Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelLow},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.5-medium", ModelID: "gpt-5.5", Name: "OpenAI Codex GPT-5.5 (Thinking: Medium)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt55Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelMedium},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.5-high", ModelID: "gpt-5.5", Name: "OpenAI Codex GPT-5.5 (Thinking: High)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt55Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelHigh},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.5-xhigh", ModelID: "gpt-5.5", Name: "OpenAI Codex GPT-5.5 (Thinking: XHigh)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt55Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4", ModelID: "gpt-5.4", Name: "OpenAI Codex GPT-5.4",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelMedium},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-none", ModelID: "gpt-5.4", Name: "OpenAI Codex GPT-5.4 (Thinking: None)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: false},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-low", ModelID: "gpt-5.4", Name: "OpenAI Codex GPT-5.4 (Thinking: Low)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelLow},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-medium", ModelID: "gpt-5.4", Name: "OpenAI Codex GPT-5.4 (Thinking: Medium)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelMedium},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-high", ModelID: "gpt-5.4", Name: "OpenAI Codex GPT-5.4 (Thinking: High)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelHigh},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-xhigh", ModelID: "gpt-5.4", Name: "OpenAI Codex GPT-5.4 (Thinking: XHigh)",
			ContextWindow:   1050000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54Pricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-mini", ModelID: "gpt-5.4-mini", Name: "OpenAI Codex GPT-5.4 Mini",
			ContextWindow:   400000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54MiniPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelMedium},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-mini-none", ModelID: "gpt-5.4-mini", Name: "OpenAI Codex GPT-5.4 Mini (Thinking: None)",
			ContextWindow:   400000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54MiniPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: false},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-mini-low", ModelID: "gpt-5.4-mini", Name: "OpenAI Codex GPT-5.4 Mini (Thinking: Low)",
			ContextWindow:   400000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54MiniPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelLow},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-mini-medium", ModelID: "gpt-5.4-mini", Name: "OpenAI Codex GPT-5.4 Mini (Thinking: Medium)",
			ContextWindow:   400000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54MiniPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelMedium},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-mini-high", ModelID: "gpt-5.4-mini", Name: "OpenAI Codex GPT-5.4 Mini (Thinking: High)",
			ContextWindow:   400000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54MiniPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelHigh},
			},
		},
		{
			ProviderID: "openai-codex", ID: "openai-codex/gpt-5.4-mini-xhigh", ModelID: "gpt-5.4-mini", Name: "OpenAI Codex GPT-5.4 Mini (Thinking: XHigh)",
			ContextWindow:   400000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: gpt54MiniPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "zenmux-anthropic", ID: "zenmux/step-3.5-flash-free", ModelID: "stepfun/step-3.5-flash-free", Name: "Step-3.5 Flash (Free)",
			ContextWindow:   256000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0, OutputTokensPer1M: 0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(256000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "zenmux-anthropic", ID: "zenmux/glm-4.7-flash-free", ModelID: "z-ai/glm-4.7-flash-free", Name: "GLM-4.7 Flash (Free)",
			ContextWindow:   200000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0, OutputTokensPer1M: 0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "zenmux-vertex", ID: "zenmux/gemini-3-pro-preview", ModelID: "google/gemini-3-pro-preview", Name: "Gemini 3 Pro",
			ContextWindow:   1000000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: gemini3ProPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65536),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "zenmux-vertex", ID: "zenmux/gemini-3-flash-preview", ModelID: "google/gemini-3-flash-preview", Name: "Gemini 3 Flash",
			ContextWindow:   1000000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: gemini3FlashPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65536),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "zenmux-vertex", ID: "zenmux/ming-flash-omni-preview", ModelID: "inclusionai/ming-flash-omni-preview", Name: "Ming Flash Omni",
			ContextWindow:   64000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.8, OutputTokensPer1M: 1.8}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
			},
		},
		{
			ProviderID: "deepseek", ID: "deepseek/deepseek-v4-flash", ModelID: "deepseek-v4-flash", Name: "Deepseek V4 Flash",
			ContextWindow:   1_000_000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.14, OutputTokensPer1M: 0.28, CacheReadTokensPer1M: 0.0028}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(384_000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "deepseek", ID: "deepseek/deepseek-v4-pro", ModelID: "deepseek-v4-pro", Name: "Deepseek V4 Pro",
			ContextWindow:   1_000_000,
			InputModalities: textOnly, OutputModalities: textOnly,
			// 75% off till 31st may 2026
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.435, OutputTokensPer1M: 0.87, CacheReadTokensPer1M: 0.003625}},
			// Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 1.74, OutputTokensPer1M: 3.48, CacheReadTokensPer1M: 0.0145}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(384_000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "moonshot", ID: "moonshot/kimi-k2p6", ModelID: "kimi-k2p6", Name: "Kimi K2.6",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.60, OutputTokensPer1M: 3.0, CacheReadTokensPer1M: 0.10}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "moonshot", ID: "moonshot/kimi-k2-0905-preview", ModelID: "kimi-k2-0905-preview", Name: "Kimi K2 (0905)",
			ContextWindow:   262144,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.60, OutputTokensPer1M: 2.50, CacheReadTokensPer1M: 0.15}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
			},
		},
		{
			ProviderID: "moonshot", ID: "moonshot/kimi-k2-thinking", ModelID: "kimi-k2-thinking", Name: "Kimi K2 Thinking",
			ContextWindow:   262144,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.60, OutputTokensPer1M: 2.50, CacheReadTokensPer1M: 0.15}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "minimax", ID: "minimax/MiniMax-M2.7-highspeed", ModelID: "MiniMax-M2.7-highspeed", Name: "MiniMax M2.7 Highspeed",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.30, OutputTokensPer1M: 1.20, CacheReadTokensPer1M: 0.03, CacheWriteTokensPer1M: 0.375}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "minimax", ID: "minimax/MiniMax-M2.7", ModelID: "MiniMax-M2.7", Name: "MiniMax M2.7",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.30, OutputTokensPer1M: 1.20, CacheReadTokensPer1M: 0.03, CacheWriteTokensPer1M: 0.375}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "minimax", ID: "minimax/MiniMax-M2.5-highspeed", ModelID: "MiniMax-M2.5-highspeed", Name: "MiniMax M2.5 Highspeed",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.30, OutputTokensPer1M: 1.20, CacheReadTokensPer1M: 0.03, CacheWriteTokensPer1M: 0.375}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "minimax", ID: "minimax/MiniMax-M2.5", ModelID: "MiniMax-M2.5", Name: "MiniMax M2.5",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.30, OutputTokensPer1M: 1.20, CacheReadTokensPer1M: 0.03, CacheWriteTokensPer1M: 0.375}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "minimax", ID: "minimax/MiniMax-M2.1", ModelID: "MiniMax-M2.1", Name: "MiniMax M2.1",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.30, OutputTokensPer1M: 1.20, CacheReadTokensPer1M: 0.03, CacheWriteTokensPer1M: 0.375}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "minimax", ID: "minimax/MiniMax-M2.1-lightning", ModelID: "MiniMax-M2.1-lightning", Name: "MiniMax M2.1 Lightning",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.30, OutputTokensPer1M: 2.40, CacheReadTokensPer1M: 0.03, CacheWriteTokensPer1M: 0.375}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "google", ID: "google/glm-5", ModelID: "glm-5-maas", Name: "GLM 5",
			ContextWindow:   200000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{CacheReadTokensPer1M: 0.1},
				Tiers: []PriceTier{
					{Boundary: -1, ModelPricing: ModelPricing{InputTokensPer1M: 1.0, OutputTokensPer1M: 3.2}},
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "google", ID: "google/gemini-3.1-pro-preview", ModelID: "gemini-3.1-pro-preview", Name: "Gemini 3.1 Pro",
			ContextWindow:   1000000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: gemini3ProPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65535),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "google", ID: "google/gemini-3.1-flash-lite-preview", ModelID: "gemini-3.1-flash-lite-preview", Name: "Gemini 3.1 Flash Lite",
			ContextWindow:   1000000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: gemini31FlashLitePricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65535),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "google", ID: "google/gemini-3-flash-preview", ModelID: "gemini-3-flash-preview", Name: "Gemini 3 Flash",
			ContextWindow:   1000000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: gemini3FlashPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65536),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			// Gemini 3.5 Flash via Vertex AI. PDF is part of the wire
			// modality list per the model card; our Modality enum doesn't
			// have a dedicated PDF entry, so PDFs flow through the
			// generic document path the Google client already handles.
			ProviderID: "google-vertex", ID: "google-vertex/gemini-3.5-flash", ModelID: "gemini-3.5-flash", Name: "Gemini 3.5 Flash",
			ContextWindow:   1048576,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: gemini35FlashPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65536),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "google", ID: "google/gemini-3-pro-preview", ModelID: "gemini-3-pro-preview", Name: "Gemini 3 Pro",
			ContextWindow:   1000000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: gemini3ProPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65535),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "google", ID: "google/gemini-2.5-flash-lite", ModelID: "gemini-2.5-flash-lite", Name: "Gemini 2.5 Flash Lite",
			ContextWindow:   1000000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.1, OutputTokensPer1M: 0.4}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65535),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-7",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-7-none",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: None) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-7-low",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: Low) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-7-medium",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: Medium) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-7-high",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: High) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-7-xhigh",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: XHigh) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-8",
			ModelID:          "claude-opus-4-8",
			Name:             "Claude 4.8 Opus (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-8-none",
			ModelID:          "claude-opus-4-8",
			Name:             "Claude 4.8 Opus (Thinking: None) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-8-low",
			ModelID:          "claude-opus-4-8",
			Name:             "Claude 4.8 Opus (Thinking: Low) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-8-medium",
			ModelID:          "claude-opus-4-8",
			Name:             "Claude 4.8 Opus (Thinking: Medium) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-8-high",
			ModelID:          "claude-opus-4-8",
			Name:             "Claude 4.8 Opus (Thinking: High) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-8-xhigh",
			ModelID:          "claude-opus-4-8",
			Name:             "Claude 4.8 Opus (Thinking: XHigh) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-6",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-6-none",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: None) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-6-low",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: Low) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-6-medium",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: Medium) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-6-high",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: High) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-6-xhigh",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: XHigh) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-6",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-6-none",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: None) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-6-low",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: Low) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-6-medium",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: Medium) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-6-high",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: High) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-6-xhigh",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: XHigh) (OAuth)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-5",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-5-none",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: None) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-5-low",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: Low) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-5-medium",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: Medium) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-5-high",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: High) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-opus-4-5-xhigh",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: XHigh) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-5",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-5-none",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: None) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-5-low",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: Low) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-5-medium",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: Medium) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-5-high",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: High) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-sonnet-4-5-xhigh",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: XHigh) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-haiku-4-5",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-haiku-4-5-none",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: None) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-haiku-4-5-low",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: Low) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-haiku-4-5-medium",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: Medium) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-haiku-4-5-high",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: High) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic-oauth",
			ID:               "anthropic-oauth/claude-haiku-4-5-xhigh",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: XHigh) (OAuth)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-7",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-7-none",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: None)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-7-low",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: Low)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-7-medium",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: Medium)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-7-high",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: High)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-7-xhigh",
			ModelID:          "claude-opus-4-7",
			Name:             "Claude 4.7 Opus (Thinking: XHigh)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-6",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-6-none",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: None)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-6-low",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: Low)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-6-medium",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: Medium)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-6-high",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: High)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-6-xhigh",
			ModelID:          "claude-sonnet-4-6",
			Name:             "Claude 4.6 Sonnet (Thinking: XHigh)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-6",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-6-none",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: None)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-6-low",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: Low)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-6-medium",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: Medium)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-6-high",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: High)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-6-xhigh",
			ModelID:          "claude-opus-4-6",
			Name:             "Claude 4.6 Opus (Thinking: XHigh)",
			ContextWindow:    1000000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-5",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-5-none",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: None)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-5-low",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: Low)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-5-medium",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: Medium)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-5-high",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: High)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-opus-4-5-xhigh",
			ModelID:          "claude-opus-4-5",
			Name:             "Claude 4.5 Opus (Thinking: XHigh)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         5,
					OutputTokensPer1M:        25,
					CacheWriteTokensPer1M:    6.25,
					CacheReadTokensPer1M:     0.5,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-5",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-5-none",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: None)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-5-low",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: Low)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-5-medium",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: Medium)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-5-high",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: High)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-sonnet-4-5-xhigh",
			ModelID:          "claude-sonnet-4-5",
			Name:             "Claude 4.5 Sonnet (Thinking: XHigh)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         3,
					OutputTokensPer1M:        15,
					CacheWriteTokensPer1M:    3.75,
					CacheReadTokensPer1M:     0.3,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-haiku-4-5",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-haiku-4-5-none",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: None)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: false,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-haiku-4-5-low",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: Low)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelLow,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-haiku-4-5-medium",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: Medium)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelMedium,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-haiku-4-5-high",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: High)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelHigh,
				},
			},
		},
		{
			ProviderID:       "anthropic",
			ID:               "anthropic/claude-haiku-4-5-xhigh",
			ModelID:          "claude-haiku-4-5",
			Name:             "Claude 4.5 Haiku (Thinking: XHigh)",
			ContextWindow:    200000,
			InputModalities:  textAndImage,
			OutputModalities: textOnly,
			Costs: &ModelCosts{
				ModelPricing: ModelPricing{
					InputTokensPer1M:         1,
					OutputTokensPer1M:        5,
					CacheWriteTokensPer1M:    1.25,
					CacheReadTokensPer1M:     0.1,
					InputCostByModality:      nil,
					OutputCostByModality:     nil,
					CacheReadCostByModality:  nil,
					CacheWriteCostByModality: nil,
				},
			},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "qwen-plan", ID: "qwen/kimi-k2p6", ModelID: "kimi-k2p6", Name: "Kimi K2.6",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "qwen-plan", ID: "qwen/glm-5", ModelID: "glm-5", Name: "GLM-5",
			ContextWindow:   262144,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "qwen-plan", ID: "qwen/minimax-m2.5", ModelID: "minimax-m2.5", Name: "Minimax M2.5",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "qwen-plan", ID: "qwen/qwen3.5-plus", ModelID: "qwen3.5-plus", Name: "Qwen 3.5 Plus",
			ContextWindow:   991000,
			InputModalities: textImageVideo, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "qwen-plan", ID: "qwen/qwen3-max", ModelID: "qwen3-max-2026-01-23", Name: "Qwen 3.5 Plus",
			ContextWindow:   252000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(64000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "qwen-plan", ID: "qwen/qwen3-coder-next", ModelID: "qwen3-coder-next", Name: "Qwen 3 Coder Next",
			ContextWindow:   256000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "qwen-plan", ID: "qwen/qwen3-coder-plus", ModelID: "qwen3-coder-plus", Name: "Qwen 3 Coder Plus",
			ContextWindow:   995000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "llamacpp-anthropic", ID: "local/qwen-3.5-35b", ModelID: "unsloth/Qwen3.5-35B-A3B-GGUF:Q3_K_S", Name: "Qwen 3.5 35B A3B",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "llamacpp-anthropic", ID: "local/omnicoder-9b", ModelID: "Tesslate/OmniCoder-9B-GGUF:Q4_K_M", Name: "OmniCoder 9B",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "openrouter", ID: "openrouter/nvidia/nemotron-3-super-120b-a12b:free", ModelID: "nvidia/nemotron-3-super-120b-a12b:free", Name: "Nemotron 3 Super 120B A12B",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "openrouter", ID: "openrouter/stepfun/step-3.5-flash:free", ModelID: "stepfun/step-3.5-flash:free", Name: "Step-3.5 Flash",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "nvidia-nim", ID: "nvidia/nvidia/nemotron-3-super-120b-a12b", ModelID: "nvidia/nemotron-3-super-120b-a12b", Name: "Nemotron 3 Super 120B A12B",
			ContextWindow:   1000000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "nvidia-nim", ID: "nvidia/moonshotai/kimi-k2p6", ModelID: "nvidia/moonshotai/kimi-k2p6", Name: "Kimi K2.6",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(16384),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "nvidia-nim", ID: "nvidia/qwen/qwen3.5-397b-a17b", ModelID: "nvidia/qwen/qwen3.5-397b-a17b", Name: "Qwen 3.5 397B A17B",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(16384),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			ProviderID: "deepinfra", ID: "deepinfra/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B", ModelID: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B", Name: "Nemotron 3 Super 120B A12B",
			ContextWindow:   1000000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.1, OutputTokensPer1M: 0.50, CacheReadTokensPer1M: 0.04, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking: &ThinkingParams{
					Enabled: true,
					Level:   ThinkingLevelXHigh,
				},
			},
		},
		{
			// DeepSeek V4 Flash via DeepInfra. Source: models.dev (release 2026-04-24).
			// Text-only, 1M context, 384K output cap. Reasoning model — interleaves
			// chain-of-thought in the reasoning_content field per provider spec.
			ProviderID: "deepinfra", ID: "deepinfra/deepseek-ai/DeepSeek-V4-Flash", ModelID: "deepseek-ai/DeepSeek-V4-Flash", Name: "DeepSeek V4 Flash",
			ContextWindow:   1000000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.14, OutputTokensPer1M: 0.28, CacheReadTokensPer1M: 0.028}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(384000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			// DeepSeek V4 Pro via DeepInfra. Source: models.dev (release 2026-04-24).
			// Same 1M context as the V4 Flash sibling — they share the V4
			// architecture. The upstream models.dev entry reports 65536 for
			// both context and output, which is almost certainly a typo
			// (Pro context smaller than Flash is implausible). Pricing
			// stays as listed: ~12x more expensive output than Flash.
			ProviderID: "deepinfra", ID: "deepinfra/deepseek-ai/DeepSeek-V4-Pro", ModelID: "deepseek-ai/DeepSeek-V4-Pro", Name: "DeepSeek V4 Pro",
			ContextWindow:   1000000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 1.74, OutputTokensPer1M: 3.48, CacheReadTokensPer1M: 0.145}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(65536),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			// GLM 5.1 via DeepInfra (zai-org/GLM-5.1). Source: models.dev (release 2026-04-07).
			// Distinct from the opencode-openai/glm-5.1 entry above — same model
			// family, different provider/pricing path. Text-only, 200K context.
			ProviderID: "deepinfra", ID: "deepinfra/zai-org/GLM-5.1", ModelID: "zai-org/GLM-5.1", Name: "GLM 5.1 (DeepInfra)",
			ContextWindow:   202752,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 1.4, OutputTokensPer1M: 4.4, CacheReadTokensPer1M: 0.26}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(16384),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			// Gemma 4 26B A4B IT via DeepInfra. Source: models.dev (release 2026-04-02).
			// Multimodal (text + image input), no cache-read pricing on this tier.
			ProviderID: "deepinfra", ID: "deepinfra/google/gemma-4-26B-A4B-it", ModelID: "google/gemma-4-26B-A4B-it", Name: "Gemma 4 26B A4B IT",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.07, OutputTokensPer1M: 0.34}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			// Gemma 4 31B IT via DeepInfra. Source: models.dev (release 2026-04-02).
			// Slightly larger sibling of the 26B-A4B model with same modality
			// surface and context window; pricier per token.
			ProviderID: "deepinfra", ID: "deepinfra/google/gemma-4-31B-it", ModelID: "google/gemma-4-31B-it", Name: "Gemma 4 31B IT",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.13, OutputTokensPer1M: 0.38}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-openai", ID: "opencode/glm-5.1", ModelID: "glm-5.1", Name: "GLM-5.1",
			ContextWindow:   200000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-openai", ID: "opencode/glm-5", ModelID: "glm-5", Name: "GLM-5",
			ContextWindow:   200000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(128000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-openai", ID: "opencode/kimi-k2.6", ModelID: "kimi-k2.6", Name: "Kimi K2.6",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-openai", ID: "opencode/kimi-k2.5", ModelID: "kimi-k2.5", Name: "Kimi K2.5",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-openai", ID: "opencode/deepseek-v4-pro", ModelID: "deepseek-v4-pro", Name: "Deepseek V4 Pro",
			ContextWindow:   1000000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(384000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-openai", ID: "opencode/deepseek-v4-flash", ModelID: "deepseek-v4-flash", Name: "Deepseek V4 Flash",
			ContextWindow:   1000000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(384000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-openai", ID: "opencode/mimo-v2.5-pro", ModelID: "mimo-v2.5-pro", Name: "MiMo V2.5 Pro",
			ContextWindow:   1048576,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131072),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-openai", ID: "opencode/mimo-v2.5", ModelID: "mimo-v2.5", Name: "MiMo V2.5",
			ContextWindow:   1048576,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131072),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-anthropic", ID: "opencode/minimax-m2.7", ModelID: "minimax-m2.7", Name: "MiniMax M2.7",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "opencode-anthropic", ID: "opencode/minimax-m2.5", ModelID: "minimax-m2.5", Name: "MiniMax M2.5",
			ContextWindow:   204800,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "fireworks", ID: "fireworks/glm-5.1-fast", ModelID: "accounts/fireworks/routers/glm-5.1-fast", Name: "GLM-5.1 Fast",
			ContextWindow:   200000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "fireworks", ID: "fireworks/kimi-k2p6-turbo", ModelID: "accounts/fireworks/routers/kimi-k2p6-turbo", Name: "Kimi K2.6 Turbo",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "fireworks", ID: "fireworks/kimi-k2p5-turbo", ModelID: "accounts/fireworks/routers/kimi-k2p5-turbo", Name: "Kimi K2.5 Turbo",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.14, OutputTokensPer1M: 0.14, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelHigh},
			},
		},
		{
			ProviderID: "fireworks", ID: "fireworks/glm-5-fast", ModelID: "accounts/fireworks/routers/glm-5-fast", Name: "GLM-5 Fast",
			ContextWindow:   200000,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32000),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "xiaomi-openai", ID: "xiaomi/mimo-v2.5-pro", ModelID: "mimo-v2.5-pro", Name: "MiMo V2.5 Pro",
			ContextWindow:   1048576,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131072),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "xiaomi-openai", ID: "xiaomi/mimo-v2.5", ModelID: "mimo-v2.5", Name: "MiMo V2.5",
			ContextWindow:   1048576,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131072),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "xiaomi-openai", ID: "xiaomi/mimo-v2-pro", ModelID: "mimo-v2-pro", Name: "MiMo V2 Pro",
			ContextWindow:   256000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "xiaomi-openai", ID: "xiaomi/mimo-v2-omni", ModelID: "mimo-v2-omni", Name: "MiMo V2 Omni",
			ContextWindow:   256000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "xiaomi-anthropic", ID: "xiaomi-anthropic/mimo-v2.5-pro", ModelID: "mimo-v2.5-pro", Name: "MiMo V2.5 Pro (Anthropic)",
			ContextWindow:   1048576,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131072),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "xiaomi-anthropic", ID: "xiaomi-anthropic/mimo-v2.5", ModelID: "mimo-v2.5", Name: "MiMo V2.5 (Anthropic)",
			ContextWindow:   1048576,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131072),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "xiaomi-anthropic", ID: "xiaomi-anthropic/mimo-v2-pro", ModelID: "mimo-v2-pro", Name: "MiMo V2 Pro (Anthropic)",
			ContextWindow:   256000,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "xiaomi-anthropic", ID: "xiaomi-anthropic/mimo-v2-omni", ModelID: "mimo-v2-omni", Name: "MiMo V2 Omni (Anthropic)",
			ContextWindow:   256000,
			InputModalities: multimodalInput, OutputModalities: textOnly,
			Costs: &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.0, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(8192),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-anthropic", ID: "umans/umans-kimi-k2.6", ModelID: "umans-kimi-k2.6", Name: "Umans Kimi K2.6",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-anthropic", ID: "umans/umans-kimi-k2.7", ModelID: "umans-kimi-k2.7", Name: "Umans Kimi K2.7 Code",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-anthropic", ID: "umans/umans-glm-5.1", ModelID: "umans-glm-5.1", Name: "Umans GLM 5.1",
			ContextWindow:   202752,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131071),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-anthropic", ID: "umans/umans-glm-5.2", ModelID: "umans-glm-5.2", Name: "Umans GLM 5.2",
			ContextWindow:   405504,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131071),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-anthropic", ID: "umans/umans-coder", ModelID: "umans-coder", Name: "Umans Coder",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-anthropic", ID: "umans/umans-flash", ModelID: "umans-flash", Name: "Umans Flash",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-anthropic", ID: "umans/umans-qwen3.6-35b-a3b", ModelID: "umans-qwen3.6-35b-a3b", Name: "Umans Qwen3.6 35B A3B",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-openai", ID: "umans-openai/umans-kimi-k2.6", ModelID: "umans-kimi-k2.6", Name: "Umans Kimi K2.6 (OpenAI)",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-openai", ID: "umans-openai/umans-kimi-k2.7", ModelID: "umans-kimi-k2.7", Name: "Umans Kimi K2.7 Code (OpenAI)",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-openai", ID: "umans-openai/umans-glm-5.1", ModelID: "umans-glm-5.1", Name: "Umans GLM 5.1 (OpenAI)",
			ContextWindow:   202752,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131071),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-openai", ID: "umans-openai/umans-glm-5.2", ModelID: "umans-glm-5.2", Name: "Umans GLM 5.2 (OpenAI)",
			ContextWindow:   405504,
			InputModalities: textOnly, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(131071),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-openai", ID: "umans-openai/umans-coder", ModelID: "umans-coder", Name: "Umans Coder (OpenAI)",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-openai", ID: "umans-openai/umans-flash", ModelID: "umans-flash", Name: "Umans Flash (OpenAI)",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
		{
			ProviderID: "umans-openai", ID: "umans-openai/umans-qwen3.6-35b-a3b", ModelID: "umans-qwen3.6-35b-a3b", Name: "Umans Qwen3.6 35B A3B (OpenAI)",
			ContextWindow:   262144,
			InputModalities: textAndImage, OutputModalities: textOnly,
			Costs: umansPricing,
			RecommendedParams: &InferenceParams{
				MaxTokens: Ptr(32768),
				Thinking:  &ThinkingParams{Enabled: true, Level: ThinkingLevelXHigh},
			},
		},
	}

	DefaultProviders = []ProviderConfig{
		{
			ID:                 "google",
			Format:             APIFormatGoogle,
			EnvKey:             "GOOGLE_API_KEY",
			SupportsEmbeddings: true,
		},
		{
			ID:                 "google-vertex",
			Format:             APIFormatGoogle,
			EnvKey:             "VERTEX_API_KEY",
			SupportsEmbeddings: true,
			Options: map[string]string{
				"project":  os.Getenv("VERTEX_PROJECT_ID"),
				"location": os.Getenv("VERTEX_PROJECT_LOCATION"),
			},
		},
		{
			ID:                 "google-vertex-us",
			Format:             APIFormatGoogle,
			EnvKey:             "VERTEX_API_KEY",
			SupportsEmbeddings: true,
			Options: map[string]string{
				"project":  os.Getenv("VERTEX_PROJECT_ID"),
				"location": "us-central1",
			},
		},
		{
			ID:     "anthropic",
			Format: APIFormatAnthropic,
			EnvKey: "ANTHROPIC_API_KEY",
		},
		{
			ID:      "anthropic-oauth",
			Format:  APIFormatAnthropic,
			BaseURL: "https://api.anthropic.com",
			Auth:    oauthConfigFromOAuthX(oauthx.AnthropicConfig("anthropic-oauth")),
		},
		{
			ID:                 "openai",
			Format:             APIFormatOpenAI,
			EnvKey:             "OPENAI_API_KEY",
			SupportsEmbeddings: true,
		},
		{
			ID:                 "openai-responses",
			Format:             APIFormatOpenAIResponses,
			EnvKey:             "OPENAI_API_KEY",
			SupportsEmbeddings: false,
		},
		{
			ID:      "openai-codex",
			Format:  APIFormatOpenAICodex,
			BaseURL: "https://chatgpt.com/backend-api",
			Auth:    oauthConfigFromOAuthX(oauthx.CodexConfig("openai-codex")),
		},
		{
			ID:      "zenmux-anthropic",
			Format:  APIFormatAnthropic,
			EnvKey:  "ZENMUX_API_KEY",
			BaseURL: "https://zenmux.ai/api/anthropic",
		},
		{
			ID:      "zenmux-vertex",
			Format:  APIFormatGoogle,
			EnvKey:  "ZENMUX_API_KEY",
			BaseURL: "https://zenmux.ai/api/vertex-ai",
			Options: map[string]string{
				"project":     "default",
				"api_version": "v1",
			},
		},
		{
			ID:      "deepseek",
			Format:  APIFormatAnthropic,
			EnvKey:  "DEEPSEEK_API_KEY",
			BaseURL: "https://api.deepseek.com/anthropic",
		},
		{
			ID:      "moonshot",
			Format:  APIFormatAnthropic,
			EnvKey:  "MOONSHOT_API_KEY",
			BaseURL: "https://api.moonshot.ai/anthropic",
		},
		{
			ID:      "minimax",
			Format:  APIFormatAnthropic,
			EnvKey:  "MINIMAX_API_KEY",
			BaseURL: "https://api.minimax.io/anthropic",
		},
		{
			ID:      "qwen-plan",
			Format:  APIFormatAnthropic,
			EnvKey:  "QWEN_PLAN_API_KEY",
			BaseURL: "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
		},
		{
			ID:      "alibaba",
			Format:  APIFormatAnthropic,
			EnvKey:  "ALIBABA_API_KEY",
			BaseURL: "https://dashscope-intl.aliyuncs.com/apps/anthropic",
		},
		{
			ID:      "ollama-anthropic",
			Format:  APIFormatAnthropic,
			BaseURL: "http://localhost:11434",
		},
		{
			ID:      "ollama",
			Format:  APIFormatOpenAI,
			BaseURL: "http://localhost:11434/v1",
		},
		{
			ID:      "llamacpp-anthropic",
			Format:  APIFormatAnthropic,
			BaseURL: "http://beebox.local:8080",
		},
		{
			ID:      "llamacpp",
			Format:  APIFormatOpenAI,
			BaseURL: "http://beebox.local:8080/v1",
		},
		{
			ID:                 "openrouter",
			Format:             APIFormatOpenAI,
			EnvKey:             "OPENROUTER_API_KEY",
			BaseURL:            "https://openrouter.ai/api/v1",
			SupportsEmbeddings: true,
		},
		{
			ID:      "nvidia-nim",
			Format:  APIFormatOpenAI,
			EnvKey:  "NVIDIA_NIM_API_KEY",
			BaseURL: "https://integrate.api.nvidia.com/v1",
		},
		{
			ID:      "deepinfra",
			Format:  APIFormatOpenAI,
			EnvKey:  "DEEPINFRA_API_KEY",
			BaseURL: "https://api.deepinfra.com/v1",
		},
		{
			ID:      "opencode-openai",
			Format:  APIFormatOpenAI,
			EnvKey:  "OPENCODE_API_KEY",
			BaseURL: "https://opencode.ai/zen/go/v1",
		},
		{
			ID:      "opencode-anthropic",
			Format:  APIFormatAnthropic,
			EnvKey:  "OPENCODE_API_KEY",
			BaseURL: "https://opencode.ai/zen/go",
		},
		{
			ID:      "fireworks-openai",
			Format:  APIFormatOpenAI,
			EnvKey:  "FIREWORKS_API_KEY",
			BaseURL: "https://api.fireworks.ai/inference/v1",
		},
		{
			ID:      "fireworks",
			Format:  APIFormatAnthropic,
			EnvKey:  "FIREWORKS_API_KEY",
			BaseURL: "https://api.fireworks.ai/inference",
		},
		{
			ID:      "xiaomi-openai",
			Format:  APIFormatOpenAI,
			EnvKey:  "XIAOMI_API_KEY",
			BaseURL: "https://token-plan-ams.xiaomimimo.com/v1",
		},
		{
			ID:      "xiaomi-anthropic",
			Format:  APIFormatAnthropic,
			EnvKey:  "XIAOMI_API_KEY",
			BaseURL: "https://token-plan-ams.xiaomimimo.com/anthropic",
		},
		{
			ID:      "umans-anthropic",
			Format:  APIFormatAnthropic,
			EnvKey:  "UMANS_API_KEY",
			BaseURL: "https://api.code.umans.ai",
		},
		{
			ID:      "umans-openai",
			Format:  APIFormatOpenAI,
			EnvKey:  "UMANS_API_KEY",
			BaseURL: "https://api.code.umans.ai/v1",
		},
	}

	DefaultEmbeddingModels = []EmbeddingModelConfig{
		{
			ID:              "google/gemini-embedding-001",
			ProviderID:      "google",
			ModelID:         "gemini-embedding-001",
			Dimensions:      Ptr(int32(768)),
			Name:            "Gemini Embedding 001",
			InputModalities: multimodalInput,
			Costs:           &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
		},
		{
			ID:              "google/gemini-embedding-2-preview",
			ProviderID:      "google-vertex",
			ModelID:         "gemini-embedding-2-preview",
			Dimensions:      Ptr(int32(768)),
			Name:            "Gemini Embedding 2 Preview",
			InputModalities: multimodalInput,
			Costs:           &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
		},
		{
			ID:              "ollama/qwen3-embedding:4b",
			ProviderID:      "ollama",
			ModelID:         "qwen3-embedding:4b",
			Name:            "Qwen3 Embedding 4B (Ollama)",
			Dimensions:      Ptr(int32(768)),
			InputModalities: textOnly,
			Costs:           &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
		},
		{
			ID:              "ollama/snowflake-arctic-embed2",
			ProviderID:      "ollama",
			ModelID:         "snowflake-arctic-embed2",
			Name:            "Snowflake Arctic Embed2 (Ollama)",
			Dimensions:      Ptr(int32(768)),
			InputModalities: textOnly,
			Costs:           &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
		},
		{
			ID:              "ollama/nomic-embed-text-v2-moe",
			ProviderID:      "ollama",
			ModelID:         "nomic-embed-text-v2-moe",
			Name:            "Nomic Embed Text V2 MOE (Ollama)",
			Dimensions:      Ptr(int32(768)),
			InputModalities: textOnly,
			Costs:           &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
		},
		{
			ID:              "llamacpp/jina-embeddings-v5-text-small-retrieval",
			ProviderID:      "llamacpp",
			ModelID:         "jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:F16",
			Name:            "Jina Embeddings V5 Small Retrieval (Llama.cpp)",
			Dimensions:      Ptr(int32(768)),
			InputModalities: textOnly,
			Costs:           &ModelCosts{ModelPricing: ModelPricing{InputTokensPer1M: 0.0, OutputTokensPer1M: 0.00, CacheReadTokensPer1M: 0.0, CacheWriteTokensPer1M: 0.0}},
		},
	}

	GlobalRegistry = DefaultRegistry()
)

func oauthConfigFromOAuthX(cfg oauthx.Config) OAuthConfig {
	return OAuthConfig{
		Type:         cfg.Type,
		ProviderID:   cfg.ProviderID,
		TokenURL:     cfg.TokenURL,
		AuthURL:      cfg.AuthURL,
		ClientID:     cfg.ClientID,
		ClientSecret: cfg.ClientSecret,
		Scopes:       append([]string(nil), cfg.Scopes...),
		RedirectURL:  cfg.RedirectURL,
		CacheKey:     cfg.CacheKey,
		AuthParams:   cloneOAuthParams(cfg.AuthParams),
	}
}

func cloneOAuthParams(params map[string]string) map[string]string {
	if len(params) == 0 {
		return nil
	}
	out := make(map[string]string, len(params))
	for key, value := range params {
		out[key] = value
	}
	return out
}

func DefaultRegistry() *Registry {
	return NewRegistry().
		WithModels(DefaultModels...).
		WithEmbeddingModels(DefaultEmbeddingModels...).
		WithProviders(DefaultProviders...)
}
