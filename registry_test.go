package llm

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"
)

func findDefaultModel(t *testing.T, id string) ModelConfig {
	t.Helper()
	for _, model := range DefaultModels {
		if model.ID == id {
			return model
		}
	}
	t.Fatalf("default model %q not found", id)
	return ModelConfig{}
}

type registryTestClient struct{}

func (c *registryTestClient) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	return &Message{Role: MessageRoleAssistant}, nil
}

func (c *registryTestClient) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	return nil, nil
}

func (c *registryTestClient) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	return &EmbeddingResponse{Model: req.Model, Embeddings: []float32{42}}, nil
}

type registryUsageEmbeddingClient struct{}

func (c *registryUsageEmbeddingClient) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	return &Message{Role: MessageRoleAssistant}, nil
}

func (c *registryUsageEmbeddingClient) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	return nil, nil
}

func (c *registryUsageEmbeddingClient) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	return &EmbeddingResponse{
		Model:      req.Model,
		Embeddings: []float32{42},
		Usage: NewUsage(UsageOperationEmbedding, TokenUsage{
			InputTokens: 1000,
			TotalTokens: 1000,
		}),
	}, nil
}

func TestResolveEmbedding(t *testing.T) {
	registry := NewRegistry().
		WithProviders(ProviderConfig{
			ID:                 "test-provider",
			Format:             APIFormatOpenAI,
			SupportsEmbeddings: true,
		}).
		WithEmbeddingModels(EmbeddingModelConfig{
			ID:         "test-embedding",
			Name:       "Test Embedding",
			ProviderID: "test-provider",
			ModelID:    "embedding-1",
		}).
		WithFactory(APIFormatOpenAI, func(ctx context.Context, config ProviderConfig) (Client, error) {
			return &registryTestClient{}, nil
		})

	client, model, err := registry.ResolveEmbedding(context.Background(), "test-embedding")
	if err != nil {
		t.Fatalf("ResolveEmbedding returned error: %v", err)
	}
	if model.ModelID != "embedding-1" {
		t.Fatalf("unexpected model id: %q", model.ModelID)
	}

	resp, err := client.CreateEmbeddings(context.Background(), model.TextRequest("hello"))
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %v", err)
	}
	if got := resp.Embeddings; len(got) != 1 || got[0] != 42 {
		t.Fatalf("unexpected embedding: %#v", got)
	}
}

func TestResolverCreateEmbeddingsAttributesUsage(t *testing.T) {
	registry := NewRegistry().
		WithProviders(ProviderConfig{
			ID:                 "test-provider",
			Format:             APIFormatOpenAI,
			SupportsEmbeddings: true,
		}).
		WithEmbeddingModels(EmbeddingModelConfig{
			ID:         "internal-embedding",
			Name:       "Internal Embedding",
			ProviderID: "test-provider",
			ModelID:    "provider-embedding",
			Costs: &ModelCosts{ModelPricing: ModelPricing{
				InputTokensPer1M: 2,
			}},
		}).
		WithFactory(APIFormatOpenAI, func(ctx context.Context, config ProviderConfig) (Client, error) {
			return &registryUsageEmbeddingClient{}, nil
		})

	resp, err := registry.CreateEmbeddings(context.Background(), NewTextEmbeddingRequest("internal-embedding", "hello"))
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %v", err)
	}
	if resp.Model != "provider-embedding" {
		t.Fatalf("expected provider model id on response, got %q", resp.Model)
	}
	if resp.Usage == nil || len(resp.Usage.Entries) != 1 {
		t.Fatalf("expected attributed embedding usage, got %#v", resp.Usage)
	}
	entry := resp.Usage.Entries[0]
	if entry.ModelID != "internal-embedding" || entry.ProviderModelID != "provider-embedding" || entry.ProviderID != "test-provider" {
		t.Fatalf("unexpected model attribution: %#v", entry)
	}
	if got, want := resp.Usage.Cost, 0.002; math.Abs(got-want) > 1e-9 {
		t.Fatalf("unexpected cost: got %f want %f", got, want)
	}
}

func TestResolveEmbeddingRequiresProviderSupport(t *testing.T) {
	registry := NewRegistry().
		WithProviders(ProviderConfig{
			ID:     "test-provider",
			Format: APIFormatOpenAI,
		}).
		WithEmbeddingModels(EmbeddingModelConfig{
			ID:         "test-embedding",
			Name:       "Test Embedding",
			ProviderID: "test-provider",
			ModelID:    "embedding-1",
		}).
		WithFactory(APIFormatOpenAI, func(ctx context.Context, config ProviderConfig) (Client, error) {
			return &registryTestClient{}, nil
		})

	if _, _, err := registry.ResolveEmbedding(context.Background(), "test-embedding"); err == nil {
		t.Fatal("expected ResolveEmbedding to fail when provider support is not declared")
	}
}

func TestDefaultRegistryIncludesDefaultEmbeddingModels(t *testing.T) {
	registry := DefaultRegistry()
	model, ok := registry.GetEmbeddingModel("google/gemini-embedding-001")
	if !ok {
		t.Fatal("expected default registry to include default embedding models")
	}
	if model.ProviderID != "google" {
		t.Fatalf("expected default Gemini API embedding model to use google provider, got %q", model.ProviderID)
	}
	vertexModel, ok := registry.GetEmbeddingModel("google/gemini-embedding-2-preview")
	if !ok {
		t.Fatal("expected default registry to include vertex Gemini embedding preview model")
	}
	if !strings.HasPrefix(vertexModel.ProviderID, "google-vertex") {
		t.Fatalf("expected preview Gemini embedding model to use a google-vertex provider, got %q", vertexModel.ProviderID)
	}
}

func TestDefaultFireworksKimi25UsesInternalIDForAccountingAndProviderIDForTransport(t *testing.T) {
	model := findDefaultModel(t, "fireworks/kimi-k2p6-turbo")
	if model.ProviderID != "fireworks" {
		t.Fatalf("unexpected provider id: %q", model.ProviderID)
	}
	if model.ModelID != "accounts/fireworks/routers/kimi-k2p6-turbo" {
		t.Fatalf("unexpected provider model id: %q", model.ModelID)
	}

	attributed, err := model.AttributeUsage(NewUsage(UsageOperationCompletion, TokenUsage{
		InputTokens:  10,
		OutputTokens: 5,
		TotalTokens:  15,
	}))
	if err != nil {
		t.Fatalf("AttributeUsage returned error: %v", err)
	}
	entry := attributed.Entries[0]
	if entry.ModelID != "fireworks/kimi-k2p6-turbo" {
		t.Fatalf("expected internal model id as accounting key, got %q", entry.ModelID)
	}
	if entry.ProviderModelID != "accounts/fireworks/routers/kimi-k2p6-turbo" {
		t.Fatalf("expected provider transport model id, got %q", entry.ProviderModelID)
	}
	if _, ok := attributed.ByModel["fireworks/kimi-k2p6-turbo"]; !ok {
		t.Fatalf("expected by-model summary keyed by internal id, got %#v", attributed.ByModel)
	}
}

func TestModelConfigCalculateCostUsesModalitySpecificPricing(t *testing.T) {
	model := ModelConfig{
		Costs: &ModelCosts{
			ModelPricing: ModelPricing{
				InputTokensPer1M:      1,
				OutputTokensPer1M:     2,
				CacheReadTokensPer1M:  0.5,
				CacheWriteTokensPer1M: 0.75,
				InputCostByModality: &ModalityPricing{
					ImageTokensPer1M:    Ptr(10.0),
					DocumentTokensPer1M: Ptr(4.0),
				},
				OutputCostByModality: &ModalityPricing{
					AudioTokensPer1M: Ptr(20.0),
				},
				CacheReadCostByModality: &ModalityPricing{
					ImageTokensPer1M: Ptr(5.0),
				},
				CacheWriteCostByModality: &ModalityPricing{
					DocumentTokensPer1M: Ptr(6.0),
				},
			},
		},
	}

	usage := TokenUsage{
		InputTokens:              100,
		OutputTokens:             40,
		CacheReadInputTokens:     10,
		CacheCreationInputTokens: 8,
		ToolUseInputTokens:       8,
		InputTokensDetails: &UsageTokenDetails{
			TextTokens:  52,
			ImageTokens: 30,
		},
		OutputTokensDetails: &UsageTokenDetails{
			AudioTokens: 10,
		},
		CacheReadInputTokensDetails: &UsageTokenDetails{
			ImageTokens: 4,
		},
		CacheCreationInputTokensDetails: &UsageTokenDetails{
			DocumentTokens: 3,
		},
		ToolUseInputTokensDetails: &UsageTokenDetails{
			DocumentTokens: 3,
		},
	}

	got := model.CalculateCost(usage)
	want := (52*1.0 + 26*10.0 + 4*1.0 + 3*4.0 + 5*1.0 + 10*20.0 + 30*2.0 + 4*5.0 + 6*0.5 + 3*6.0 + 5*0.75) / 1_000_000
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("unexpected calculated cost: got %f want %f", got, want)
	}
}
func TestEmbeddingModelConfigCalculateCost(t *testing.T) {
	model := EmbeddingModelConfig{
		Costs: &ModelCosts{
			ModelPricing: ModelPricing{
				InputTokensPer1M: 1.0,
				InputCostByModality: &ModalityPricing{
					ImageTokensPer1M: Ptr(5.0),
				},
			},
		},
	}

	usage := TokenUsage{
		InputTokens: 1000,
		InputTokensDetails: &UsageTokenDetails{
			TextTokens:  800,
			ImageTokens: 200,
		},
	}

	got := model.CalculateCost(usage)
	want := (800*1.0 + 200*5.0) / 1_000_000
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("unexpected calculated cost: got %f want %f", got, want)
	}
}

func TestDefaultGemini3FlashPricingUsesAudioSpecificRates(t *testing.T) {
	model := findDefaultModel(t, "google/gemini-3-flash-preview")
	usage := TokenUsage{
		InputTokens:          1000,
		OutputTokens:         100,
		CacheReadInputTokens: 200,
		InputTokensDetails: &UsageTokenDetails{
			TextTokens:  600,
			AudioTokens: 400,
		},
		OutputTokensDetails: &UsageTokenDetails{
			TextTokens: 100,
		},
		CacheReadInputTokensDetails: &UsageTokenDetails{
			TextTokens:  100,
			AudioTokens: 100,
		},
	}

	got := model.CalculateCost(usage)
	want := (500*0.5 + 300*1.0 + 100*3.0 + 100*0.05 + 100*0.1) / 1_000_000
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("unexpected calculated cost: got %f want %f", got, want)
	}
}

func TestDefaultGemini31ProPricingUsesLongContextTierForCache(t *testing.T) {
	model := findDefaultModel(t, "google/gemini-3.1-pro-preview")
	usage := TokenUsage{
		InputTokens:          250000,
		OutputTokens:         1000,
		CacheReadInputTokens: 50000,
		InputTokensDetails: &UsageTokenDetails{
			TextTokens: 250000,
		},
		OutputTokensDetails: &UsageTokenDetails{
			TextTokens: 1000,
		},
		CacheReadInputTokensDetails: &UsageTokenDetails{
			TextTokens: 50000,
		},
	}

	got := model.CalculateCost(usage)
	want := (200000*4.0 + 1000*18.0 + 50000*0.4) / 1_000_000
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("unexpected calculated cost: got %f want %f", got, want)
	}
}

func TestModelConfigCalculateCostUsesTierSpecificModalityCachePricing(t *testing.T) {
	model := ModelConfig{
		Costs: &ModelCosts{
			ModelPricing: ModelPricing{
				InputTokensPer1M:      1.0,
				OutputTokensPer1M:     2.0,
				CacheReadTokensPer1M:  0.5,
				CacheWriteTokensPer1M: 0.75,
				InputCostByModality: &ModalityPricing{
					AudioTokensPer1M: Ptr(3.0),
				},
				CacheReadCostByModality: &ModalityPricing{
					AudioTokensPer1M: Ptr(1.5),
				},
			},
			Tiers: []PriceTier{
				{
					Boundary: 200000,
					ModelPricing: ModelPricing{
						InputTokensPer1M:     4.0,
						OutputTokensPer1M:    8.0,
						CacheReadTokensPer1M: 0.4,
						InputCostByModality: &ModalityPricing{
							AudioTokensPer1M: Ptr(6.0),
						},
						CacheReadCostByModality: &ModalityPricing{
							AudioTokensPer1M: Ptr(0.1),
						},
					},
				},
			},
		},
	}

	usage := TokenUsage{
		InputTokens:          150000,
		OutputTokens:         1000,
		CacheReadInputTokens: 20000,
		InputTokensDetails: &UsageTokenDetails{
			TextTokens:  90000,
			AudioTokens: 60000,
		},
		OutputTokensDetails: &UsageTokenDetails{
			TextTokens: 1000,
		},
		CacheReadInputTokensDetails: &UsageTokenDetails{
			AudioTokens: 20000,
		},
	}

	got := model.CalculateCost(usage)
	want := (90000*4.0 + 40000*6.0 + 1000*8.0 + 20000*0.1) / 1_000_000
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("unexpected calculated cost: got %f want %f", got, want)
	}
}

func TestRegisterProviderInvalidatesCachedClient(t *testing.T) {
	registry := NewRegistry().WithFactory(APIFormatOpenAI, func(ctx context.Context, config ProviderConfig) (Client, error) {
		return &factoryTaggedClient{tag: config.BaseURL}, nil
	})
	registry.RegisterProvider("test-provider", ProviderConfig{
		ID:      "test-provider",
		Format:  APIFormatOpenAI,
		BaseURL: "https://first.example",
	})

	client, _, err := registry.Resolve(context.Background(), ModelConfig{
		ID:         "test-model",
		ModelID:    "model-1",
		ProviderID: "test-provider",
	})
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	first, ok := unwrapTaggedClient(client)
	if !ok || first.tag != "https://first.example" {
		t.Fatalf("unexpected first client: %#v", client)
	}

	registry.RegisterProvider("test-provider", ProviderConfig{
		ID:      "test-provider",
		Format:  APIFormatOpenAI,
		BaseURL: "https://second.example",
	})

	client, _, err = registry.Resolve(context.Background(), ModelConfig{
		ID:         "test-model",
		ModelID:    "model-1",
		ProviderID: "test-provider",
	})
	if err != nil {
		t.Fatalf("Resolve returned error after provider update: %v", err)
	}
	second, ok := unwrapTaggedClient(client)
	if !ok || second.tag != "https://second.example" {
		t.Fatalf("expected cache invalidation after provider update, got %#v", client)
	}
}

func TestWithRetryPolicyInvalidatesCachedClients(t *testing.T) {
	registry := NewRegistry().
		WithProviders(ProviderConfig{
			ID:     "test-provider",
			Format: APIFormatOpenAI,
		}).
		WithFactory(APIFormatOpenAI, func(ctx context.Context, config ProviderConfig) (Client, error) {
			return &registryTestClient{}, nil
		})

	client, _, err := registry.Resolve(context.Background(), ModelConfig{
		ID:         "test-model",
		ModelID:    "model-1",
		ProviderID: "test-provider",
	})
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if !hasRetryWrapper(client) {
		t.Fatalf("expected default resolution to wrap client with retry, got %T", client)
	}

	registry.WithRetryPolicy(NoRetryPolicy())

	client, _, err = registry.Resolve(context.Background(), ModelConfig{
		ID:         "test-model",
		ModelID:    "model-1",
		ProviderID: "test-provider",
	})
	if err != nil {
		t.Fatalf("Resolve returned error after retry policy change: %v", err)
	}
	if hasRetryWrapper(client) {
		t.Fatalf("expected no retry wrapper after policy change, got %T", client)
	}
	if !hasDirectRegistryTestClient(client) {
		t.Fatalf("expected cache invalidation after retry policy update, got %T", client)
	}
}

func hasRetryWrapper(client Client) bool {
	if rl, ok := client.(*ClientWithRateLimit); ok {
		_, hasRetry := rl.client.(*ClientWithRetry)
		return hasRetry
	}
	_, ok := client.(*ClientWithRetry)
	return ok
}

func hasDirectRegistryTestClient(client Client) bool {
	if rl, ok := client.(*ClientWithRateLimit); ok {
		_, isDirect := rl.client.(*registryTestClient)
		return isDirect
	}
	_, ok := client.(*registryTestClient)
	return ok
}

func TestResolverProviderRateLimitSettingsInvalidateClients(t *testing.T) {
	registry := NewRegistry().
		WithFactory(APIFormatOpenAI, func(ctx context.Context, cfg ProviderConfig) (Client, error) {
			return &registryTestClient{}, nil
		}).
		WithProviders(ProviderConfig{ID: "test-provider", Format: APIFormatOpenAI})

	client, _, err := registry.Resolve(context.Background(), ModelConfig{ID: "m", ModelID: "m", ProviderID: "test-provider"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := client.(*ClientWithRateLimit); !ok {
		t.Fatalf("expected default rate-limited client, got %T", client)
	}
	registry.WithProviderRateLimits(0, 0).WithProviderCircuitBreaker(0, time.Hour)
	client, _, err = registry.Resolve(context.Background(), ModelConfig{ID: "m", ModelID: "m", ProviderID: "test-provider"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := client.(*ClientWithRateLimit); ok {
		t.Fatalf("expected direct/retry client after disabling rate limits, got %T", client)
	}
}

func TestListMethodsReturnStableSortedOrder(t *testing.T) {
	registry := NewRegistry().
		WithProviders(
			ProviderConfig{ID: "z-provider", Format: APIFormatOpenAI},
			ProviderConfig{ID: "a-provider", Format: APIFormatAnthropic, SupportsEmbeddings: true},
		).
		WithModels(
			ModelConfig{ID: "z-model", ProviderID: "z-provider"},
			ModelConfig{ID: "a-model", ProviderID: "a-provider"},
		).
		WithEmbeddingModels(
			EmbeddingModelConfig{ID: "z-embedding", ProviderID: "z-provider"},
			EmbeddingModelConfig{ID: "a-embedding", ProviderID: "a-provider"},
		)

	providers := registry.ListProviders()
	if len(providers) != 2 || providers[0].ID != "a-provider" || providers[1].ID != "z-provider" {
		t.Fatalf("expected sorted providers, got %#v", providers)
	}

	models := registry.ListModels()
	if len(models) != 2 || models[0].ID != "a-model" || models[1].ID != "z-model" {
		t.Fatalf("expected sorted models, got %#v", models)
	}

	embeddingModels := registry.ListEmbeddingModels()
	if len(embeddingModels) != 2 || embeddingModels[0].ID != "a-embedding" || embeddingModels[1].ID != "z-embedding" {
		t.Fatalf("expected sorted embedding models, got %#v", embeddingModels)
	}

	embeddingProviders := registry.ListEmbeddingProviders()
	if len(embeddingProviders) != 1 || embeddingProviders[0].ID != "a-provider" {
		t.Fatalf("expected sorted embedding providers, got %#v", embeddingProviders)
	}
}

type factoryTaggedClient struct {
	tag string
}

func (c *factoryTaggedClient) CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error) {
	return &Message{Role: MessageRoleAssistant, Metadata: map[string]any{"tag": c.tag}}, nil
}

func (c *factoryTaggedClient) CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error) {
	return nil, nil
}

func unwrapTaggedClient(client Client) (*factoryTaggedClient, bool) {
	if direct, ok := client.(*factoryTaggedClient); ok {
		return direct, true
	}
	if wrapped, ok := client.(*ClientWithRateLimit); ok {
		tagged, ok := unwrapTaggedClient(wrapped.client)
		return tagged, ok
	}
	if wrapped, ok := client.(*ClientWithRetry); ok {
		tagged, ok := wrapped.client.(*factoryTaggedClient)
		return tagged, ok
	}
	return nil, false
}

func TestModelConfigCalculateCostCacheBilledSeparately(t *testing.T) {
	model := ModelConfig{
		Costs: &ModelCosts{
			ModelPricing: ModelPricing{
				InputTokensPer1M:      1.0,
				OutputTokensPer1M:     2.0,
				CacheReadTokensPer1M:  0.5,
				CacheWriteTokensPer1M: 0.75,
			},
		},
	}

	// Default (CacheBilledSeparately=false): cache is INCLUDED in InputTokens,
	// so uncached prompt = InputTokens - cache tokens.
	includedUsage := TokenUsage{
		InputTokens:              1000,
		OutputTokens:             100,
		CacheReadInputTokens:     200,
		CacheCreationInputTokens: 100,
	}
	gotIncluded := model.CalculateCost(includedUsage)
	wantIncluded := (700*1.0 + 100*2.0 + 200*0.5 + 100*0.75) / 1_000_000
	if math.Abs(gotIncluded-wantIncluded) > 1e-9 {
		t.Fatalf("default (cache included): got %f want %f", gotIncluded, wantIncluded)
	}

	// CacheBilledSeparately=true: cache is NOT included in InputTokens,
	// so uncached prompt = InputTokens (no subtraction).
	separateUsage := TokenUsage{
		InputTokens:              1000,
		OutputTokens:             100,
		CacheReadInputTokens:     200,
		CacheCreationInputTokens: 100,
		CacheBilledSeparately:    true,
	}
	gotSeparate := model.CalculateCost(separateUsage)
	wantSeparate := (1000*1.0 + 100*2.0 + 200*0.5 + 100*0.75) / 1_000_000
	if math.Abs(gotSeparate-wantSeparate) > 1e-9 {
		t.Fatalf("separate billing: got %f want %f", gotSeparate, wantSeparate)
	}
}

func TestTokenUsageNormalizedDefaultsCacheBilledSeparatelyToFalse(t *testing.T) {
	usage := TokenUsage{
		InputTokens:          100,
		OutputTokens:         50,
		CacheReadInputTokens: 20,
	}
	normalized := usage.normalized()
	if normalized.CacheBilledSeparately {
		t.Fatal("expected default CacheBilledSeparately=false")
	}
	if normalized.TotalTokens != 150 {
		t.Fatalf("expected TotalTokens=150, got %d", normalized.TotalTokens)
	}
}

func TestTokenUsageNormalizedIncludesCacheInTotalWhenBilledSeparately(t *testing.T) {
	usage := TokenUsage{
		InputTokens:           100,
		OutputTokens:          50,
		CacheReadInputTokens:  20,
		CacheBilledSeparately: true,
	}
	normalized := usage.normalized()
	if !normalized.CacheBilledSeparately {
		t.Fatal("expected CacheBilledSeparately=true preserved")
	}
	if normalized.TotalTokens != 170 {
		t.Fatalf("expected TotalTokens=170 (100+50+20), got %d", normalized.TotalTokens)
	}
}

func TestModelConfigCalculateCostUsesServiceTierPricing(t *testing.T) {
	// Standard model (not gpt-5.5)
	modelStandard := ModelConfig{
		ModelID: "gpt-4o",
		Costs: &ModelCosts{
			ModelPricing: ModelPricing{
				InputTokensPer1M:  10.0,
				OutputTokensPer1M: 30.0,
			},
		},
	}

	// GPT-5.5 model
	modelGPT55 := ModelConfig{
		ModelID: "gpt-5.5",
		Costs: &ModelCosts{
			ModelPricing: ModelPricing{
				InputTokensPer1M:  10.0,
				OutputTokensPer1M: 30.0,
			},
		},
	}

	baseUsage := TokenUsage{
		InputTokens:  1000,
		OutputTokens: 2000,
	}

	baseCost := (1000*10.0 + 2000*30.0) / 1_000_000 // 0.07

	testCases := []struct {
		name        string
		model       ModelConfig
		serviceTier string
		wantMult    float64
	}{
		{
			name:        "Standard model with default (empty) service tier",
			model:       modelStandard,
			serviceTier: "",
			wantMult:    1.0,
		},
		{
			name:        "Standard model with default service tier (other)",
			model:       modelStandard,
			serviceTier: "default",
			wantMult:    1.0,
		},
		{
			name:        "Standard model with flex service tier",
			model:       modelStandard,
			serviceTier: "flex",
			wantMult:    0.5,
		},
		{
			name:        "Standard model with priority service tier",
			model:       modelStandard,
			serviceTier: "priority",
			wantMult:    2.0,
		},
		{
			name:        "GPT-5.5 with default service tier",
			model:       modelGPT55,
			serviceTier: "",
			wantMult:    1.0,
		},
		{
			name:        "GPT-5.5 with flex service tier",
			model:       modelGPT55,
			serviceTier: "flex",
			wantMult:    0.5,
		},
		{
			name:        "GPT-5.5 with priority service tier",
			model:       modelGPT55,
			serviceTier: "priority",
			wantMult:    2.5,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			usage := baseUsage
			usage.ServiceTier = tc.serviceTier
			got := tc.model.CalculateCost(usage)
			want := baseCost * tc.wantMult
			if math.Abs(got-want) > 1e-9 {
				t.Errorf("expected cost %f, got %f (multiplier: %f)", want, got, tc.wantMult)
			}
		})
	}
}
