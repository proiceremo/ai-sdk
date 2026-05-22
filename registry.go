package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"
)

type ProviderConfig struct {
	ID                 string            `json:"id"`
	Format             APIFormat         `json:"format"`
	EnvKey             string            `json:"env_key,omitempty"`
	BaseURL            string            `json:"base_url,omitempty"`
	Auth               OAuthConfig       `json:"auth,omitempty"`
	SupportsEmbeddings bool              `json:"supports_embeddings,omitempty"`
	Options            map[string]string `json:"options,omitempty"`
}

type OAuthConfig struct {
	Type         string            `json:"type,omitempty"`
	ProviderID   string            `json:"provider_id,omitempty"`
	TokenURL     string            `json:"token_url,omitempty"`
	AuthURL      string            `json:"auth_url,omitempty"`
	ClientID     string            `json:"client_id,omitempty"`
	ClientSecret string            `json:"client_secret,omitempty"`
	Scopes       []string          `json:"scopes,omitempty"`
	RedirectURL  string            `json:"redirect_url,omitempty"`
	CacheKey     string            `json:"cache_key,omitempty"`
	AuthParams   map[string]string `json:"auth_params,omitempty"`
}

type ProviderFactory func(ctx context.Context, config ProviderConfig) (Client, error)

type Modality string

const (
	ModalityText  Modality = "text"
	ModalityAudio Modality = "audio"
	ModalityImage Modality = "image"
	ModalityVideo Modality = "video"
)

type ModelPricing struct {
	InputTokensPer1M      float64 `json:"input_tokens_per_1m"`
	OutputTokensPer1M     float64 `json:"output_tokens_per_1m"`
	CacheWriteTokensPer1M float64 `json:"cache_write_tokens_per_1m,omitempty"`
	CacheReadTokensPer1M  float64 `json:"cache_read_tokens_per_1m,omitempty"`

	InputCostByModality      *ModalityPricing `json:"input_cost_by_modality,omitempty"`
	OutputCostByModality     *ModalityPricing `json:"output_cost_by_modality,omitempty"`
	CacheWriteCostByModality *ModalityPricing `json:"cache_write_cost_by_modality,omitempty"`
	CacheReadCostByModality  *ModalityPricing `json:"cache_read_cost_by_modality,omitempty"`
}

type ModalityPricing struct {
	TextTokensPer1M     *float64 `json:"text_tokens_per_1m,omitempty"`
	AudioTokensPer1M    *float64 `json:"audio_tokens_per_1m,omitempty"`
	ImageTokensPer1M    *float64 `json:"image_tokens_per_1m,omitempty"`
	VideoTokensPer1M    *float64 `json:"video_tokens_per_1m,omitempty"`
	DocumentTokensPer1M *float64 `json:"document_tokens_per_1m,omitempty"`
}

func (p *ModalityPricing) priceFor(kind string, fallback float64) float64 {
	if p == nil {
		return fallback
	}

	var price *float64
	switch kind {
	case "text":
		price = p.TextTokensPer1M
	case "audio":
		price = p.AudioTokensPer1M
	case "image":
		price = p.ImageTokensPer1M
	case "video":
		price = p.VideoTokensPer1M
	case "document":
		price = p.DocumentTokensPer1M
	}
	if price == nil {
		return fallback
	}
	return *price
}

type PriceTier struct {
	ModelPricing
	Boundary int `json:"boundary"`
}

type ModelCosts struct {
	ModelPricing
	Tiers []PriceTier `json:"tiers,omitempty"`
}

type ModelConfig struct {
	ProviderID       string      `json:"provider_id"`
	ID               string      `json:"id"`
	ModelID          string      `json:"model_id"`
	Name             string      `json:"name"`
	ContextWindow    int         `json:"context_window"`
	InputModalities  []Modality  `json:"input_modalities"`
	OutputModalities []Modality  `json:"output_modalities"`
	Costs            *ModelCosts `json:"costs,omitempty"`

	RecommendedParams *InferenceParams `json:"recommended_params,omitempty"`
}

type EmbeddingModelConfig struct {
	ProviderID      string      `json:"provider_id"`
	ID              string      `json:"id"`
	ModelID         string      `json:"model_id"`
	Name            string      `json:"name"`
	InputModalities []Modality  `json:"input_modalities,omitempty"`
	TaskType        string      `json:"task_type,omitempty"`
	Dimensions      *int32      `json:"dimensions,omitempty"`
	Costs           *ModelCosts `json:"costs,omitempty"`
}

func (m EmbeddingModelConfig) CalculateCost(tokens TokenUsage) float64 {
	if m.Costs == nil {
		return 0
	}

	return calculateModelCost(m.Costs, tokens)
}

func (m EmbeddingModelConfig) AttributeUsage(usage *Usage) (*Usage, error) {
	return attributeUsageWithModel(usage, UsageOperationEmbedding, m.ProviderID, m.ID, m.ModelID, m.CalculateCost)
}

type Catalog struct {
	Providers       []ProviderConfig       `json:"providers"`
	Models          []ModelConfig          `json:"models"`
	EmbeddingModels []EmbeddingModelConfig `json:"embedding_models,omitempty"`
}

type RegistryConfig = Catalog

func getServiceTierCostMultiplier(modelID, serviceTier string) float64 {
	switch serviceTier {
	case "flex":
		return 0.5
	case "priority":
		if modelID == "gpt-5.5" {
			return 2.5
		}
		return 2.0
	default:
		return 1.0
	}
}

func (m ModelConfig) CalculateCost(tokens TokenUsage) float64 {
	if m.Costs == nil {
		return 0
	}
	cost := calculateModelCost(m.Costs, tokens)
	if tokens.ServiceTier != "" {
		cost *= getServiceTierCostMultiplier(m.ModelID, tokens.ServiceTier)
	}
	return cost
}

func (m ModelConfig) AttributeUsage(usage *Usage) (*Usage, error) {
	return attributeUsageWithModel(usage, UsageOperationCompletion, m.ProviderID, m.ID, m.ModelID, m.CalculateCost)
}

func calculateModelCost(costs *ModelCosts, tokens TokenUsage) float64 {
	tokens = tokens.normalized()
	pricing := activeModelPricing(costs, tokens.InputTokens)

	var uncachedPromptTokens int
	var uncachedPromptDetails *UsageTokenDetails
	if tokens.CacheBilledSeparately {
		// Cache is NOT included in InputTokens; bill it separately.
		uncachedPromptTokens = tokens.InputTokens
		uncachedPromptDetails = tokens.InputTokensDetails
	} else {
		// Default: cache IS included in InputTokens; subtract to avoid double-counting.
		uncachedPromptTokens = max(tokens.InputTokens-tokens.CacheReadInputTokens-tokens.CacheCreationInputTokens, 0)
		uncachedPromptDetails = subtractUsageTokenDetails(
			tokens.InputTokensDetails,
			tokens.CacheReadInputTokensDetails,
			tokens.CacheCreationInputTokensDetails,
		)
	}

	var cost float64
	cost += calculateBucketCost(uncachedPromptTokens, uncachedPromptDetails, pricing.InputTokensPer1M, pricing.InputCostByModality)
	cost += calculateBucketCost(tokens.ToolUseInputTokens, tokens.ToolUseInputTokensDetails, pricing.InputTokensPer1M, pricing.InputCostByModality)
	cost += calculateBucketCost(tokens.OutputTokens, tokens.OutputTokensDetails, pricing.OutputTokensPer1M, pricing.OutputCostByModality)
	cost += calculateBucketCost(tokens.CacheCreationInputTokens, tokens.CacheCreationInputTokensDetails, pricing.CacheWriteTokensPer1M, pricing.CacheWriteCostByModality)
	cost += calculateBucketCost(tokens.CacheReadInputTokens, tokens.CacheReadInputTokensDetails, pricing.CacheReadTokensPer1M, pricing.CacheReadCostByModality)

	return cost
}

func attributeUsageWithModel(usage *Usage, defaultOperation UsageOperation, providerID string, modelID string, providerModelID string, calculateCost func(TokenUsage) float64) (*Usage, error) {
	if usage == nil {
		return nil, nil
	}
	if providerID == "" || modelID == "" || providerModelID == "" {
		return nil, fmt.Errorf("usage attribution requires provider id, internal model id, and provider model id")
	}
	out := &Usage{}
	entries := usage.Entries
	if len(entries) == 0 {
		entries = []UsageEntry{{
			Operation: defaultOperation,
			Tokens:    usage.Totals,
		}}
	}
	for _, entry := range entries {
		if entry.Operation == "" {
			entry.Operation = defaultOperation
		}
		entry.ProviderID = providerID
		entry.ModelID = modelID
		entry.ProviderModelID = providerModelID
		entry.Cost = calculateCost(entry.Tokens)
		out.AddEntry(entry)
	}
	return out, nil
}

func activeModelPricing(costs *ModelCosts, promptTokens int) ModelPricing {
	if costs == nil {
		return ModelPricing{}
	}

	pricing := costs.ModelPricing
	if len(costs.Tiers) == 0 {
		return pricing
	}

	for _, tier := range costs.Tiers {
		if tier.Boundary > 0 && promptTokens > tier.Boundary {
			continue
		}
		pricing.InputTokensPer1M = tier.InputTokensPer1M
		pricing.OutputTokensPer1M = tier.OutputTokensPer1M
		if tier.InputCostByModality != nil {
			pricing.InputCostByModality = tier.InputCostByModality
		}
		if tier.OutputCostByModality != nil {
			pricing.OutputCostByModality = tier.OutputCostByModality
		}
		if tier.CacheWriteTokensPer1M > 0 || tier.CacheWriteCostByModality != nil {
			pricing.CacheWriteTokensPer1M = tier.CacheWriteTokensPer1M
		}
		if tier.CacheReadTokensPer1M > 0 || tier.CacheReadCostByModality != nil {
			pricing.CacheReadTokensPer1M = tier.CacheReadTokensPer1M
		}
		if tier.CacheWriteCostByModality != nil {
			pricing.CacheWriteCostByModality = tier.CacheWriteCostByModality
		}
		if tier.CacheReadCostByModality != nil {
			pricing.CacheReadCostByModality = tier.CacheReadCostByModality
		}
		return pricing
	}

	return pricing
}

func calculateBucketCost(total int, details *UsageTokenDetails, fallbackPrice float64, modalityPricing *ModalityPricing) float64 {
	if total <= 0 {
		return 0
	}

	remaining := total
	var cost float64

	add := func(tokens int, kind string) {
		if tokens <= 0 || remaining <= 0 {
			return
		}
		if tokens > remaining {
			tokens = remaining
		}
		cost += float64(tokens) * modalityPricing.priceFor(kind, fallbackPrice) / 1_000_000
		remaining -= tokens
	}

	if details != nil {
		add(details.TextTokens, "text")
		add(details.AudioTokens, "audio")
		add(details.ImageTokens, "image")
		add(details.VideoTokens, "video")
		add(details.DocumentTokens, "document")
	}

	if remaining > 0 {
		cost += float64(remaining) * fallbackPrice / 1_000_000
	}

	return cost
}

func subtractUsageTokenDetails(base *UsageTokenDetails, subtract ...*UsageTokenDetails) *UsageTokenDetails {
	if base == nil {
		return nil
	}

	result := cloneUsageTokenDetails(base)
	for _, item := range subtract {
		if item == nil {
			continue
		}
		result.TextTokens = max(result.TextTokens-item.TextTokens, 0)
		result.AudioTokens = max(result.AudioTokens-item.AudioTokens, 0)
		result.ImageTokens = max(result.ImageTokens-item.ImageTokens, 0)
		result.VideoTokens = max(result.VideoTokens-item.VideoTokens, 0)
		result.DocumentTokens = max(result.DocumentTokens-item.DocumentTokens, 0)
	}

	if result.Empty() {
		return nil
	}
	return result
}

func (m ModelConfig) SupportsModality(modality Modality) bool {
	if len(m.InputModalities) == 0 {
		return modality == ModalityText
	}
	return slices.Contains(m.InputModalities, modality)
}

func (m ModelConfig) FilterContent(content MessageContent) MessageContent {
	var filtered MessageContent
	for _, block := range content {
		supported := true
		switch block.Type {
		case ContentBlockTypeImage:
			supported = m.SupportsModality(ModalityImage)
		case ContentBlockTypeAudio:
			supported = m.SupportsModality(ModalityAudio)
		case ContentBlockTypeVideo:
			supported = m.SupportsModality(ModalityVideo)
		case ContentBlockTypeDocument:
			supported = m.SupportsModality(ModalityText)
		}
		if supported {
			filtered = append(filtered, block)
		}
	}
	return filtered
}

func (m EmbeddingModelConfig) Request(inputs ...ContentBlock) EmbeddingRequest {
	req := NewEmbeddingRequest(m.ModelID, inputs...)
	req.TaskType = m.TaskType
	req.Dimensions = cloneInt32Ptr(m.Dimensions)
	return req
}

func (m EmbeddingModelConfig) TextRequest(inputs ...string) EmbeddingRequest {
	req := NewTextEmbeddingRequest(m.ModelID, inputs...)
	req.TaskType = m.TaskType
	req.Dimensions = cloneInt32Ptr(m.Dimensions)
	return req
}

func (m EmbeddingModelConfig) SupportsModality(modality Modality) bool {
	if len(m.InputModalities) == 0 {
		return modality == ModalityText
	}
	return slices.Contains(m.InputModalities, modality)
}

func cloneInt32Ptr(value *int32) *int32 {
	if value == nil {
		return nil
	}
	copyValue := *value
	return &copyValue
}

func LoadCatalog(path string) (Catalog, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Catalog{}, err
	}
	var catalog Catalog
	if err := json.Unmarshal(data, &catalog); err != nil {
		return Catalog{}, err
	}
	return catalog, nil
}

func (c Catalog) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

type Resolver struct {
	mu                         sync.RWMutex
	providers                  map[string]ProviderConfig
	factories                  map[string]ProviderFactory
	models                     map[string]ModelConfig
	embeddingModels            map[string]EmbeddingModelConfig
	clients                    map[string]Client
	retryPolicy                RetryPolicy
	rateLimiters               map[string]*ProviderRateLimiter
	circuitBreakers            map[string]*ProviderCircuitBreaker
	providerMaxConcurrency     int
	providerMaxRPM             int
	providerBreakerMaxFailures int
	providerBreakerResetAfter  time.Duration
}

type Registry = Resolver

func NewResolver() *Resolver {
	return &Resolver{
		providers:                  make(map[string]ProviderConfig),
		factories:                  make(map[string]ProviderFactory),
		models:                     make(map[string]ModelConfig),
		embeddingModels:            make(map[string]EmbeddingModelConfig),
		clients:                    make(map[string]Client),
		retryPolicy:                DefaultRetryPolicy(),
		rateLimiters:               make(map[string]*ProviderRateLimiter),
		circuitBreakers:            make(map[string]*ProviderCircuitBreaker),
		providerMaxConcurrency:     4,
		providerMaxRPM:             60,
		providerBreakerMaxFailures: 5,
		providerBreakerResetAfter:  2 * time.Minute,
	}
}

func NewRegistry() *Resolver {
	return NewResolver()
}

func (r *Resolver) WithCatalog(c Catalog) *Resolver {
	return r.WithProviders(c.Providers...).WithModels(c.Models...).WithEmbeddingModels(c.EmbeddingModels...)
}

func (r *Resolver) ExportCatalog() Catalog {
	return Catalog{
		Providers:       r.ListProviders(),
		Models:          r.ListModels(),
		EmbeddingModels: r.ListEmbeddingModels(),
	}
}

func (r *Resolver) WithProviders(providers ...ProviderConfig) *Resolver {
	for _, p := range providers {
		r.RegisterProvider(p.ID, p)
	}
	return r
}

func (r *Resolver) WithModels(models ...ModelConfig) *Resolver {
	for _, m := range models {
		r.RegisterModel(m.ID, m)
	}
	return r
}

func (r *Resolver) WithEmbeddingModels(models ...EmbeddingModelConfig) *Resolver {
	for _, m := range models {
		r.RegisterEmbeddingModel(m.ID, m)
	}
	return r
}

func (r *Resolver) WithFactory(format APIFormat, factory ProviderFactory) *Resolver {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories[string(format)] = factory
	r.invalidateClientsForFormatLocked(format)
	return r
}

func (r *Resolver) WithRetryPolicy(policy RetryPolicy) *Resolver {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.retryPolicy = policy.normalized()
	clear(r.clients)
	return r
}

func (r *Resolver) WithProviderRateLimits(maxConcurrency, maxRPM int) *Resolver {
	r.mu.Lock()
	defer r.mu.Unlock()
	if maxConcurrency < 0 {
		maxConcurrency = 0
	}
	if maxRPM < 0 {
		maxRPM = 0
	}
	for _, limiter := range r.rateLimiters {
		if limiter != nil {
			limiter.Stop()
		}
	}
	r.providerMaxConcurrency = maxConcurrency
	r.providerMaxRPM = maxRPM
	r.rateLimiters = make(map[string]*ProviderRateLimiter)
	clear(r.clients)
	return r
}

func (r *Resolver) WithProviderCircuitBreaker(maxFailures int, resetAfter time.Duration) *Resolver {
	r.mu.Lock()
	defer r.mu.Unlock()
	if maxFailures < 0 {
		maxFailures = 0
	}
	if resetAfter <= 0 {
		resetAfter = 2 * time.Minute
	}
	r.providerBreakerMaxFailures = maxFailures
	r.providerBreakerResetAfter = resetAfter
	r.circuitBreakers = make(map[string]*ProviderCircuitBreaker)
	clear(r.clients)
	return r
}

func (r *Resolver) RegisterProvider(id string, config ProviderConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if config.ID == "" {
		config.ID = id
	}
	r.providers[id] = config
	delete(r.clients, id)
}

func (r *Resolver) RegisterModel(id string, config ModelConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if config.ID == "" {
		config.ID = id
	}
	r.models[id] = config
}

func (r *Resolver) RegisterEmbeddingModel(id string, config EmbeddingModelConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if config.ID == "" {
		config.ID = id
	}
	r.embeddingModels[id] = config
}

func (r *Resolver) GetModel(id string) (ModelConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.models[id]
	return m, ok
}

func (r *Resolver) GetProvider(id string) (ProviderConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.providers[id]
	return p, ok
}

func (r *Resolver) GetEmbeddingModel(id string) (EmbeddingModelConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.embeddingModels[id]
	return m, ok
}

func (r *Resolver) ListModels() []ModelConfig {
	r.mu.RLock()
	models := make([]ModelConfig, 0, len(r.models))
	for _, m := range r.models {
		models = append(models, m)
	}
	r.mu.RUnlock()
	slices.SortFunc(models, func(a, b ModelConfig) int {
		return compareCatalogIDs(a.ID, b.ID)
	})
	return models
}

func (r *Resolver) ListProviders() []ProviderConfig {
	r.mu.RLock()
	providers := make([]ProviderConfig, 0, len(r.providers))
	for _, p := range r.providers {
		providers = append(providers, p)
	}
	r.mu.RUnlock()
	slices.SortFunc(providers, func(a, b ProviderConfig) int {
		return compareCatalogIDs(a.ID, b.ID)
	})
	return providers
}

func (r *Resolver) ListEmbeddingModels() []EmbeddingModelConfig {
	r.mu.RLock()
	models := make([]EmbeddingModelConfig, 0, len(r.embeddingModels))
	for _, m := range r.embeddingModels {
		models = append(models, m)
	}
	r.mu.RUnlock()
	slices.SortFunc(models, func(a, b EmbeddingModelConfig) int {
		return compareCatalogIDs(a.ID, b.ID)
	})
	return models
}

func (r *Resolver) ListEmbeddingProviders() []ProviderConfig {
	r.mu.RLock()
	providers := make([]ProviderConfig, 0, len(r.providers))
	for _, p := range r.providers {
		if p.SupportsEmbeddings {
			providers = append(providers, p)
		}
	}
	r.mu.RUnlock()
	slices.SortFunc(providers, func(a, b ProviderConfig) int {
		return compareCatalogIDs(a.ID, b.ID)
	})
	return providers
}

func (r *Resolver) Resolve(ctx context.Context, model any) (Client, ModelConfig, error) {
	mCfg, err := r.resolveModelConfig(model)
	if err != nil {
		return nil, ModelConfig{}, err
	}

	client, _, err := r.resolveProviderClient(ctx, mCfg.ProviderID)
	if err != nil {
		return nil, ModelConfig{}, err
	}

	return client, mCfg, nil
}

func (r *Resolver) ResolveEmbedding(ctx context.Context, model any) (EmbeddingCapable, EmbeddingModelConfig, error) {
	mCfg, err := r.resolveEmbeddingModelConfig(model)
	if err != nil {
		return nil, EmbeddingModelConfig{}, err
	}

	client, provider, err := r.resolveProviderClient(ctx, mCfg.ProviderID)
	if err != nil {
		return nil, EmbeddingModelConfig{}, err
	}
	if !provider.SupportsEmbeddings {
		return nil, EmbeddingModelConfig{}, fmt.Errorf("provider %q does not expose embeddings", provider.ID)
	}

	embeddingClient, ok := client.(EmbeddingCapable)
	if !ok {
		return nil, EmbeddingModelConfig{}, fmt.Errorf("provider %q is not embedding capable", provider.ID)
	}

	return embeddingClient, mCfg, nil
}

func (r *Resolver) CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	client, config, err := r.ResolveEmbedding(ctx, req.Model)
	if err != nil {
		return nil, err
	}
	req.Model = config.ModelID
	resp, err := client.CreateEmbeddings(ctx, req)
	if err != nil || resp == nil || resp.Usage == nil {
		return resp, err
	}
	attributed, attrErr := config.AttributeUsage(resp.Usage)
	if attrErr != nil {
		return nil, attrErr
	}
	resp.Usage = attributed
	return resp, nil
}

func (r *Resolver) SupportsModality(modality Modality) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, m := range r.models {
		if m.SupportsModality(modality) {
			return true
		}
	}
	return false
}

func (r *Resolver) resolveModelConfig(model any) (ModelConfig, error) {
	switch m := model.(type) {
	case string:
		r.mu.RLock()
		cfg, ok := r.models[m]
		r.mu.RUnlock()
		if !ok {
			return ModelConfig{}, fmt.Errorf("model %q not found in catalog", m)
		}
		return cfg, nil
	case ModelConfig:
		var regCfg ModelConfig
		var found bool

		if m.ID != "" {
			r.mu.RLock()
			regCfg, found = r.models[m.ID]
			r.mu.RUnlock()
		}
		if !found && m.ModelID != "" {
			r.mu.RLock()
			for _, cfg := range r.models {
				if cfg.ModelID == m.ModelID {
					regCfg = cfg
					found = true
					break
				}
			}
			r.mu.RUnlock()
		}

		if found {
			if m.ID == "" {
				m.ID = regCfg.ID
			}
			if m.ProviderID == "" {
				m.ProviderID = regCfg.ProviderID
			}
			if m.ModelID == "" {
				m.ModelID = regCfg.ModelID
			}
			if m.Name == "" {
				m.Name = regCfg.Name
			}
			if m.ContextWindow == 0 {
				m.ContextWindow = regCfg.ContextWindow
			}
			if len(m.InputModalities) == 0 {
				m.InputModalities = regCfg.InputModalities
			}
			if len(m.OutputModalities) == 0 {
				m.OutputModalities = regCfg.OutputModalities
			}
			if m.Costs == nil {
				m.Costs = regCfg.Costs
			}
			if m.RecommendedParams == nil {
				m.RecommendedParams = regCfg.RecommendedParams
			}
		}

		return m, nil
	default:
		return ModelConfig{}, fmt.Errorf("invalid model type: %T", model)
	}
}

func (r *Resolver) resolveEmbeddingModelConfig(model any) (EmbeddingModelConfig, error) {
	switch m := model.(type) {
	case string:
		r.mu.RLock()
		cfg, ok := r.embeddingModels[m]
		r.mu.RUnlock()
		if !ok {
			return EmbeddingModelConfig{}, fmt.Errorf("embedding model %q not found in catalog", m)
		}
		return cfg, nil
	case EmbeddingModelConfig:
		return m, nil
	default:
		return EmbeddingModelConfig{}, fmt.Errorf("invalid embedding model type: %T", model)
	}
}

func (r *Resolver) resolveProviderClient(ctx context.Context, providerID string) (Client, ProviderConfig, error) {
	r.mu.RLock()
	if cached, ok := r.clients[providerID]; ok {
		provider, hasProvider := r.providers[providerID]
		r.mu.RUnlock()
		if !hasProvider {
			return nil, ProviderConfig{}, fmt.Errorf("provider %q not found", providerID)
		}
		return cached, provider, nil
	}
	provider, ok := r.providers[providerID]
	factory, hasFactory := r.factories[string(provider.Format)]
	policy := r.retryPolicy
	r.mu.RUnlock()
	if !ok {
		return nil, ProviderConfig{}, fmt.Errorf("provider %q not found", providerID)
	}
	if !hasFactory {
		return nil, ProviderConfig{}, fmt.Errorf("no provider factory registered for API format %q", provider.Format)
	}

	client, err := factory(ctx, provider)
	if err != nil {
		return nil, ProviderConfig{}, err
	}
	client = WrapClientWithRetry(client, policy)

	// Wrap with provider-level rate limiter and circuit breaker
	limiter, ok := r.rateLimiters[providerID]
	if !ok && (r.providerMaxConcurrency > 0 || r.providerMaxRPM > 0) {
		limiter = NewProviderRateLimiter(providerID, r.providerMaxConcurrency, r.providerMaxRPM)
		r.rateLimiters[providerID] = limiter
	}
	breaker, ok := r.circuitBreakers[providerID]
	if !ok && r.providerBreakerMaxFailures > 0 {
		breaker = NewProviderCircuitBreaker(providerID, r.providerBreakerMaxFailures, r.providerBreakerResetAfter)
		r.circuitBreakers[providerID] = breaker
	}
	if limiter != nil || breaker != nil {
		client = newClientWithRateLimit(client, limiter, breaker)
	}

	r.mu.Lock()
	if cached, ok := r.clients[providerID]; ok {
		r.mu.Unlock()
		return cached, provider, nil
	}
	r.clients[providerID] = client
	r.mu.Unlock()

	return client, provider, nil
}

func (r *Resolver) invalidateClientsForFormatLocked(format APIFormat) {
	for providerID, provider := range r.providers {
		if provider.Format == format {
			delete(r.clients, providerID)
		}
	}
}

func compareCatalogIDs(a, b string) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}
