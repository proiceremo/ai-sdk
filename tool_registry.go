package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

type ToolConfig struct {
	ID     string          `json:"id"`
	Config json.RawMessage `json:"config,omitempty"`
}

type ToolBuildContext struct {
	WorkingDirectory string
	ModelConfig      *ModelConfig
	Vars             map[string]any
	// InnerTools are the tools that should be exposed inside js_execute.
	// When an agent definition lists inner tools, the registry builds them
	// first and passes them here so the js_execute factory can wrap them.
	InnerTools []Tool
}

type ToolFactory func(ctx context.Context, build ToolBuildContext, config json.RawMessage) (Tool, error)

type ToolRegistry struct {
	mu        sync.RWMutex
	factories map[string]ToolFactory
}

func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{factories: map[string]ToolFactory{}}
}

func (r *ToolRegistry) Register(id string, factory ToolFactory) error {
	id = strings.TrimSpace(id)
	if id == "" {
		return fmt.Errorf("tool factory id is required")
	}
	if factory == nil {
		return fmt.Errorf("tool factory %q is nil", id)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories[id] = factory
	return nil
}

func (r *ToolRegistry) MustRegister(id string, factory ToolFactory) *ToolRegistry {
	if err := r.Register(id, factory); err != nil {
		panic(err)
	}
	return r
}

// Has returns true if a factory is registered for the given tool id.
func (r *ToolRegistry) Has(id string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.factories[id]
	return ok
}

func (r *ToolRegistry) BuildTools(ctx context.Context, build ToolBuildContext, configs []ToolConfig) ([]Tool, error) {
	if len(configs) == 0 {
		return nil, nil
	}
	if r == nil {
		return nil, fmt.Errorf("tool registry is nil")
	}
	seen := map[string]bool{}
	tools := make([]Tool, 0, len(configs))
	for _, cfg := range configs {
		id := strings.TrimSpace(cfg.ID)
		if id == "" {
			return nil, fmt.Errorf("tool config id is required")
		}
		if seen[id] {
			return nil, fmt.Errorf("duplicate tool config id %q", id)
		}
		seen[id] = true

		r.mu.RLock()
		factory := r.factories[id]
		r.mu.RUnlock()
		if factory == nil {
			return nil, fmt.Errorf("unknown tool config id %q", id)
		}
		tool, err := factory(ctx, build, cfg.Config)
		if err != nil {
			return nil, fmt.Errorf("build tool %q: %w", id, err)
		}
		if tool == nil {
			return nil, fmt.Errorf("build tool %q returned nil", id)
		}
		tools = append(tools, tool)
	}
	return tools, nil
}
