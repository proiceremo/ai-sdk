package agent

import (
	"encoding/json"

	llm "ai-sdk"
)

// Definition is a declarative agent recipe.
type Definition struct {
	ID            string              `json:"id"`
	Name          string              `json:"name"`
	Description   string              `json:"description,omitempty"`
	SystemPrompt  string              `json:"system_prompt,omitempty"`
	ModelType     string              `json:"model_type,omitempty"`
	Model         string              `json:"model,omitempty"`
	Inference     llm.InferenceParams `json:"inference,omitempty"`
	Tools         []llm.Tool          `json:"-"`
	Extensions    []ExtensionConfig   `json:"extensions,omitempty"`
	Finish        *llm.FinishConfig   `json:"finish,omitempty"`
	ChildAgents   []string            `json:"child_agents,omitempty"`
	Metadata      map[string]any      `json:"metadata,omitempty"`
	TemplateData  map[string]any      `json:"template_data,omitempty"`
	TemplateFuncs map[string]any      `json:"-"`
}

type ExtensionConfig struct {
	ID     string          `json:"id"`
	Config json.RawMessage `json:"config,omitempty"`
}

// Registry holds agent definitions in memory.
type Registry struct {
	defs map[string]Definition
}

func NewRegistry() *Registry {
	return &Registry{defs: map[string]Definition{}}
}

func (r *Registry) Register(d Definition) {
	r.defs[d.ID] = d
}

func (r *Registry) Get(id string) (Definition, bool) {
	d, ok := r.defs[id]
	return d, ok
}

func (r *Registry) List() []Definition {
	out := make([]Definition, 0, len(r.defs))
	for _, d := range r.defs {
		out = append(out, d)
	}
	return out
}
