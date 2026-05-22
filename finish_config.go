package llm

type FinishConfig struct {
	Required    *bool      `json:"required,omitempty"`
	Name        string     `json:"name,omitempty"`
	Description string     `json:"description,omitempty"`
	InputSchema JSONSchema `json:"input_schema,omitempty"`
}
