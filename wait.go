package llm

type AgentWaitKind string

const (
	AgentWaitKindToolApproval AgentWaitKind = "tool_approval"
	AgentWaitKindElicitation  AgentWaitKind = "elicitation"
	AgentWaitKindWorkflow     AgentWaitKind = "workflow"
)

type AgentWait struct {
	Kind     AgentWaitKind    `json:"kind"`
	ToolCall ToolUse          `json:"tool_call,omitempty"`
	Guard    *PermissionGuard `json:"guard,omitempty"`
	Prompt   string           `json:"prompt,omitempty"`
	Metadata map[string]any   `json:"metadata,omitempty"`
}
