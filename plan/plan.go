package plan

import (
	llm "ai-sdk"
	"ai-sdk/tools"
)

type Entry struct {
	ID       string `json:"id"`
	Content  string `json:"content"`
	Priority string `json:"priority,omitempty"`
	Status   string `json:"status,omitempty"`
}

type Input struct {
	Entries []Entry `json:"entries"`
}

type Sink interface {
	UpdatePlan(ctx llm.ToolContext, entries []Entry) error
}

func NewTool(sink Sink) llm.Tool {
	return tools.NewGenericTool("plan", "Update the execution plan with task entries", func(ctx llm.ToolContext, input Input) llm.ToolResult {
		if sink != nil {
			_ = sink.UpdatePlan(ctx, input.Entries)
		}
		return tools.JSONResult("Update plan", llm.ToolKindThink, input.Entries, "")
	})
}
