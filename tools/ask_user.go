package tools

import (
	"fmt"
	"strings"

	llm "github.com/proiceremo/ai-sdk"
)

type AskUserInput struct {
	Prompt string         `json:"prompt" jsonschema:"description=Question or request to show to the user"`
	Schema map[string]any `json:"schema,omitempty" jsonschema:"description=Optional JSON schema for structured response fields"`
}

type AskUserResponse struct {
	Values map[string]any `json:"values,omitempty"`
}

type AskUserHandler func(ctx llm.ToolContext, input AskUserInput) (AskUserResponse, error)

type AskUserToolOptions struct {
	Name        string
	Description string
	Handler     AskUserHandler
}

func NewAskUserTool(opts AskUserToolOptions) llm.Tool {
	name := opts.Name
	if name == "" {
		name = "ask_user"
	}
	description := opts.Description
	if description == "" {
		description = "Ask the user for clarification, feedback, or approval via the client UI."
	}
	return NewGenericTool(name, description, func(ctx llm.ToolContext, input AskUserInput) llm.ToolResult {
		if strings.TrimSpace(input.Prompt) == "" {
			return ErrorResult(fmt.Errorf("prompt is required"))
		}
		if opts.Handler == nil {
			return ErrorResult(fmt.Errorf("ask_user handler is not configured"))
		}
		resp, err := opts.Handler(ctx, input)
		if err != nil {
			return ErrorResult(err)
		}
		return JSONResult("Ask user", llm.ToolKindOther, resp.Values, "")
	})
}
