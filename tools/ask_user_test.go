package tools

import (
	"testing"

	llm "ai-sdk"
)

func TestAskUserToolRequiresPrompt(t *testing.T) {
	tool := NewAskUserTool(AskUserToolOptions{
		Handler: func(ctx llm.ToolContext, input AskUserInput) (AskUserResponse, error) {
			t.Fatal("handler should not run for empty prompt")
			return AskUserResponse{}, nil
		},
	})
	result := tool.Execute(llm.ToolContext{}, map[string]any{"prompt": ""})
	if result.Error == nil {
		t.Fatal("expected empty prompt to fail")
	}
}

func TestAskUserToolCallsHandler(t *testing.T) {
	tool := NewAskUserTool(AskUserToolOptions{
		Handler: func(ctx llm.ToolContext, input AskUserInput) (AskUserResponse, error) {
			if input.Prompt != "Proceed?" {
				t.Fatalf("unexpected prompt: %q", input.Prompt)
			}
			return AskUserResponse{Values: map[string]any{"answer": "yes"}}, nil
		},
	})
	result := tool.Execute(llm.ToolContext{}, map[string]any{"prompt": "Proceed?"})
	if result.Error != nil {
		t.Fatal(result.Error)
	}
	values, ok := result.StructuredOutput.(map[string]any)
	if !ok || values["answer"] != "yes" {
		t.Fatalf("unexpected structured output: %#v", result.StructuredOutput)
	}
}
