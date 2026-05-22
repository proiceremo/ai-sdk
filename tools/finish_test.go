package tools_test

import (
	"context"
	"testing"

	llm "ai-sdk"
	"ai-sdk/tools"
)

func TestFinishToolReturnsStructuredOutput(t *testing.T) {
	type finalAnswer struct {
		Answer string `json:"answer" jsonschema:"required"`
		Score  int    `json:"score" jsonschema:"required"`
	}

	finishTool := tools.NewFinishTool[finalAnswer](tools.FinishToolOptions{})
	result := finishTool.Execute(llm.ToolContext{Context: context.Background()}, map[string]any{
		"answer": "done",
		"score":  7,
	})

	if result.Error != nil {
		t.Fatalf("finish tool returned error: %v", result.Error)
	}
	if !result.Final {
		t.Fatalf("finish tool should mark final result")
	}
	output := llm.MessageContent(result.Output)
	if got := output.Text(); got != `{"answer":"done","score":7}` {
		t.Fatalf("unexpected finish output text: %q", got)
	}
	if got, ok := result.StructuredOutput.(finalAnswer); !ok || got.Answer != "done" || got.Score != 7 {
		t.Fatalf("unexpected structured output: %#v", result.StructuredOutput)
	}
	if !finishTool.Schema().Strict {
		t.Fatal("finish tool schema should be strict")
	}
}
