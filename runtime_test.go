package llm

import (
	"encoding/json"
	"testing"
)

func TestFinalizeToolUseInputMergesSeedObjectAndDelta(t *testing.T) {
	tool := &ToolUse{
		Input:        json.RawMessage(`{"path":"README.md"}`),
		partialInput: `,"content":"updated"}`,
	}

	got := finalizeToolUseInput(tool)
	want := `{"path":"README.md","content":"updated"}`
	if string(got) != want {
		t.Fatalf("finalizeToolUseInput() = %q, want %q", got, want)
	}
}

func TestFinalizeToolUseInputEscapesControlCharsInsideStrings(t *testing.T) {
	tool := &ToolUse{
		partialInput: "{\"content\":\"line1\n\tline2\"}",
	}

	got := finalizeToolUseInput(tool)
	if !json.Valid(got) {
		t.Fatalf("expected repaired tool input to be valid JSON, got %q", got)
	}

	var payload struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(got, &payload); err != nil {
		t.Fatalf("failed to unmarshal repaired tool input: %v", err)
	}
	if payload.Content != "line1\n\tline2" {
		t.Fatalf("unexpected repaired content %q", payload.Content)
	}
}
