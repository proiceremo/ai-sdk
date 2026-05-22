package llm

import (
	"strings"
	"testing"
)

func TestRenderFinishOutputSummaryOnly(t *testing.T) {
	got, err := RenderFinishOutput(map[string]any{
		"summary": "All checks passed.",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "All checks passed." {
		t.Fatalf("expected bare summary, got %q", got)
	}
}

func TestRenderFinishOutputSummaryWithStructuredOutputOmitsJSONFence(t *testing.T) {
	got, err := RenderFinishOutput(map[string]any{
		"summary": "Detailed codebase breakdown delivered below.",
		"output": map[string]any{
			"verified": []any{"docs/index.md", "docs/architecture.md"},
			"checks":   true,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// When both summary and structured output are present, only the summary
	// is rendered to avoid duplicating the payload (StructuredOutput carries
	// the raw data separately).
	if got != "Detailed codebase breakdown delivered below." {
		t.Fatalf("expected bare summary, got %q", got)
	}
	if strings.Contains(got, "```json") {
		t.Fatalf("structured output should not be rendered in the text block, got %q", got)
	}
}

func TestRenderFinishOutputStringOutputOmitsTextFence(t *testing.T) {
	got, err := RenderFinishOutput(map[string]any{
		"summary": "Generated patch:",
		"output":  "diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// String output is available via StructuredOutput; it must not be
	// duplicated inside the rendered text block.
	if got != "Generated patch:" {
		t.Fatalf("expected bare summary, got %q", got)
	}
	if strings.Contains(got, "```text") || strings.Contains(got, "```json") {
		t.Fatalf("output should not be rendered in any fence, got %q", got)
	}
}

func TestRenderFinishOutputGenericSummaryShowsUsefulStringOutput(t *testing.T) {
	got, err := RenderFinishOutput(map[string]any{
		"summary": "Done.",
		"output":  "Created `Sidebar.svelte`, wired keyboard shortcuts, and verified `npm test`.",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "Created `Sidebar.svelte`, wired keyboard shortcuts, and verified `npm test`."
	if got != want {
		t.Fatalf("expected useful output, got %q", got)
	}
}

func TestRenderFinishOutputGenericSummaryShowsUsefulStructuredOutput(t *testing.T) {
	got, err := RenderFinishOutput(map[string]any{
		"summary": "Complete",
		"output": map[string]any{
			"changed": []any{"src/Sidebar.svelte"},
			"tests":   "npm test",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "\"changed\"") || !strings.HasPrefix(got, "```json\n") {
		t.Fatalf("expected structured output to be rendered, got %q", got)
	}
}

func TestRenderFinishOutputDropsEmptyAndNilOutputs(t *testing.T) {
	cases := []map[string]any{
		{"summary": "Done.", "output": nil},
		{"summary": "Done.", "output": ""},
		{"summary": "Done.", "output": "   "},
		{"summary": "Done."}, // missing key
	}
	for _, input := range cases {
		got, err := RenderFinishOutput(input)
		if err != nil {
			t.Fatalf("unexpected error for %#v: %v", input, err)
		}
		if got != "Done." {
			t.Fatalf("expected bare summary for %#v, got %q", input, got)
		}
	}
}

func TestRenderFinishOutputFallsBackToFullDumpWhenBothEmpty(t *testing.T) {
	got, err := RenderFinishOutput(map[string]any{
		"answer": "42",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(got, "```json\n") || !strings.HasSuffix(got, "\n```") {
		t.Fatalf("expected JSON-fenced fallback, got %q", got)
	}
	if !strings.Contains(got, "\"answer\": \"42\"") {
		t.Fatalf("fallback should include all input keys, got %q", got)
	}
}

func TestRenderFinishOutputOutputOnly(t *testing.T) {
	got, err := RenderFinishOutput(map[string]any{
		"output": map[string]any{"k": "v"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(got, "```json\n") {
		t.Fatalf("expected JSON-fenced output when summary is missing, got %q", got)
	}
}

func TestFinishToolRejectsPlaceholderOnlyOutput(t *testing.T) {
	tool, err := NewFinishTool(nil)
	if err != nil {
		t.Fatalf("NewFinishTool: %v", err)
	}
	result := tool.Execute(ToolContext{}, map[string]any{
		"summary": "Completed a full detailed breakdown of every file in the repo.",
		"output":  map[string]any{"report": "see summary"},
	})
	if result.Error == nil {
		t.Fatalf("expected placeholder finish to be rejected")
	}
	if result.Final {
		t.Fatalf("invalid finish must not end the run")
	}
	if !strings.Contains(result.ErrorStr, "placeholder") {
		t.Fatalf("expected placeholder guidance, got %q", result.ErrorStr)
	}
}

func TestFinishToolAllowsSubstantiveOutput(t *testing.T) {
	tool, err := NewFinishTool(nil)
	if err != nil {
		t.Fatalf("NewFinishTool: %v", err)
	}
	result := tool.Execute(ToolContext{}, map[string]any{
		"summary": "Completed the audit.",
		"output":  map[string]any{"files_changed": []any{"v2/runtime.go"}, "tests": "go test ./v2"},
	})
	if result.Error != nil {
		t.Fatalf("expected substantive finish to pass: %v", result.Error)
	}
	if !result.Final {
		t.Fatalf("valid finish should end the run")
	}
}
