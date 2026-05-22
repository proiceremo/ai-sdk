package tools

import (
	"encoding/json"
	"fmt"

	llm "github.com/proiceremo/ai-sdk"
)

// JSONResult builds a ToolResult around a value rendered as pretty-printed
// JSON. Use it when a tool wants to emit a structured payload that's both
// human-readable in the transcript and machine-parsable downstream. The
// `path` field, when non-empty, attaches a ToolCallLocation so the host can
// surface the tool call in file UI.
func JSONResult(title string, kind llm.ToolKind, value any, path string) llm.ToolResult {
	json := mustJSON(value)
	block := llm.NewTextContentBlock("```json\n" + json + "\n```")
	locations := []llm.ToolCallLocation{}
	if path != "" {
		locations = append(locations, llm.ToolCallLocation{Path: path})
	}
	return llm.ToolResult{
		Output:           []llm.ContentBlock{block},
		StructuredOutput: value,
		Metadata: llm.ToolMetadata{
			Title:     title,
			Kind:      kind,
			Locations: locations,
			Content:   []llm.ToolCallContent{llm.NewToolCallContentBlock(block)},
		},
	}
}

// ErrorResult is the canonical "this tool call failed" result shape: it wires
// up Error, ErrorStr, and a textual ERROR block so every layer (Go callers,
// transcript renderers, JS bridge) sees a consistent failure signal.
func ErrorResult(err error) llm.ToolResult {
	if err == nil {
		return llm.ToolResult{}
	}
	errBlock := llm.NewTextContentBlock("ERROR: " + err.Error())
	return llm.ToolResult{
		Error:    err,
		ErrorStr: err.Error(),
		Output:   []llm.ContentBlock{errBlock},
		StructuredOutput: map[string]any{
			"error": err.Error(),
		},
		Metadata: llm.ToolMetadata{
			Title:   "Error",
			Kind:    llm.ToolKindOther,
			Content: []llm.ToolCallContent{llm.NewToolCallContentBlock(errBlock)},
		},
	}
}

func mustJSON(value any) string {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Sprintf(`{"error":%q}`, err.Error())
	}
	return string(data)
}
