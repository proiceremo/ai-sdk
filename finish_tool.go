package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

type coreFinishTool struct {
	name        string
	description string
	inputSchema JSONSchema
}

func newCoreFinishTool(cfg *FinishConfig) (Tool, error) {
	name := "finish"
	description := "Finish the run. `summary` must be the complete concise user-facing answer, not a generic done/complete note. Put machine-readable details in `output`."
	schema := DefaultFinishInputSchema()
	if cfg != nil {
		if strings.TrimSpace(cfg.Name) != "" {
			name = strings.TrimSpace(cfg.Name)
		}
		if strings.TrimSpace(cfg.Description) != "" {
			description = strings.TrimSpace(cfg.Description)
		}
		if len(cfg.InputSchema) > 0 {
			schema = cfg.InputSchema
		}
	}
	normalized, err := NormalizeToolInputSchema(schema)
	if err != nil {
		return nil, fmt.Errorf("invalid finish input schema: %w", err)
	}
	return coreFinishTool{name: name, description: description, inputSchema: normalized}, nil
}

func NewFinishTool(cfg *FinishConfig) (Tool, error) {
	return newCoreFinishTool(cfg)
}

func (t coreFinishTool) Schema() ToolSchema {
	return ToolSchema{
		Name:        t.name,
		Description: t.description,
		InputSchema: t.inputSchema,
		Strict:      true,
	}
}

func (t coreFinishTool) Execute(_ ToolContext, input map[string]any) ToolResult {
	if input == nil {
		input = map[string]any{}
	}
	if err := validateFinishInput(input); err != nil {
		errBlock := NewTextContentBlock("ERROR: " + err.Error())
		return ToolResult{
			Error:    err,
			ErrorStr: err.Error(),
			Output:   []ContentBlock{errBlock},
			Metadata: ToolMetadata{Title: "Invalid finish", Kind: ToolKindOther, Content: []ToolCallContent{NewToolCallContentBlock(errBlock)}},
		}
	}
	rendered, err := RenderFinishOutput(input)
	if err != nil {
		errBlock := NewTextContentBlock("ERROR: " + err.Error())
		return ToolResult{
			Error:    err,
			ErrorStr: err.Error(),
			Output:   []ContentBlock{errBlock},
			Metadata: ToolMetadata{Title: "Error", Kind: ToolKindOther, Content: []ToolCallContent{NewToolCallContentBlock(errBlock)}},
		}
	}
	return ToolResult{
		Output:           []ContentBlock{NewTextContentBlock(rendered)},
		StructuredOutput: input,
		Final:            true,
		Metadata: ToolMetadata{
			Title: "Finish",
			Kind:  ToolKindOther,
		},
	}
}

func validateFinishInput(input map[string]any) error {
	output, hasOutput := input["output"]
	if !hasOutput || output == nil {
		return nil
	}
	if !isPlaceholderOnlyValue(output) {
		return nil
	}
	summary := strings.TrimSpace(stringOf(input["summary"]))
	if len(summary) > 240 || strings.Contains(summary, "\n") {
		return nil
	}
	return fmt.Errorf("finish output is only a placeholder such as \"see summary\"; provide the actual answer/evidence, answer directly as normal assistant text for long responses, or write a report file and finish with its path")
}

// RenderFinishOutput composes the human-facing message produced by a
// finish-tool call. The model passes a `{summary, output}` object: summary
// is the prose answer the user sees first, output is structured data the
// caller might want to inspect or hand to a downstream system.
//
// The rendering rules:
//   - If summary is empty and output is empty/missing, fall back to a
//     pretty-printed JSON dump of the whole input so the model's intent
//     isn't silently lost.
//   - If only summary is present (or output is empty/null/empty-string),
//     emit it as plain text on its own.
//   - If both are present, emit the summary unless it is just a generic
//     completion phrase. In that case render output so useful details are not
//     hidden behind "Done.".
func RenderFinishOutput(input map[string]any) (string, error) {
	if input == nil {
		input = map[string]any{}
	}
	summary := strings.TrimSpace(stringOf(input["summary"]))
	output, hasOutput := input["output"]
	outputBlock, err := renderFinishOutputField(output, hasOutput)
	if err != nil {
		return "", err
	}
	if summary == "" && outputBlock == "" {
		data, err := json.MarshalIndent(input, "", "  ")
		if err != nil {
			return "", err
		}
		return "```json\n" + string(data) + "\n```", nil
	}
	if outputBlock == "" {
		return summary, nil
	}
	if summary == "" {
		return outputBlock, nil
	}
	if isGenericFinishSummary(summary) {
		return renderUsefulFinishOutput(output, hasOutput, outputBlock)
	}
	return summary, nil
}

// renderFinishOutputField turns the value of the `output` key into a
// fenced block. Returns the empty string when output is absent, null, or
// an empty string — in those cases callers should fall back to summary
// alone.
func renderFinishOutputField(value any, present bool) (string, error) {
	if !present || value == nil {
		return "", nil
	}
	switch v := value.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return "", nil
		}
		return "```text\n" + v + "\n```", nil
	default:
		data, err := json.MarshalIndent(v, "", "  ")
		if err != nil {
			return "", err
		}
		return "```json\n" + string(data) + "\n```", nil
	}
}

func renderUsefulFinishOutput(value any, present bool, fallback string) (string, error) {
	if !present || value == nil {
		return fallback, nil
	}
	if text, ok := value.(string); ok {
		text = strings.TrimSpace(text)
		if text != "" {
			return text, nil
		}
	}
	if fallback != "" {
		return fallback, nil
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return "", err
	}
	return "```json\n" + string(data) + "\n```", nil
}

func isGenericFinishSummary(summary string) bool {
	s := strings.ToLower(strings.TrimSpace(summary))
	s = strings.Trim(s, " \t\r\n.!:")
	if s == "" || len(s) > 80 {
		return false
	}
	switch s {
	case "done", "all done", "complete", "completed", "finished", "all set", "success", "succeeded",
		"fixed", "updated", "implemented", "created", "task complete", "task completed",
		"finished successfully", "completed successfully", "done successfully":
		return true
	}
	words := strings.Fields(s)
	if len(words) <= 4 {
		for _, word := range words {
			switch word {
			case "done", "complete", "completed", "finished", "fixed", "updated", "implemented", "created":
				return true
			}
		}
	}
	return false
}

func isPlaceholderOnlyValue(value any) bool {
	switch v := value.(type) {
	case nil:
		return true
	case string:
		return isPlaceholderString(v)
	case []any:
		if len(v) == 0 {
			return true
		}
		for _, item := range v {
			if !isPlaceholderOnlyValue(item) {
				return false
			}
		}
		return true
	case map[string]any:
		if len(v) == 0 {
			return true
		}
		for _, item := range v {
			if !isPlaceholderOnlyValue(item) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func isPlaceholderString(value string) bool {
	s := strings.ToLower(strings.TrimSpace(value))
	s = strings.Trim(s, " \t\r\n.!:")
	switch s {
	case "", "n/a", "na", "none", "null", "todo", "tbd", "placeholder", "see summary",
		"same as summary", "see above", "as above", "omitted", "omitted for brevity",
		"see previous", "see response", "see final":
		return true
	default:
		return false
	}
}

func stringOf(value any) string {
	if value == nil {
		return ""
	}
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}
