package codeexec

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	llm "github.com/proiceremo/ai-sdk"
	aitools "github.com/proiceremo/ai-sdk/tools"
)

const nestedFinishMarker = "__ai_sdk_final"

// jsMaxInlineStringBytes is the hard ceiling for any individual string,
// template, or single-quote literal that appears inside the `code` field.
// Above this size we refuse to execute and instruct the model to move the
// content to the `input` parameter. Without this ceiling, models — especially
// smaller ones — routinely emit long inline strings, mis-escape a quote or
// backtick somewhere in the middle, and either trip a SyntaxError or hit
// max_tokens partway through the call and corrupt the tool_use JSON.
//
// 4KB is large enough that legitimate inline content (a short README, a sed
// expression, a multi-line code snippet) passes through unchallenged, while
// still small enough that the escape-risk per literal stays bounded. Truly
// large content — anything multi-page — must still route through the
// `input` parameter, which has no size cap on our end.
const jsMaxInlineStringBytes = 4 * 1024

type JSExecuteInput struct {
	Code      string         `json:"code" jsonschema:"description=JavaScript to run in QuickJS. Return the value to inspect.,minLength=1"`
	Summary   string         `json:"summary,omitempty" jsonschema:"description=Plain-language title for this call. ACP displays this as the tool title."`
	Input     map[string]any `json:"input,omitempty" jsonschema:"description=Values exposed as global input inside JS. Use this for long text, markdown, patches, file bodies, and strings with many quotes/backticks/backslashes."`
	TimeoutMS int            `json:"timeout_ms,omitempty" jsonschema:"description=Optional execution timeout in milliseconds (default 180000 = 3 minutes)."`
}

func effectiveJSTitle(input JSExecuteInput) string {
	if input.Summary != "" {
		return input.Summary
	}
	title := "Execute JS"
	snippet := strings.TrimSpace(input.Code)
	if len(snippet) > 0 {
		if len(snippet) > 40 {
			snippet = snippet[:37] + "..."
		}
		title += ": " + snippet
	}
	return title
}

func NewTool(executor *QuickJSExecutor, actionTools []llm.Tool) llm.Tool {
	actionTools = withFinishTool(actionTools)
	description := jsExecuteDescription(actionTools)
	tool := aitools.NewGenericTool("js_execute", description, func(ctx llm.ToolContext, input JSExecuteInput) llm.ToolResult {
		if executor == nil {
			errBlock := llm.NewTextContentBlock("ERROR: javascript executor is not configured")
			return llm.ToolResult{
				Error:    fmt.Errorf("javascript executor is not configured"),
				ErrorStr: "javascript executor is not configured",
				Output:   []llm.ContentBlock{errBlock},
				Metadata: llm.ToolMetadata{Title: "Error", Kind: llm.ToolKindOther, Content: []llm.ToolCallContent{llm.NewToolCallContentBlock(errBlock)}},
			}
		}
		// Pre-execution gate: enforce the "long content rides through `input`,
		// not the JS source" contract. Smaller models otherwise inline huge
		// strings into the code, mis-escape a quote/backtick somewhere in the
		// middle, and either trip a SyntaxError or hit max_tokens partway
		// through the call (which then corrupts the tool_use JSON itself).
		if err := validateJSExecuteInput(input); err != nil {
			errBlock := llm.NewTextContentBlock("ERROR: " + err.Error())
			return llm.ToolResult{
				Error:    err,
				ErrorStr: err.Error(),
				Output:   []llm.ContentBlock{errBlock},
				Metadata: llm.ToolMetadata{
					Title:   "Inline content too large",
					Kind:    llm.ToolKindOther,
					Content: []llm.ToolCallContent{llm.NewToolCallContentBlock(errBlock)},
				},
			}
		}
		timeout := time.Duration(input.TimeoutMS) * time.Millisecond
		reqCtx := ctx.Context
		if input.Summary != "" {
			reqCtx = context.WithValue(reqCtx, llm.JSSummaryContextKey, input.Summary)
		}
		forwarded := NewForwardedContentCollector()
		result, err := executor.Execute(reqCtx, JSRequest{
			Code:             input.Code,
			Input:            input.Input,
			Timeout:          timeout,
			WorkingDirectory: ctx.WorkingDirectory,
			HomeDir:          llm.ToolProHome(ctx),
			ScopeSessionID:   llm.ToolSessionID(ctx),
			ScopeActorID:     llm.ToolActorID(ctx),
			ScopeRunID:       llm.ToolRunID(ctx),
			Emit: func(event string) {
				if ctx.Emit != nil {
					ctx.Emit(event)
				}
			},
			NestedToolUpdate: ctx.NestedToolUpdate,
			ForwardedContent: forwarded,
			ToolContext:      ctx,
			Tools:            actionTools,
			PermissionHook:   nestedToolPermission,
		})
		if err != nil {
			errBlock := llm.NewTextContentBlock("ERROR: " + err.Error())
			title := input.Summary
			if title == "" {
				title = "Execute JS"
				if len(input.Code) > 0 {
					snippet := strings.TrimSpace(input.Code)
					if len(snippet) > 40 {
						snippet = snippet[:37] + "..."
					}
					title += ": " + snippet
				}
			}
			return llm.ToolResult{
				Error:    err,
				ErrorStr: err.Error(),
				Output:   []llm.ContentBlock{errBlock},
				Metadata: llm.ToolMetadata{Title: title, Kind: llm.ToolKindOther, Content: []llm.ToolCallContent{llm.NewToolCallContentBlock(errBlock)}},
			}
		}

		output := make([]llm.ContentBlock, 0, len(result.Logs)+1)
		for _, log := range result.Logs {
			output = append(output, llm.NewTextContentBlock("[log] "+log))
		}

		if result.JSON != "" {
			block := llm.NewTextContentBlock("```json\n" + result.JSON + "\n```")
			output = append(output, block)
			output = append(output, result.ForwardedContent...)
			metadataContent := toolCallContentFromBlocks(output)
			var structured any
			_ = json.Unmarshal([]byte(result.JSON), &structured)
			toolResult := llm.ToolResult{
				Output:           output,
				StructuredOutput: structured,
				Metadata: llm.ToolMetadata{
					Title:   effectiveJSTitle(input),
					Kind:    llm.ToolKindOther,
					Content: metadataContent,
				},
			}
			if finalOutput, ok := nestedFinishOutput(structured); ok {
				toolResult.Final = true
				toolResult.StructuredOutput = finalOutput
				if rendered, err := llm.RenderFinishOutput(finalOutput); err == nil && rendered != "" {
					// Replace the JSON dump with the user-facing rendering:
					// summary as prose, output (if any) as a fenced block
					// underneath. Falls back gracefully when only one of
					// summary/output is present.
					toolResult.Output = []llm.ContentBlock{llm.NewTextContentBlock(rendered)}
					toolResult.Output = append(toolResult.Output, result.ForwardedContent...)
					toolResult.Metadata.Content = toolCallContentFromBlocks(toolResult.Output)
				}
			}
			return toolResult
		}

		text := strings.TrimSpace(result.Text)
		if text == "undefined" || text == "" {
			text = "(no value returned — make sure your code returns the value you want, e.g. `return await tools.foo({...})`)"
		}
		block := llm.NewTextContentBlock("```text\n" + text + "\n```")
		output = append(output, block)
		output = append(output, result.ForwardedContent...)
		return llm.ToolResult{
			Output: output,
			Metadata: llm.ToolMetadata{
				Title:   effectiveJSTitle(input),
				Kind:    llm.ToolKindOther,
				Content: toolCallContentFromBlocks(output),
			},
		}
	})
	return jsExecuteTool{tool: tool}
}

func toolCallContentFromBlocks(blocks []llm.ContentBlock) []llm.ToolCallContent {
	out := make([]llm.ToolCallContent, 0, len(blocks))
	for _, block := range blocks {
		out = append(out, llm.NewToolCallContentBlock(block))
	}
	return out
}

type jsExecuteTool struct {
	tool llm.Tool
}

func (t jsExecuteTool) Schema() llm.ToolSchema {
	schema := t.tool.Schema()
	schema.InputSchema = jsExecuteInputSchema(schema.InputSchema)
	return schema
}

func (t jsExecuteTool) Execute(ctx llm.ToolContext, input map[string]any) llm.ToolResult {
	return t.tool.Execute(ctx, input)
}

func jsExecuteInputSchema(schema llm.JSONSchema) llm.JSONSchema {
	cloned := cloneJSONSchema(schema)
	properties, _ := cloned["properties"].(map[string]any)
	if properties == nil {
		properties = map[string]any{}
		cloned["properties"] = properties
	}
	inputSchema, _ := properties["input"].(map[string]any)
	if inputSchema == nil {
		inputSchema = map[string]any{}
		properties["input"] = inputSchema
	}
	inputSchema["type"] = "object"
	inputSchema["additionalProperties"] = true
	return cloned
}

func cloneJSONSchema(schema llm.JSONSchema) llm.JSONSchema {
	data, err := json.Marshal(schema)
	if err != nil {
		return llm.JSONSchema{}
	}
	var cloned llm.JSONSchema
	if err := json.Unmarshal(data, &cloned); err != nil {
		return llm.JSONSchema{}
	}
	return cloned
}

func nestedToolPermission(ctx context.Context, req JSRequest, tool llm.Tool, call llm.ToolUse) (bool, error) {
	guarded, ok := tool.(llm.GuardedTool)
	if !ok {
		return true, nil
	}
	toolCtx := req.ToolContext
	if toolCtx.Context == nil {
		toolCtx.Context = ctx
	}
	toolCtx = withToolRuntimeVars(toolCtx, req.ScopeSessionID, req.ScopeActorID, req.ScopeRunID, "")
	if toolCtx.WorkingDirectory == "" {
		toolCtx.WorkingDirectory = req.WorkingDirectory
	}
	var inputMap map[string]any
	_ = json.Unmarshal(call.Input, &inputMap)
	guards, err := guarded.GetPermissions(toolCtx, inputMap)
	if err != nil || len(guards) == 0 {
		return len(guards) == 0, err
	}
	if toolCtx.Permission == nil {
		return false, fmt.Errorf("tool %s requires permission but no permission handler is configured", tool.Schema().Name)
	}
	resp, err := toolCtx.Permission(ctx, llm.PermissionRequest{ToolCall: call, Guards: guards})
	if err != nil {
		return false, err
	}
	return resp.Approved, nil
}

func withFinishTool(actionTools []llm.Tool) []llm.Tool {
	out := append([]llm.Tool(nil), actionTools...)
	for _, tool := range out {
		if tool != nil && tool.Schema().Name == "finish" {
			return out
		}
	}
	finish, err := llm.NewFinishTool(nil)
	if err != nil {
		return out
	}
	return append(out, finish)
}

// jsExecuteDescription is the model-facing contract for js_execute. Keep it
// compact and prescriptive so smaller models can follow it under pressure.
func jsExecuteDescription(actionTools []llm.Tool) string {
	var b strings.Builder
	b.WriteString("Execute JavaScript in a QuickJS sandbox to call tools.\n\n")
	b.WriteString("Rules:\n")
	b.WriteString("- Not Node.js: no require/import/fetch/filesystem/network. Use `tools.<name>({...})` only.\n")
	b.WriteString("- Always pass one object argument. Return the value you need to inspect.\n")
	b.WriteString("- Include `summary`; ACP shows it as the tool title.\n")
	b.WriteString("- For concise structured completion: `return await tools.finish({ summary: input.summary, output: input.output })`. `summary` must be useful, not just \"done\".\n")
	b.WriteString("- If the final user-facing answer is long, do not pack it into a tool call. Stop using tools and answer directly as normal assistant text, or write a report file and finish with the path.\n")
	b.WriteString("- Batch independent reads with `Promise.all`; use `Promise.allSettled` when partial failure is acceptable. Keep dependent edits sequential.\n")
	b.WriteString("- Use `find` for path discovery (glob) and `grep` for content search (re2 regex; set `literal:true` for plain text). `ls` lists a directory.\n")
	b.WriteString("- Use `read({ path })` to get the full file (errors if over 2000 lines / 50KB). Use `read_range({ path, offset, limit })` to inspect a slice; the result is intentionally partial — do NOT pass it into `write`. If you need to modify a large file, use `edit` with a unique `oldText` or `read` it in full first.\n")
	b.WriteString("- Use `edit({ path, edits:[{oldText, newText}, ...] })` for in-place changes. Each `oldText` must appear exactly once in the original file and must not overlap with other edits in the call. Include enough surrounding context to disambiguate, and merge nearby changes into a single edit rather than emitting overlapping edits.\n")
	b.WriteString("- Use `write({ path, content })` only for new files or complete rewrites. It creates parent directories automatically.\n")
	b.WriteString("- Put long text, markdown, patches, file bodies, and strings with many quotes/backticks/backslashes in `input` and reference them via `input.key` to keep code clean.\n")
	b.WriteString("- If a tool input is rejected as invalid/truncated, do not retry the same payload. Split it or move large content into `input`.\n\n")
	b.WriteString("Globals: `tools`, `input`, `cwd`, `env`, `vars`, `console`. `vars` persists across js_execute calls; keep it small: handles, paths, hashes, offsets, short notes. Do not store large outputs or secrets.\n\n")
	b.WriteString("Example: batched read\n")
	b.WriteString("```js\n")
	b.WriteString("const [readme, tests] = await Promise.all([\n")
	b.WriteString("  tools.read({ path: 'README.md' }),\n")
	b.WriteString("  tools.grep({ pattern: 'RunAgent', path: 'ai-sdk' })\n")
	b.WriteString("]);\n")
	b.WriteString("vars.lastRead = { path: 'README.md' };\n")
	b.WriteString("return { readme, tests };\n")
	b.WriteString("```\n\n")
	b.WriteString("Example: targeted edit\n")
	b.WriteString("```js\n")
	b.WriteString("const hit = await tools.grep({ pattern: 'Clipboard.Text', path: '.', glob: '**/*.go' });\n")
	b.WriteString("const m = hit.matches.find(x => x.marker === ':');\n")
	b.WriteString("return await tools.edit({\n")
	b.WriteString("  path: m.path,\n")
	b.WriteString("  edits: [{ oldText: 'app.Clipboard.Text()', newText: 'app.Clipboard.Text() // updated' }]\n")
	b.WriteString("});\n")
	b.WriteString("```\n\n")
	b.WriteString("Example: concise structured finish\n")
	b.WriteString("```json\n")
	b.WriteString("{\n")
	b.WriteString("  \"code\": \"return await tools.finish({ summary: input.summary, output: input.output });\",\n")
	b.WriteString("  \"input\": {\n")
	b.WriteString("    \"summary\": \"Audited the docs and found two stale setup references.\",\n")
	b.WriteString("    \"output\": { \"verified\": [\"docs/index.md\", \"docs/architecture.md\"] }\n")
	b.WriteString("  }\n")
	b.WriteString("}\n")
	b.WriteString("```\n\n")
	b.WriteString("Tool type contract:\n")
	b.WriteString("```ts\n")
	b.WriteString(BuildToolTypeDeclarations(actionTools))
	b.WriteString("```\n")
	return b.String()
}

func nestedFinishOutput(value any) (map[string]any, bool) {
	object, ok := value.(map[string]any)
	if !ok || object[nestedFinishMarker] != true {
		return nil, false
	}
	if structured, ok := object["structured_output"].(map[string]any); ok {
		return structured, true
	}
	delete(object, nestedFinishMarker)
	return object, true
}

// validateJSExecuteInput was previously used to enforce the "long content
// rides through input" contract. The 4KB inline string literal limit was
// removed because it was found to be more detrimental than beneficial (capable
// models routinely write clean, inline content and scripts over 4KB).
func validateJSExecuteInput(in JSExecuteInput) error {
	return nil
}

// longestInlineStringLiteral scans JavaScript source for string, template,
// and single-quote literals and returns the kind ("template" / "string" /
// "char") and length of the longest one found. The scan respects:
//   - Backslash escapes inside ' " ` literals.
//   - `${ ... }` interpolation spans inside template literals (interpolated
//     code is NOT counted toward the template literal's length).
//   - Line and block comments (skipped, never count as literal content).
//
// The scan is deliberately tolerant: malformed source returns whatever it
// found before falling out the end. The aim is to flag "the model is about
// to inline a huge blob of escape-prone content", not to be a full JS parser.
func longestInlineStringLiteral(code string) (kind string, length int, found bool) {
	const (
		stateCode = iota
		stateLineComment
		stateBlockComment
		stateSingle
		stateDouble
		stateTemplate
	)
	state := stateCode
	literalStart := 0
	templateDepth := 0
	interpDepth := 0
	escape := false

	consider := func(name string, n int) {
		if n > length {
			length = n
			kind = name
			found = true
		}
	}

	for i := 0; i < len(code); i++ {
		c := code[i]
		switch state {
		case stateCode:
			switch {
			case c == '/' && i+1 < len(code) && code[i+1] == '/':
				state = stateLineComment
				i++
			case c == '/' && i+1 < len(code) && code[i+1] == '*':
				state = stateBlockComment
				i++
			case c == '\'':
				state = stateSingle
				literalStart = i + 1
			case c == '"':
				state = stateDouble
				literalStart = i + 1
			case c == '`':
				state = stateTemplate
				literalStart = i + 1
				templateDepth = 1
			}
		case stateLineComment:
			if c == '\n' {
				state = stateCode
			}
		case stateBlockComment:
			if c == '*' && i+1 < len(code) && code[i+1] == '/' {
				state = stateCode
				i++
			}
		case stateSingle:
			switch {
			case escape:
				escape = false
			case c == '\\':
				escape = true
			case c == '\'':
				consider("char", i-literalStart)
				state = stateCode
			}
		case stateDouble:
			switch {
			case escape:
				escape = false
			case c == '\\':
				escape = true
			case c == '"':
				consider("string", i-literalStart)
				state = stateCode
			}
		case stateTemplate:
			switch {
			case escape:
				escape = false
			case c == '\\':
				escape = true
			case c == '$' && i+1 < len(code) && code[i+1] == '{':
				// Enter interpolation — interpolated source code does not
				// count toward the template literal's content length. We
				// treat the interpolation as nested code by switching
				// state and counting brace depth.
				consider("template", i-literalStart)
				state = stateCode
				interpDepth = 1
				i++
				// Continue scanning the interpolated code; when the
				// matching `}` fires we resume the template.
				for i++; i < len(code) && interpDepth > 0; i++ {
					ic := code[i]
					switch ic {
					case '{':
						interpDepth++
					case '}':
						interpDepth--
						if interpDepth == 0 {
							// Resume template starting at the next byte.
							state = stateTemplate
							literalStart = i + 1
						}
					case '\'':
						// Skip nested string literal in interpolation.
						for j := i + 1; j < len(code); j++ {
							if code[j] == '\\' {
								j++
								continue
							}
							if code[j] == '\'' {
								i = j
								break
							}
						}
					case '"':
						for j := i + 1; j < len(code); j++ {
							if code[j] == '\\' {
								j++
								continue
							}
							if code[j] == '"' {
								i = j
								break
							}
						}
					case '`':
						// Nested template inside interpolation: just skip
						// to its closing backtick without counting.
						for j := i + 1; j < len(code); j++ {
							if code[j] == '\\' {
								j++
								continue
							}
							if code[j] == '`' {
								i = j
								break
							}
						}
					}
				}
				// step back one so the outer loop's i++ lands correctly
				i--
			case c == '`':
				consider("template", i-literalStart)
				state = stateCode
				templateDepth = 0
			}
		}
	}
	// If we ran off the end while still inside a literal, count the partial
	// span — better to over-trigger the safety net than under-trigger.
	switch state {
	case stateSingle:
		consider("char", len(code)-literalStart)
	case stateDouble:
		consider("string", len(code)-literalStart)
	case stateTemplate:
		consider("template", len(code)-literalStart)
	}
	_ = templateDepth
	return kind, length, found
}
