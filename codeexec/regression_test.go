package codeexec

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	llm "github.com/proiceremo/ai-sdk"
	aitools "github.com/proiceremo/ai-sdk/tools"
)

// TestStringArgYieldsActionableError pins down the previous "json: cannot
// unmarshal string into Go value of type map[string]interface {}" failure
// mode: when the model passes a non-object to a tool, we want a clear,
// model-actionable error instead of a Go internals leak.
func TestStringArgYieldsActionableError(t *testing.T) {
	executor := NewQuickJSExecutor()
	tool := aitools.NewGenericTool("echo", "Echo.", func(ctx llm.ToolContext, input struct {
		Value string `json:"value"`
	}) llm.ToolResult {
		return llm.ToolResult{StructuredOutput: map[string]any{"value": input.Value}}
	})
	executor.SetTools([]llm.Tool{tool})
	_, err := executor.Execute(context.Background(), JSRequest{
		Code:    `return await tools.echo("not-an-object");`,
		Timeout: 2 * time.Second,
		Tools:   []llm.Tool{tool},
	})
	if err == nil {
		t.Fatalf("expected error from string argument, got nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, "object argument") {
		t.Fatalf("expected actionable error mentioning object argument, got: %v", msg)
	}
	if strings.Contains(msg, "cannot unmarshal string into Go value") {
		t.Fatalf("expected error to NOT leak Go internals, got: %v", msg)
	}
}

// TestMissingReturnIsCaptured pins down the "tool succeeded but the agent
// got `undefined` back" failure mode: we want the LAST expression of a
// multi-line block to be returned automatically, like Cloudflare's codemode
// AST normaliser does.
func TestMissingReturnIsCaptured(t *testing.T) {
	executor := NewQuickJSExecutor()
	result, err := executor.Execute(context.Background(), JSRequest{
		Code: `
const a = 2;
const b = 3;
({ sum: a + b })
`,
		Timeout: 2 * time.Second,
	})
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.JSON == "" {
		t.Fatalf("expected JSON result with last expression captured, got %#v", result)
	}
	var payload struct {
		Sum int `json:"sum"`
	}
	if err := json.Unmarshal([]byte(result.JSON), &payload); err != nil {
		t.Fatalf("failed to parse payload %q: %v", result.JSON, err)
	}
	if payload.Sum != 5 {
		t.Fatalf("expected sum=5, got %d", payload.Sum)
	}
}

// TestPersistentVarsSurviveBetweenExecutions pins down the new "vars persists
// across js_execute calls within a session" contract.
func TestPersistentVarsSurviveBetweenExecutions(t *testing.T) {
	executor := NewQuickJSExecutor()
	sessionID := "ses_persistent_vars"

	if _, err := executor.Execute(context.Background(), JSRequest{
		Code:           `vars.lastFile = "README.md"; vars.count = (vars.count || 0) + 1; return vars;`,
		ScopeSessionID: sessionID,
		Timeout:        2 * time.Second,
	}); err != nil {
		t.Fatalf("first call failed: %v", err)
	}

	result, err := executor.Execute(context.Background(), JSRequest{
		Code:           `vars.count += 1; return vars;`,
		ScopeSessionID: sessionID,
		Timeout:        2 * time.Second,
	})
	if err != nil {
		t.Fatalf("second call failed: %v", err)
	}
	var payload struct {
		LastFile string `json:"lastFile"`
		Count    int    `json:"count"`
	}
	if err := json.Unmarshal([]byte(result.JSON), &payload); err != nil {
		t.Fatalf("failed to parse payload %q: %v", result.JSON, err)
	}
	if payload.LastFile != "README.md" {
		t.Fatalf("expected vars.lastFile to persist, got %q", payload.LastFile)
	}
	if payload.Count != 2 {
		t.Fatalf("expected vars.count to be 2 across two calls, got %d", payload.Count)
	}

	// And ResetSession actually wipes the session-scoped store.
	executor.ResetSession(sessionID)
	if got := executor.SessionVars(sessionID); got != nil {
		t.Fatalf("expected session vars to be cleared after ResetSession, got %#v", got)
	}
}

// TestUnknownToolErrorListsAvailable pins down the "model called a tool by a
// name that doesn't exist" failure mode: we want the error to list what IS
// available so the model can self-correct on the next turn.
func TestUnknownToolErrorListsAvailable(t *testing.T) {
	executor := NewQuickJSExecutor()
	echo := aitools.NewGenericTool("echo", "Echo.", func(ctx llm.ToolContext, _ struct{}) llm.ToolResult {
		return llm.ToolResult{StructuredOutput: map[string]any{}}
	})
	_, err := executor.Execute(context.Background(), JSRequest{
		Code:    `return await tools.does_not_exist({});`,
		Timeout: 2 * time.Second,
		Tools:   []llm.Tool{echo},
	})
	if err == nil {
		t.Fatalf("expected error for unknown tool, got nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, "does_not_exist") {
		t.Fatalf("expected error to name the missing tool, got: %v", msg)
	}
	if !strings.Contains(msg, "echo") {
		t.Fatalf("expected error to list available tools, got: %v", msg)
	}
}

// TestTSDeclarationsRender pins down the model-facing tool surface format —
// we need a TypeScript-style declaration block so the model picks the right
// argument names.
func TestTSDeclarationsRender(t *testing.T) {
	echo := aitools.NewGenericTool("echo_value", "Echo back the value.", func(ctx llm.ToolContext, input struct {
		Value string `json:"value" jsonschema:"description=The value to echo,required"`
	}) llm.ToolResult {
		return llm.ToolResult{}
	})
	decls := BuildToolTypeDeclarations([]llm.Tool{echo})
	if !strings.Contains(decls, "declare const vars: Record<string, JSONValue>") {
		t.Fatalf("expected persistent vars declaration, got:\n%s", decls)
	}
	if !strings.Contains(decls, "declare const env: Readonly") || !strings.Contains(decls, "session_id") {
		t.Fatalf("expected env declaration with session metadata, got:\n%s", decls)
	}
	if !strings.Contains(decls, "type EchoValueInput") {
		t.Fatalf("expected EchoValueInput declaration, got:\n%s", decls)
	}
	if !strings.Contains(decls, "echo_value(input: EchoValueInput): Promise<any>") {
		t.Fatalf("expected echo_value method signature, got:\n%s", decls)
	}
	if !strings.Contains(decls, "value") {
		t.Fatalf("expected value field in declarations, got:\n%s", decls)
	}
}

func TestJSExecuteDescriptionExplainsHarnessAndVars(t *testing.T) {
	echo := aitools.NewGenericTool("echo_value", "Echo back the value.", func(ctx llm.ToolContext, input struct {
		Value string `json:"value"`
	}) llm.ToolResult {
		return llm.ToolResult{}
	})
	description := jsExecuteDescription([]llm.Tool{echo})
	for _, want := range []string{
		"QuickJS sandbox",
		"Always pass one object argument",
		"Promise.all",
		"vars",
		"persists across js_execute calls",
		"Do not store large outputs",
		"tools.finish",
		"declare const vars",
		"echo_value(input: EchoValueInput)",
	} {
		if !strings.Contains(description, want) {
			t.Fatalf("expected description to contain %q, got:\n%s", want, description)
		}
	}
}

func TestJSExecuteInputSchemaAllowsArbitraryInputObject(t *testing.T) {
	tool := NewTool(NewQuickJSExecutor(), nil)
	schema := tool.Schema().InputSchema
	properties, _ := schema["properties"].(map[string]any)
	inputSchema, _ := properties["input"].(map[string]any)
	if inputSchema == nil {
		t.Fatalf("expected input property schema, got %#v", schema)
	}
	if inputSchema["additionalProperties"] != true {
		t.Fatalf("expected js_execute input to allow arbitrary properties, got %#v", inputSchema)
	}
}

func TestNestedToolCallsUsePermissionHandler(t *testing.T) {
	executor := NewQuickJSExecutor()
	guarded := aitools.NewGenericTool("guarded", "Guarded.", func(ctx llm.ToolContext, input struct {
		Value string `json:"value"`
	}) llm.ToolResult {
		return llm.ToolResult{StructuredOutput: map[string]any{"value": input.Value}}
	}).WithPermissionExtractor(func(ctx llm.ToolContext, input struct {
		Value string `json:"value"`
	}) ([]llm.PermissionGuard, error) {
		return []llm.PermissionGuard{{Key: "guarded", Specifiers: []string{input.Value}}}, nil
	})

	var seen llm.ToolUse
	result := NewTool(executor, []llm.Tool{guarded}).Execute(withToolRuntimeVars(llm.ToolContext{
		Context: context.Background(),
		Permission: func(ctx context.Context, req llm.PermissionRequest) (llm.PermissionResponse, error) {
			seen = req.ToolCall
			if len(req.Guards) != 1 || req.Guards[0].Key != "guarded" {
				t.Fatalf("unexpected guards: %#v", req.Guards)
			}
			return llm.PermissionResponse{Approved: true}, nil
		},
	}, "ses_nested_permission", "", "", ""), map[string]any{
		"code": `return await tools.guarded({ value: "ok" });`,
	})
	if result.ErrorStr != "" || result.Error != nil {
		t.Fatalf("expected nested guarded call to pass, got %v %s", result.Error, result.ErrorStr)
	}
	if seen.Name != "guarded" || !strings.Contains(string(seen.Input), `"value":"ok"`) {
		t.Fatalf("expected permission handler to see guarded tool call, got %#v", seen)
	}
}

func TestNestedToolPermissionHandlerReceivesMultipleGuards(t *testing.T) {
	executor := NewQuickJSExecutor()
	guarded := aitools.NewGenericTool("guarded", "Guarded.", func(ctx llm.ToolContext, input struct{}) llm.ToolResult {
		return llm.ToolResult{StructuredOutput: map[string]any{"ok": true}}
	}).WithPermissionExtractor(func(ctx llm.ToolContext, input struct{}) ([]llm.PermissionGuard, error) {
		return []llm.PermissionGuard{
			{Key: "first"},
			{Key: "second"},
		}, nil
	})

	var seen []llm.PermissionGuard
	result := NewTool(executor, []llm.Tool{guarded}).Execute(withToolRuntimeVars(llm.ToolContext{
		Context: context.Background(),
		Permission: func(ctx context.Context, req llm.PermissionRequest) (llm.PermissionResponse, error) {
			seen = req.Guards
			return llm.PermissionResponse{Approved: true, Results: []llm.PermissionDecision{
				{Guard: req.Guards[0], Approved: true},
				{Guard: req.Guards[1], Approved: true},
			}}, nil
		},
	}, "ses_nested_multi_permission", "", "", ""), map[string]any{
		"code": `return await tools.guarded({});`,
	})
	if result.ErrorStr != "" || result.Error != nil {
		t.Fatalf("expected nested guarded call to pass, got %v %s", result.Error, result.ErrorStr)
	}
	if len(seen) != 2 || seen[0].Key != "first" || seen[1].Key != "second" {
		t.Fatalf("expected both guards, got %#v", seen)
	}
}

func TestNestedToolPermissionDenialRejectsCall(t *testing.T) {
	executor := NewQuickJSExecutor()
	guarded := aitools.NewGenericTool("guarded", "Guarded.", func(ctx llm.ToolContext, input struct{}) llm.ToolResult {
		return llm.ToolResult{StructuredOutput: map[string]any{"ok": true}}
	}).WithPermissionExtractor(func(ctx llm.ToolContext, input struct{}) ([]llm.PermissionGuard, error) {
		return []llm.PermissionGuard{{Key: "guarded"}}, nil
	})

	result := NewTool(executor, []llm.Tool{guarded}).Execute(withToolRuntimeVars(llm.ToolContext{
		Context: context.Background(),
		Permission: func(ctx context.Context, req llm.PermissionRequest) (llm.PermissionResponse, error) {
			return llm.PermissionResponse{Approved: false}, nil
		},
	}, "ses_nested_permission_denied", "", "", ""), map[string]any{
		"code": `return await tools.guarded({});`,
	})
	if result.ErrorStr == "" || !strings.Contains(result.ErrorStr, "rejected") {
		t.Fatalf("expected permission denial error, got %v %s", result.Error, result.ErrorStr)
	}
}

func TestJSExecuteFeedsForwardNestedRichToolResults(t *testing.T) {
	executor := NewQuickJSExecutor()
	screenshot := aitools.NewGenericTool("screenshot", "Return screenshot.", func(ctx llm.ToolContext, input struct{}) llm.ToolResult {
		block := llm.NewImageContentBlockFromBase64("iVBORw0KGgo=", "image/png")
		return llm.ToolResult{
			Output:           []llm.ContentBlock{llm.NewTextContentBlock("screenshot captured"), block},
			StructuredOutput: map[string]any{"ok": true},
			Metadata:         llm.ToolMetadata{Content: []llm.ToolCallContent{llm.NewToolCallContentBlock(block)}},
		}
	})
	model := llm.ModelConfig{InputModalities: []llm.Modality{llm.ModalityText, llm.ModalityImage}}
	result := NewTool(executor, []llm.Tool{screenshot}).Execute(llm.ToolContext{
		Context:     context.Background(),
		ModelConfig: &model,
	}, map[string]any{
		"code": `return await tools.screenshot({});`,
	})
	if result.ErrorStr != "" || result.Error != nil {
		t.Fatalf("expected js_execute success, got %v %s", result.Error, result.ErrorStr)
	}
	var images int
	for _, block := range result.Output {
		if block.Type == llm.ContentBlockTypeImage {
			images++
			if !block.Ephemeral {
				t.Fatalf("expected forwarded image block to be ephemeral: %#v", block)
			}
			continue
		}
		if block.Type == llm.ContentBlockTypeText {
			if strings.Contains(block.Text, "iVBORw0KGgo=") {
				t.Fatalf("js_execute JSON should not inline nested image bytes: %s", block.Text)
			}
		}
	}
	if images != 1 {
		t.Fatalf("expected one forwarded image block, got %d in %#v", images, result.Output)
	}
}
