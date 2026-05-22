package codeexec

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	llm "github.com/proiceremo/ai-sdk"

	"github.com/fastschema/qjs"
)

type JSRequest struct {
	Code             string
	Input            map[string]any
	Timeout          time.Duration
	WorkingDirectory string
	HomeDir          string
	ScopeSessionID   string
	ScopeActorID     string
	ScopeRunID       string
	Emit             func(event string)
	ToolContext      llm.ToolContext
	Tools            []llm.Tool
	// PermissionHook is consulted before invoking a guarded tool from the JS
	// sandbox. Returning false rejects the call and surfaces an error to the
	// generated code.
	PermissionHook func(ctx context.Context, req JSRequest, tool llm.Tool, call llm.ToolUse) (bool, error)
	// NestedToolUpdate is called before and after each tool invocation from
	// inside the js_execute sandbox so the harness can surface them to the
	// bridge / client.
	NestedToolUpdate func(ctx context.Context, call llm.ToolUse, status string, result *llm.ToolResult) error
	ForwardedContent *ForwardedContentCollector
	// Globals lets callers inject extra host-side functions (e.g. plan emit)
	// into the global scope before the user code runs.
	Globals map[string]func(jsCtx *qjs.Context) (qjs.Value, error)
}

type JSResult struct {
	JSON             string             `json:"json,omitempty"`
	Text             string             `json:"text,omitempty"`
	Logs             []string           `json:"logs,omitempty"`
	Vars             map[string]any     `json:"vars,omitempty"`
	ForwardedContent []llm.ContentBlock `json:"-"`
}

type ForwardedContentCollector struct {
	mu     sync.Mutex
	seen   map[string]bool
	blocks []llm.ContentBlock
}

func NewForwardedContentCollector() *ForwardedContentCollector {
	return &ForwardedContentCollector{seen: map[string]bool{}}
}

func (c *ForwardedContentCollector) Add(callID string, blocks []llm.ContentBlock, model *llm.ModelConfig) {
	if c == nil || len(blocks) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.seen == nil {
		c.seen = map[string]bool{}
	}
	for _, block := range blocks {
		if !isForwardableContentBlock(block, model) {
			continue
		}
		key := forwardedContentKey(callID, block)
		if c.seen[key] {
			continue
		}
		c.seen[key] = true
		c.blocks = append(c.blocks, llm.EphemeralContentBlock(block.Clone()))
	}
}

func (c *ForwardedContentCollector) Blocks() []llm.ContentBlock {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]llm.ContentBlock, len(c.blocks))
	for i, block := range c.blocks {
		out[i] = block.Clone()
	}
	return out
}

func isForwardableContentBlock(block llm.ContentBlock, model *llm.ModelConfig) bool {
	switch block.Type {
	case llm.ContentBlockTypeImage:
		return model == nil || model.SupportsModality(llm.ModalityImage)
	case llm.ContentBlockTypeAudio:
		return model == nil || model.SupportsModality(llm.ModalityAudio)
	case llm.ContentBlockTypeVideo:
		return model == nil || model.SupportsModality(llm.ModalityVideo)
	case llm.ContentBlockTypeDocument:
		return true
	default:
		return false
	}
}

func forwardedContentKey(callID string, block llm.ContentBlock) string {
	data, _ := json.Marshal(block)
	sum := sha256.Sum256(data)
	return callID + ":" + hex.EncodeToString(sum[:])
}

type QuickJSExecutor struct {
	mu          sync.Mutex
	globalTools []llm.Tool
	sessionVars map[string]map[string]any
}

func NewQuickJSExecutor() *QuickJSExecutor {
	return &QuickJSExecutor{sessionVars: map[string]map[string]any{}}
}

func (e *QuickJSExecutor) SetTools(tools []llm.Tool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.globalTools = append([]llm.Tool(nil), tools...)
}

// ResetSession drops any persistent variables associated with sessionID. Call
// this when a session is closed so we don't keep stale state forever.
func (e *QuickJSExecutor) ResetSession(sessionID string) {
	if sessionID == "" {
		return
	}
	e.mu.Lock()
	delete(e.sessionVars, sessionID)
	e.mu.Unlock()
}

// SessionVars exposes a snapshot of the persistent vars for a session so
// callers (tests, debug surfaces) can inspect them.
func (e *QuickJSExecutor) SessionVars(sessionID string) map[string]any {
	if sessionID == "" {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	src := e.sessionVars[sessionID]
	if len(src) == 0 {
		return nil
	}
	out := make(map[string]any, len(src))
	for k, v := range src {
		out[k] = v
	}
	return out
}

func newRandomID(n int) string {
	b := make([]byte, (n+1)/2)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)[:n]
}

func (e *QuickJSExecutor) loadVars(sessionID string) map[string]any {
	if sessionID == "" {
		return map[string]any{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	src := e.sessionVars[sessionID]
	out := make(map[string]any, len(src))
	for k, v := range src {
		out[k] = v
	}
	return out
}

func (e *QuickJSExecutor) storeVars(sessionID string, vars map[string]any) {
	if sessionID == "" {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if vars == nil {
		delete(e.sessionVars, sessionID)
		return
	}
	cloned := make(map[string]any, len(vars))
	for k, v := range vars {
		cloned[k] = v
	}
	e.sessionVars[sessionID] = cloned
}

func (e *QuickJSExecutor) Execute(ctx context.Context, req JSRequest) (JSResult, error) {
	if strings.TrimSpace(req.Code) == "" {
		return JSResult{}, fmt.Errorf("code is required")
	}
	if req.Timeout <= 0 {
		req.Timeout = 180 * time.Second
	}
	done := make(chan struct {
		result JSResult
		err    error
	}, 1)
	go func() {
		result, err := e.execute(ctx, req)
		done <- struct {
			result JSResult
			err    error
		}{result: result, err: err}
	}()
	timer := time.NewTimer(req.Timeout)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return JSResult{}, ctx.Err()
	case <-timer.C:
		return JSResult{}, fmt.Errorf("javascript execution timed out after %s", req.Timeout)
	case item := <-done:
		return item.result, item.err
	}
}

func (e *QuickJSExecutor) execute(ctx context.Context, req JSRequest) (JSResult, error) {
	rt, err := qjs.New(qjs.Option{
		MaxExecutionTime: int(req.Timeout / time.Millisecond),
		MemoryLimit:      64 * 1024 * 1024,
	})
	if err != nil {
		return JSResult{}, err
	}
	defer rt.Close()
	jsCtx := rt.Context()

	var logs []string
	logFn, err := qjs.FuncToJS(jsCtx, func(args ...string) (bool, error) {
		logs = append(logs, strings.Join(args, " "))
		return true, nil
	})
	if err != nil {
		return JSResult{}, err
	}
	jsCtx.Global().SetPropertyStr("__log", logFn)

	input := req.Input
	if input == nil {
		input = map[string]any{}
	}
	inputValue, err := qjs.ToJsValue(jsCtx, input)
	if err != nil {
		return JSResult{}, err
	}
	jsCtx.Global().SetPropertyStr("__input", inputValue)

	cwdValue := jsCtx.NewString(req.WorkingDirectory)
	jsCtx.Global().SetPropertyStr("__cwd", cwdValue)

	envValue, err := qjs.ToJsValue(jsCtx, jsExecutionEnv(req))
	if err != nil {
		return JSResult{}, err
	}
	jsCtx.Global().SetPropertyStr("__env", envValue)

	availableTools := req.Tools
	e.mu.Lock()
	if len(availableTools) == 0 {
		availableTools = e.globalTools
	}
	e.mu.Unlock()

	toolCatalogJSON, err := jsonString(jsToolCatalog(availableTools))
	if err != nil {
		return JSResult{}, err
	}
	jsCtx.Global().SetPropertyStr("__toolCatalogJSON", jsCtx.NewString(toolCatalogJSON))

	persistentVars := e.loadVars(req.ScopeSessionID)
	varsJSON, err := jsonString(persistentVars)
	if err != nil {
		return JSResult{}, err
	}
	jsCtx.Global().SetPropertyStr("__varsJSON", jsCtx.NewString(varsJSON))

	emitFn, err := qjs.FuncToJS(jsCtx, func(message string) (bool, error) {
		if req.Emit != nil {
			req.Emit(message)
		}
		return true, nil
	})
	if err != nil {
		return JSResult{}, err
	}
	jsCtx.Global().SetPropertyStr("emit", emitFn)

	callToolFn, err := qjs.FuncToJS(jsCtx, func(name string, inputJSON string) (string, error) {
		value, err := e.callTool(ctx, req, availableTools, name, inputJSON)
		if err != nil {
			return "", err
		}
		return jsonString(value)
	})
	if err != nil {
		return JSResult{}, err
	}
	jsCtx.Global().SetPropertyStr("__callTool", callToolFn)

	for name, builder := range req.Globals {
		val, err := builder(jsCtx)
		if err != nil {
			return JSResult{}, fmt.Errorf("failed to inject global %q: %w", name, err)
		}
		jsCtx.Global().SetPropertyStr(name, &val)
	}

	if _, err := jsCtx.Eval("ai_sdk_prelude.js", qjs.Code(jsPrelude()), qjs.FlagStrict()); err != nil {
		return JSResult{}, err
	}

	normalized := NormalizeCode(req.Code)
	value, err := jsCtx.Eval("ai_sdk_eval.js", qjs.Code(normalized), qjs.FlagStrict())
	if err != nil {
		return JSResult{Logs: logs}, simplifyJSError(err)
	}
	var res JSResult
	if value.IsPromise() {
		awaited, err := value.Await()
		if err != nil {
			return JSResult{Logs: logs}, simplifyJSError(err)
		}
		res, err = jsValueResult(awaited)
		if err != nil {
			return JSResult{Logs: logs}, err
		}
	} else {
		res, err = jsValueResult(value)
		if err != nil {
			return JSResult{Logs: logs}, err
		}
	}

	updated := readVars(jsCtx)
	e.storeVars(req.ScopeSessionID, updated)
	res.Logs = logs
	if len(updated) > 0 {
		res.Vars = updated
	}
	if req.ForwardedContent != nil {
		res.ForwardedContent = req.ForwardedContent.Blocks()
	}
	return res, nil
}

// simplifyJSError trims the noisy QuickJS stack frames so the model sees the
// actual exception line rather than a wall of internal traces — but keeps
// the first frame that points at our eval/prelude source so syntax errors
// retain a usable line:column. Without that, "SyntaxError: Unexpected
// identifier 'Read'" is unactionable; with it the model can locate the
// stray quote/escape that broke the parse.
func simplifyJSError(err error) error {
	if err == nil {
		return nil
	}
	lines := strings.Split(err.Error(), "\n")
	if len(lines) == 0 {
		return err
	}
	desc := strings.TrimSpace(lines[0])
	if loc := firstUserFrameLocation(lines[1:]); loc != "" {
		return fmt.Errorf("%s (at %s)", desc, loc)
	}
	return fmt.Errorf("%s", desc)
}

// firstUserFrameLocation returns "<source>:<line>:<col>" for the first stack
// frame referring to model-supplied code (ai_sdk_eval.js) or the harness
// prelude (ai_sdk_prelude.js). Internal qjs frames are skipped.
func firstUserFrameLocation(stack []string) string {
	const evalSource = "ai_sdk_eval.js"
	const preludeSource = "ai_sdk_prelude.js"
	for _, raw := range stack {
		line := strings.TrimSpace(raw)
		if !strings.HasPrefix(line, "at ") {
			continue
		}
		// Frames come in two shapes:
		//   "at <source>:<line>:<col>"
		//   "at <fn> (<source>:<line>:<col>)"
		body := strings.TrimPrefix(line, "at ")
		if open := strings.LastIndex(body, "("); open != -1 {
			if close := strings.LastIndex(body, ")"); close > open {
				body = body[open+1 : close]
			}
		}
		if strings.Contains(body, evalSource) || strings.Contains(body, preludeSource) {
			return body
		}
	}
	return ""
}

func readVars(jsCtx *qjs.Context) map[string]any {
	value := jsCtx.Global().GetPropertyStr("vars")
	if value == nil || value.IsUndefined() || value.IsNull() {
		return nil
	}
	jsonText, err := value.JSONStringify()
	if err != nil || jsonText == "" || jsonText == "undefined" {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(jsonText), &out); err != nil {
		return nil
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func jsExecutionEnv(req JSRequest) map[string]any {
	model := ""
	if req.ToolContext.ModelConfig != nil {
		model = req.ToolContext.ModelConfig.ID
	}
	return map[string]any{
		"cwd":        req.WorkingDirectory,
		"pro_home":   req.HomeDir,
		"session_id": req.ScopeSessionID,
		"run_id":     req.ScopeRunID,
		"model":      model,
		"vars":       req.ToolContext.Vars,
	}
}

func jsToolCatalog(tools []llm.Tool) map[string]any {
	catalog := map[string]any{}
	for _, tool := range tools {
		if tool == nil {
			continue
		}
		schema := tool.Schema()
		if schema.Name == "" {
			continue
		}
		catalog[schema.Name] = map[string]any{
			"name":         schema.Name,
			"description":  schema.Description,
			"input_schema": schema.InputSchema,
			"strict":       schema.Strict,
		}
	}
	return catalog
}

func (e *QuickJSExecutor) callTool(ctx context.Context, req JSRequest, tools []llm.Tool, name string, inputJSON string) (any, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, fmt.Errorf("tool name is required (call as tools.<name>(args))")
	}
	var tool llm.Tool
	available := make([]string, 0, len(tools))
	for _, t := range tools {
		if t == nil {
			continue
		}
		schemaName := t.Schema().Name
		if schemaName == "" {
			continue
		}
		available = append(available, schemaName)
		if schemaName == name && tool == nil {
			tool = t
		}
	}
	if tool == nil {
		return nil, fmt.Errorf("tool %q is not available in the action space (have: %s)", name, strings.Join(available, ", "))
	}

	input, err := decodeToolInput(name, inputJSON)
	if err != nil {
		return nil, err
	}

	// Generate a unique call ID so nested tool calls can be surfaced
	// and permission requests can match the invocation.
	callID := "js_" + name + "_" + newRandomID(8)
	inputJSONBytes, _ := json.Marshal(input)
	call := llm.ToolUse{
		ID:    callID,
		Name:  name,
		Input: inputJSONBytes,
	}

	if req.NestedToolUpdate != nil {
		if err := req.NestedToolUpdate(ctx, call, "start", nil); err != nil {
			return nil, err
		}
	}

	toolCtx := req.ToolContext
	if toolCtx.Context == nil {
		toolCtx.Context = ctx
	}
	toolCtx = withToolRuntimeVars(toolCtx, req.ScopeSessionID, req.ScopeActorID, req.ScopeRunID, req.HomeDir)
	if toolCtx.WorkingDirectory == "" {
		toolCtx.WorkingDirectory = req.WorkingDirectory
	}

	if req.PermissionHook != nil {
		approved, err := req.PermissionHook(ctx, req, tool, call)
		if err != nil {
			if req.NestedToolUpdate != nil {
				_ = req.NestedToolUpdate(ctx, call, "end", &llm.ToolResult{Error: err, ErrorStr: err.Error()})
			}
			return nil, err
		}
		if !approved {
			err := fmt.Errorf("tool call %s rejected", name)
			if req.NestedToolUpdate != nil {
				_ = req.NestedToolUpdate(ctx, call, "end", &llm.ToolResult{Error: err, ErrorStr: err.Error()})
			}
			return nil, err
		}
	}

	result := tool.Execute(toolCtx, input)
	if result.Error != nil && result.ErrorStr == "" {
		result.ErrorStr = result.Error.Error()
	}

	mc := llm.MessageContent(result.Output)
	var value any
	if result.StructuredOutput != nil {
		value = result.StructuredOutput
	} else {
		text := strings.TrimSpace((&mc).Text())
		if text != "" {
			if err := json.Unmarshal([]byte(text), &value); err != nil {
				value = text
			}
		}
	}

	if req.ForwardedContent != nil {
		req.ForwardedContent.Add(call.ID, result.Output, req.ToolContext.ModelConfig)
		if len(result.Metadata.Content) > 0 {
			var blocks []llm.ContentBlock
			for _, content := range result.Metadata.Content {
				if content.Content != nil {
					blocks = append(blocks, content.Content.Block)
				}
			}
			req.ForwardedContent.Add(call.ID, blocks, req.ToolContext.ModelConfig)
		}
	}

	text := (&mc).Text()
	value, text = spillNestedToolOutputIfNeeded(req.HomeDir, req.ScopeSessionID, req.ScopeActorID, req.ScopeRunID, call.ID, call.Name, text, value, result)
	if text != (&mc).Text() {
		result.Output = []llm.ContentBlock{llm.NewTextContentBlock(text)}
		result.StructuredOutput = value
		result.Metadata.Content = []llm.ToolCallContent{llm.NewToolCallContentBlock(llm.NewTextContentBlock(text))}
	}

	if req.NestedToolUpdate != nil {
		if err := req.NestedToolUpdate(ctx, call, "end", &result); err != nil {
			return nil, err
		}
	}

	if result.ErrorStr != "" {
		if result.StructuredOutput != nil {
			return map[string]any{
				"error":             result.ErrorStr,
				"structured_output": value,
				"result":            value,
			}, nil
		}
		return nil, fmt.Errorf("%s", result.ErrorStr)
	}

	return map[string]any{
		"output":            jsVisibleToolOutput(result.Output),
		"text":              text,
		"structured_output": value,
		"result":            value,
		"metadata":          jsVisibleToolMetadata(result.Metadata),
		"variables":         result.Variables,
		"usage":             result.Usage,
		"final":             result.Final,
	}, nil
}

func jsVisibleToolOutput(blocks []llm.ContentBlock) []map[string]any {
	if len(blocks) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		out = append(out, jsVisibleContentBlock(block))
	}
	return out
}

func jsVisibleContentBlock(block llm.ContentBlock) map[string]any {
	switch block.Type {
	case llm.ContentBlockTypeText:
		return map[string]any{"type": string(block.Type), "text": block.Text}
	case llm.ContentBlockTypeImage:
		mediaType := ""
		if block.Image != nil {
			mediaType = block.Image.MediaType
		}
		return map[string]any{
			"type": string(block.Type),
			"image": map[string]any{
				"media_type": mediaType,
				"omitted":    true,
				"reason":     "forwarded as visual tool-result content",
			},
		}
	case llm.ContentBlockTypeAudio, llm.ContentBlockTypeVideo, llm.ContentBlockTypeDocument:
		return map[string]any{
			"type":    string(block.Type),
			"omitted": true,
			"reason":  "forwarded as rich tool-result content",
		}
	default:
		return map[string]any{
			"type":    string(block.Type),
			"omitted": true,
		}
	}
}

func jsVisibleToolMetadata(meta llm.ToolMetadata) map[string]any {
	out := map[string]any{}
	if meta.Title != "" {
		out["title"] = meta.Title
	}
	if meta.Kind != "" {
		out["kind"] = meta.Kind
	}
	if len(meta.Locations) > 0 {
		out["locations"] = meta.Locations
	}
	if len(meta.Content) > 0 {
		out["content_omitted"] = len(meta.Content)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// decodeToolInput is intentionally lenient. The model frequently passes a
// non-object (string, number, or undefined) and the strict map decode error
// is opaque ("cannot unmarshal string into Go value of type
// map[string]interface {}"). We coerce sensibly when we can and return a
// clear, model-actionable error otherwise.
func decodeToolInput(name string, inputJSON string) (map[string]any, error) {
	trimmed := strings.TrimSpace(inputJSON)
	if trimmed == "" || trimmed == "null" || trimmed == "undefined" {
		return map[string]any{}, nil
	}
	var raw any
	if err := json.Unmarshal([]byte(trimmed), &raw); err != nil {
		return nil, fmt.Errorf("invalid input for %s: %w (received: %s)", name, err, truncateForError(trimmed))
	}
	switch v := raw.(type) {
	case map[string]any:
		return v, nil
	case nil:
		return map[string]any{}, nil
	default:
		return nil, fmt.Errorf("tools.%s expects an object argument like tools.%s({...}); received %s — wrap your argument in {} or pass {} when no input is needed", name, name, describeJSValueType(raw))
	}
}

func describeJSValueType(v any) string {
	switch v.(type) {
	case string:
		return "a string"
	case float64, int, int64:
		return "a number"
	case bool:
		return "a boolean"
	case []any:
		return "an array"
	default:
		return fmt.Sprintf("%T", v)
	}
}

func truncateForError(s string) string {
	if len(s) <= 80 {
		return s
	}
	return s[:80] + "…"
}

func jsonString(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func jsValueResult(value *qjs.Value) (JSResult, error) {
	if value.IsUndefined() {
		return JSResult{Text: "undefined"}, nil
	}
	if value.IsNull() {
		return JSResult{Text: "null"}, nil
	}
	if value.IsString() {
		text := value.String()
		if json.Valid([]byte(text)) {
			return JSResult{JSON: text}, nil
		}
		return JSResult{Text: text}, nil
	}
	if jsonText, err := value.JSONStringify(); err == nil && jsonText != "" {
		return JSResult{JSON: jsonText}, nil
	}
	return JSResult{Text: value.String()}, nil
}

func jsPrelude() string {
	return `
const __tool = async (name, args) => {
  const payload = (args === undefined || args === null) ? "" : JSON.stringify(args);
  return JSON.parse(globalThis.__callTool(name, payload));
};
const __call = async (name, args) => {
  const result = await __tool(name, args);
  if (result && result.final) {
    return {
      __ai_sdk_final: true,
      summary: result.structured_output && result.structured_output.summary,
      output: result.structured_output && Object.prototype.hasOwnProperty.call(result.structured_output, "output")
        ? result.structured_output.output
        : result.structured_output,
      structured_output: result.structured_output
    };
  }
  return result && Object.prototype.hasOwnProperty.call(result, "structured_output") && result.structured_output != null
    ? result.structured_output
    : result;
};
const __toolCatalog = Object.freeze(JSON.parse(globalThis.__toolCatalogJSON || "{}"));
const __toolTarget = {};
for (const name of Object.keys(__toolCatalog)) {
  Object.defineProperty(__toolTarget, name, {
    enumerable: true,
    configurable: false,
    value: async (args) => __call(name, args)
  });
}
Object.defineProperty(__toolTarget, "list", {
  enumerable: false,
  value: () => Object.values(__toolCatalog)
});
Object.defineProperty(__toolTarget, "schema", {
  enumerable: false,
  value: (name) => __toolCatalog[name] || null
});

globalThis.env = Object.freeze(globalThis.__env || {});
globalThis.input = Object.freeze(globalThis.__input || {});
globalThis.cwd = globalThis.__cwd || "";
globalThis.tools = new Proxy(__toolTarget, {
  get(target, name) {
    if (typeof name === "symbol") return target[name];
    if (name in target) return target[name];
    const available = Object.keys(__toolCatalog).join(", ");
    throw new Error("tools." + String(name) + " is not available in the action space. Available tools: " + available);
  },
});
globalThis.vars = JSON.parse(globalThis.__varsJSON || "{}");

const __stringify = (v) => {
  if (v === undefined) return "undefined";
  if (v === null) return "null";
  if (typeof v === "string") return v;
  if (v instanceof Error) return v.stack || v.message;
  try {
    return JSON.stringify(v, null, 2);
  } catch (e) {
    return String(v);
  }
};

globalThis.console = {
  log: (...args) => globalThis.__log(...args.map(__stringify)),
  warn: (...args) => globalThis.__log("[warn]", ...args.map(__stringify)),
  error: (...args) => globalThis.__log("[error]", ...args.map(__stringify)),
};
`
}
