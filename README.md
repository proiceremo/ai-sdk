# ai-sdk

[![Go Reference](https://pkg.go.dev/badge/github.com/proiceremo/ai-sdk.svg)](https://pkg.go.dev/github.com/proiceremo/ai-sdk)

Unified Go SDK for LLM provider clients, multimodal content, streaming, tool schemas, embeddings, and pure retrieval primitives.

- **Module path:** `github.com/proiceremo/ai-sdk`
- **Go version:** 1.25+

**Design principle:** `ai-sdk` owns reusable LLM, tool, agent-definition, and optional local-runtime primitives. It avoids ProAgent-specific session/actor/run assumptions; any persistence it provides is opt-in, path-configured, and keyed by generic scope IDs.

---

## Table of Contents

- [What it provides](#what-it-provides)
- [Installation](#installation)
- [Core Types](#core-types)
- [Messages and Content](#messages-and-content)
- [Provider System](#provider-system)
- [Streaming](#streaming)
- [Usage Tracking](#usage-tracking)
- [OAuth Authentication](#oauth-authentication)
- [Tool System](#tool-system)
- [Tool Registry](#tool-registry)
- [Code Execution](#code-execution)
- [RLM Primitives](#rlm-primitives)
- [Embeddings](#embeddings)
- [Content I/O](#content-io)
- [MCP Management](#mcp-management)
- [Skills](#skills)
- [Testing](#testing)

---

## What it provides

- **Multi-provider LLM client** — OpenAI Codex (Responses API + OAuth), Anthropic Messages (OAuth), Google Gemini / Vertex, and a long tail of OpenAI- and Anthropic-compatible third-party hosts (OpenRouter, Fireworks, DeepInfra, DeepSeek, Moonshot, NVIDIA NIM, MiniMax, Qwen, etc.) — all behind one `Client` interface
- **Streaming + non-streaming completions** with full multimodal usage tracking — input/output token counts, cache read/write, reasoning sub-counts, per-modality audio / image / video / document sub-counts
- **OAuth authentication** (`oauthx`) — authorization-code + Anthropic flows with refresh-token persistence under `<PRO_HOME>/oauth/{providers,mcp}/`
- **Embeddings** (one vector per request) on supporting providers
- **Tool system** — typed JSON-schema tools (`Tool` interface) plus a `tools.NewGenericTool` generic wrapper and a JSON-config-driven `ToolRegistry`
- **Content I/O** — URI / file / data-URI helpers that produce `ContentBlock` values
- **RLM primitives** — deterministic indexing, BM25-style retrieval, and bounded lambda-analysis
- **Agent definitions** — small recipe structs and registries for model/tool prompts, without persistence or orchestration policy
- **QuickJS sandbox** — model-generated JavaScript execution with action-space tool access
- **Plan, terminal, MCP, and skills tools** — reusable tool implementations with dependency-injected storage, executors, and event sinks

---

## Installation

```bash
go get github.com/proiceremo/ai-sdk
```

The canonical import path is `github.com/proiceremo/ai-sdk`. Sub-packages live at `github.com/proiceremo/ai-sdk/{tools,agent,plan,terminal,mcp,skills,oauthx,codeexec,contentio,rlm,providers/...}`.

In Go source files the convention across this repo is to alias the root package as `llm`, which is what every example below assumes:

```go
import (
    llm "github.com/proiceremo/ai-sdk"
    "github.com/proiceremo/ai-sdk/tools"
    "github.com/proiceremo/ai-sdk/providers/anthropic"
)
```

---

## Core Types

### Messages

```go
type Message struct {
    Role       MessageRole
    Content    MessageContent // []ContentBlock
    Timestamp  time.Time
    Metadata   map[string]any
    StopReason StopReason
    Usage      *Usage
}

type MessageRole string

const (
    MessageRoleUser      MessageRole = "user"
    MessageRoleAssistant MessageRole = "assistant"
)
```

There is **no** `system` role. System prompts are passed out-of-band via `InferenceParams.SystemPrompt` — every supported provider lowers that into the right place (Anthropic's `system` field, OpenAI's leading system message, Google's system instruction).

### Content Blocks

```go
type ContentBlock struct {
    Type       ContentBlockType
    Text       string          // for text / thinking blocks (thinking uses Thinking)
    Thinking   string          // chain-of-thought (Anthropic, gpt-5 reasoning)
    Redacted   string          // provider-redacted thinking payload
    Signature  string          // signed-thinking signature (Anthropic)
    Image      *ImageSource
    Document   *DocumentSource
    Audio      *AudioSource
    Video      *VideoSource
    ToolUse    *ToolUse        // assistant requesting a tool call
    ToolOutput *ToolOutput     // result of a tool call
    Ephemeral  bool            // strip from history once consumed
}

type ContentBlockType string

const (
    ContentBlockTypeText             ContentBlockType = "text"
    ContentBlockTypeImage            ContentBlockType = "image"
    ContentBlockTypeDocument         ContentBlockType = "document"
    ContentBlockTypeAudio            ContentBlockType = "audio"
    ContentBlockTypeVideo            ContentBlockType = "video"
    ContentBlockTypeToolUse          ContentBlockType = "tool_use"
    ContentBlockTypeToolResult       ContentBlockType = "tool_result"
    ContentBlockTypeThinking         ContentBlockType = "thinking"
    ContentBlockTypeRedactedThinking ContentBlockType = "redacted_thinking"
)
```

Tool results live on `ContentBlock.ToolOutput` (a `*ToolOutput`), not on a separate `ToolResult` field. `ToolOutput.Output` is itself `MessageContent`, so tool calls can return multimodal payloads (text + image, etc.).

### Tool Abstraction

```go
type Tool interface {
    Schema() ToolSchema
    Execute(ctx ToolContext, input map[string]any) ToolResult
}

type ToolSchema struct {
    Name         string
    Description  string
    InputSchema  JSONSchema
    OutputSchema JSONSchema
    Strict       bool
}

// ToolContext embeds context.Context, so it IS a context.Context —
// pass it directly to anything that takes one.
type ToolContext struct {
    context.Context

    WorkingDirectory string
    Vars             map[string]any
    ModelConfig      *ModelConfig
    Permission       ToolPermissionHandler

    Emit             func(event any)
    NestedToolUpdate func(ctx context.Context, call ToolUse, status string, result *ToolResult) error
}

type ToolResult struct {
    Output           []ContentBlock
    StructuredOutput any            // typed return value (used by finish-style tools)
    Usage            *Usage         // tool's own token/cost usage if it ran inference
    Metadata         ToolMetadata
    Variables        map[string]any // surfaced back as ToolContext.Vars on the next call
    Wait             *AgentWait     // pause the agent loop pending external input
    Final            bool           // ends the turn (finish-style tools set this)
    Error            error          // not serialised; populates ErrorStr
    ErrorStr         string
}
```

Tools that need permission gating implement the optional `GuardedTool` interface (adds `GetPermissions(ctx, input) ([]PermissionGuard, error)`); the host calls the `Permission` handler before `Execute`.

---

## Messages and Content

### Creating Messages

```go
msg := llm.Message{
    Role: llm.MessageRoleUser,
    Content: llm.MessageContent{
        llm.NewTextContentBlock("Hello, world"),
    },
}
```

### Multimodal Content

```go
msg := llm.Message{
    Role: llm.MessageRoleUser,
    Content: llm.MessageContent{
        llm.NewTextContentBlock("Describe this image"),
        llm.NewImageContentBlockFromURL("https://example.com/image.png", "image/png"),
        // or from a base64 payload:
        // llm.NewImageContentBlockFromBase64(b64, "image/png"),
    },
}
```

Sibling helpers in the root package: `NewTextContentBlock`, `NewImageContentBlockFromURL`, `NewImageContentBlockFromBase64`, `NewEphemeralTextContentBlock`. For loading from disk or arbitrary URIs, use the `contentio` package (see [Content I/O](#content-io)).

---

## Provider System

### Resolving a Client

```go
registry := llm.DefaultRegistry() // *llm.Registry (alias for *llm.Resolver)
client, model, err := registry.Resolve(ctx, "anthropic-oauth/claude-opus-4-7")
if err != nil {
    log.Fatal(err)
}
// client implements llm.Client; model is the resolved llm.ModelConfig
// (pricing, modalities, capabilities). Most code threads `model` through
// to ToolContext.ModelConfig so tools can read the same metadata.
```

`Resolve` accepts a catalog ID (`"provider/model"`) or a typed `ModelConfig`. For embeddings use `ResolveEmbedding`, which returns an `EmbeddingCapable` plus the `EmbeddingModelConfig`.

### Catalog ID prefixes shipped in `DefaultRegistry`

The default catalog is opinionated — it ships the routes the maintainers actually use rather than a bare provider/model list. The underlying API formats (`APIFormatOpenAI`, `APIFormatOpenAICodex`, `APIFormatAnthropic`, `APIFormatGoogle`) are reused across many prefixes:

| Prefix | API format | Auth | Notes |
|---|---|---|---|
| `openai-codex/...` | OpenAI Codex (Responses API) | OAuth (`/login codex`) | The reasoning-models path — surfaces `reasoning_tokens` sub-count |
| `anthropic-oauth/...` | Anthropic Messages | OAuth (`/login claude`) | Native Anthropic auth via the Claude OAuth flow |
| `google/...`, `google-vertex/...` | Google Gemini | API key / Vertex | |
| `openrouter/...` | OpenAI-compatible | API key | Aggregator route |
| `fireworks/...`, `deepinfra/...`, `deepseek/...`, `moonshot/...`, `nvidia-nim/...`, `minimax/...`, `qwen-plan/...` | OpenAI-compatible | API key | Third-party hosts speaking the OpenAI chat-completions format |
| `llamacpp-anthropic/...`, `opencode-anthropic/...`, `xiaomi-anthropic/...`, `zenmux-anthropic/...` | Anthropic Messages | varies | Hosts re-exposing the Anthropic protocol |
| `opencode-openai/...`, `xiaomi-openai/...` | OpenAI chat completions | varies | |
| `zenmux-vertex/...` | Google Vertex | API key | |

You can register your own `ProviderConfig` / `ModelConfig` entries via `Registry.WithProviders` / `WithModels` to extend the catalog. To use plain "openai/" or a bare "anthropic/" prefix, supply your own `ProviderConfig` — they aren't in the default set.

OAuth-backed providers read tokens from `<PRO_HOME>/oauth/providers/<provider>.json` automatically (see [OAuth Authentication](#oauth-authentication)). Codex (`openai-codex/...`) is the only path that surfaces the `reasoning_tokens` sub-count out of the box.

### Provider Interface

```go
type Client interface {
    CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error)
    CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error)
}

// Optional flag interface — providers that can't stream return false
// from SupportsStreaming(), in which case CreateCompletionStream
// typically errors. Use this to branch in caller code.
type StreamingCapable interface {
    SupportsStreaming() bool
}

// Optional, served alongside Client when the model has embedding support.
type EmbeddingCapable interface {
    CreateEmbeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error)
}
```

### Inference Parameters

```go
type InferenceParams struct {
    Model        string
    SystemPrompt string         // system prompt is a top-level param, not a role
    Tools        []Tool         // typed Tool instances, not config structs
    ToolChoice   *ToolChoice

    // Numerics are pointers so "unset" survives the round-trip — the
    // provider falls back to its own default rather than 0.
    Temperature *float64
    MaxTokens   *int
    TopP        *float64
    TopK        *int

    Thinking       *ThinkingParams      // explicit reasoning budget / level
    WebSearch      *WebSearchToolParams // hosted web-search tool toggle
    ResponseFormat *ResponseFormat      // structured outputs (json_object + schema)
    ServiceTier    *string              // priority / flex / batch on supporting providers

    // CacheRetention: "" / "default" (5 min) or "long" (1h, Anthropic).
    // Other providers ignore this. See doc comment in types.go.
    CacheRetention string
}
```

Streaming is not a flag on `InferenceParams` — call `CreateCompletionStream` to stream, `CreateCompletion` to buffer. Same params struct for both.

---

## Streaming

### The Stream interface

```go
type Stream interface {
    Recv(ctx context.Context) (*StreamEvent, error)
    Current() *Message  // accumulated message-so-far snapshot
    Close() error
}

type StreamEvent struct {
    Type     StreamEventType
    Delta    MessageDelta  // incremental content / stop / usage
    Snapshot Message       // full message at this point in the stream
}

const (
    EventTypeMessageStart StreamEventType = "message_start"
    EventTypeContentStart StreamEventType = "content_start"
    EventTypeContentDelta StreamEventType = "content_delta"
    EventTypeContentEnd   StreamEventType = "content_end"
    EventTypeMessageEnd   StreamEventType = "message_end"
)
```

### Creating a Stream

```go
mt := 4096
stream, err := client.CreateCompletionStream(ctx, messages, llm.InferenceParams{
    Model:        "anthropic/claude-sonnet-4",
    SystemPrompt: "You are a careful assistant.",
    MaxTokens:    &mt,
})
if err != nil {
    log.Fatal(err)
}
defer stream.Close()
```

### Consuming Events

`Recv` returns one `*StreamEvent` per call, with `io.EOF` at end-of-stream. Use `Delta.Content` for incremental blocks and `Delta.Usage` for the running token count — there are no separate `StreamText` / `StreamToolUse` concrete types.

```go
for {
    event, err := stream.Recv(ctx)
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    switch event.Type {
    case llm.EventTypeContentDelta:
        for _, block := range event.Delta.Content {
            if block.Type == llm.ContentBlockTypeText {
                fmt.Print(block.Text)
            }
        }
    case llm.EventTypeMessageEnd:
        if event.Delta.Usage != nil {
            fmt.Printf("\nTokens: %+v\n", event.Delta.Usage.Totals)
        }
    }
}

// Or, if you only care about the final message:
final := stream.Current()
```

---

## Usage Tracking

Every completion (streaming or not) yields a `*llm.Usage` whose `Totals` field is a `TokenUsage` with full multimodal accounting. Provider-specific extensions are normalised so downstream code reads the same shape regardless of which client produced it.

```go
type TokenUsage struct {
    InputTokens              int
    OutputTokens             int
    TotalTokens              int

    // Cache accounting (Anthropic billed separately; OpenAI/OpenRouter
    // bundled into InputTokens; CacheBilledSeparately disambiguates).
    CacheCreationInputTokens int
    CacheReadInputTokens     int
    CacheBilledSeparately    bool

    // Reasoning models (o-series, GPT-5, Codex) report this as a
    // SUB-count of OutputTokens — already included, surfaced for
    // visibility ("X% of output was hidden chain-of-thought").
    ReasoningOutputTokens    int

    // Multimodal sub-counts. AudioTokens populated by gpt-4o-audio;
    // ImageTokens / VideoTokens / DocumentTokens reserved for providers
    // that report them. nil when the response has no breakdown.
    InputTokensDetails              *UsageTokenDetails
    OutputTokensDetails             *UsageTokenDetails
    CacheCreationInputTokensDetails *UsageTokenDetails
    CacheReadInputTokensDetails     *UsageTokenDetails

    ToolUseInputTokens       int
    ServerToolUse            map[string]int
}

type UsageTokenDetails struct {
    TextTokens, AudioTokens, ImageTokens, VideoTokens, DocumentTokens int
}
```

### Per-provider field sources

| Field | OpenAI Chat | OpenAI Codex | Anthropic |
|---|---|---|---|
| `CacheReadInputTokens` | `prompt_tokens_details.cached_tokens` | `input_tokens_details.cached_tokens` (subtracted from input) | `cache_read_input_tokens` |
| `CacheCreationInputTokens` | `prompt_tokens_details.cache_write_tokens` (OpenRouter ext) | n/a | `cache_creation_input_tokens` |
| `ReasoningOutputTokens` | `completion_tokens_details.reasoning_tokens` | `output_tokens_details.reasoning_tokens` | n/a |
| `InputTokensDetails.AudioTokens` | `prompt_tokens_details.audio_tokens` | `input_tokens_details.audio_tokens` | n/a |
| `OutputTokensDetails.AudioTokens` | `completion_tokens_details.audio_tokens` | `output_tokens_details.audio_tokens` | n/a |

`TokenUsage.Add(other)` merges two usages including all sub-counts; aggregation in benchmarks and dashboards goes through this path so nothing silently drops.

---

## OAuth Authentication

The `oauthx` sub-package (`github.com/proiceremo/ai-sdk/oauthx`) implements OAuth flows for model providers (OpenAI Codex, Anthropic Claude) and is reused by the `mcp` sub-package for MCP server tokens. All credentials live under a single root so they're easy to seed, back up, and reason about:

```
$PRO_HOME/oauth/
  providers/<provider>.json   # ai-sdk model providers
  mcp/<server>.json           # MCP server tokens
```

`PRO_HOME` resolves via the env var (containerised runs) or `~/.pro` (host runs). `oauthx.Root()`, `oauthx.ProvidersStore()`, and `oauthx.MCPStore()` expose those paths.

### Logging in

```go
// CLI flow — opens a browser and prints the URL to stderr.
creds, err := oauthx.LoginAuthorizationCode(ctx, oauthx.CodexConfig("openai-codex"), oauthx.LoginOptions{})

// Anthropic uses a separate flow with refresh-token rotation.
creds, err := oauthx.LoginAnthropic(ctx, "anthropic-oauth", oauthx.LoginOptions{})

// UI-driven flow — stream the URL to the user instead of opening
// a local browser (used by the ACP adapter's /login slash command).
creds, err := oauthx.LoginAuthorizationCode(ctx, oauthx.CodexConfig("openai-codex"), oauthx.LoginOptions{
    OnAuthURL: func(u string) { sendToClient(u) },
})
```

### Resolving a token source

```go
source, err := oauthx.TokenSource(ctx, oauthx.CodexConfig("openai-codex"))
// nil source means no stored credentials; otherwise calls to source.Token()
// refresh on demand AND persist new tokens back to the FileStore.
```

### Status, logout

```go
store := oauthx.ProvidersStore()
exists := store.Has("openai-codex")
creds, err := store.Load("openai-codex")  // includes expiry, refresh-token presence
_ = store.Delete("openai-codex")           // idempotent — no error if already gone
```

### ACP slash commands

Apps embedding `proagent/v2/acp` automatically advertise `/login`, `/logout`, and `/oauth-status` as session/update::available_commands_update notifications. Users authenticate without leaving the chat surface:

```
/login codex
/login claude
/oauth-status
/logout codex
```

---

## Tool System

### Implementing a Tool — typed, with `tools.NewGenericTool`

This is the idiomatic path. Define an input struct (with `jsonschema` tags) and an executor; `NewGenericTool` reflects the schema for you.

```go
type echoInput struct {
    Message string `json:"message" jsonschema:"description=Text to echo back,minLength=1"`
}

tool := tools.NewGenericTool(
    "echo",
    "Echo a message back to the caller",
    func(ctx llm.ToolContext, in echoInput) llm.ToolResult {
        return llm.ToolResult{
            Output: []llm.ContentBlock{llm.NewTextContentBlock("Echo: " + in.Message)},
        }
    },
)
```

### Implementing a Tool — by hand

If you need full control, implement the `Tool` interface directly:

```go
type myTool struct{}

func (myTool) Schema() llm.ToolSchema {
    return llm.ToolSchema{
        Name:        "my_tool",
        Description: "Does something useful",
        InputSchema: llm.JSONSchema{
            "type": "object",
            "properties": map[string]any{
                "input": map[string]any{"type": "string"},
            },
            "required": []string{"input"},
        },
    }
}

func (myTool) Execute(ctx llm.ToolContext, input map[string]any) llm.ToolResult {
    val, _ := input["input"].(string)
    return llm.ToolResult{
        Output: []llm.ContentBlock{llm.NewTextContentBlock("Result: " + val)},
    }
}
```

### Built-in tool constructors

All return `llm.Tool` instances ready to drop into `InferenceParams.Tools`.

| Constructor | Package | What it gives the model |
|---|---|---|
| `tools.NewReadTool(opts)` | `tools` | `read` — read a file with offset/limit/range |
| `tools.NewReadRangeTool(opts)` | `tools` | `read_range` — random-access byte ranges |
| `tools.NewEditTool(opts)` | `tools` | `edit` — atomic file edits (replace/insert/write) with SHA-256 guards |
| `tools.NewWriteTool(opts)` | `tools` | `write` — full-file rewrite |
| `tools.NewGenericTool[T](name, desc, fn)` | `tools` | typed wrapper around any executor |
| `tools.NewFinishTool[T](opts)` | `tools` | typed finish tool — sets `ToolResult.Final = true` and serialises a `T` as the answer |
| `llm.NewFinishTool(cfg)` | root | dynamic-schema finish tool driven by `FinishConfig` |
| `codeexec.NewTool(executor, actionTools)` | `codeexec` | `js_execute` — QuickJS sandbox with nested tool calls |
| `plan.NewTool(sink)` | `plan` | `plan` — update structured task plans |
| `terminal.NewTool(manager)` | `terminal` | `terminal` — durable shell sessions |
| `mcp.NewTool(manager)` | `mcp` | `mcp` — list/call tools on configured MCP servers |
| `skills.NewTool(loader)` | `skills` | `skills` — list / read / create local `SKILL.md` bundles |

---

## Tool Registry

The registry is for runtimes that build tools dynamically from JSON config (e.g. an agent definition file). Factories receive the raw config as `json.RawMessage`.

```go
registry := llm.NewToolRegistry()
registry.MustRegister("echo", func(ctx context.Context, build llm.ToolBuildContext, raw json.RawMessage) (llm.Tool, error) {
    var cfg struct{ Prefix string `json:"prefix"` }
    if len(raw) > 0 {
        if err := json.Unmarshal(raw, &cfg); err != nil {
            return nil, err
        }
    }
    return tools.NewGenericTool("echo", "Echo with prefix",
        func(ctx llm.ToolContext, in echoInput) llm.ToolResult {
            return llm.ToolResult{Output: []llm.ContentBlock{
                llm.NewTextContentBlock(cfg.Prefix + in.Message),
            }}
        }), nil
})

built, err := registry.BuildTools(ctx, buildCtx, []llm.ToolConfig{
    {ID: "echo", Config: json.RawMessage(`{"prefix": "> "}`)},
})
```

---

## Code Execution

The QuickJS executor runs model-generated JavaScript in an isolated sandbox. Most callers don't drive it directly — they expose it as a tool via `codeexec.NewTool` and let the agent loop call `js_execute`.

```go
executor := codeexec.NewQuickJSExecutor()
result, err := executor.Execute(ctx, codeexec.JSRequest{
    Code:    `return {sum: 1 + 1};`,
    Input:   map[string]any{"x": 41},   // surfaces as `input` global in JS
    Timeout: 10 * time.Second,
    Tools:   actionSpaceTools,           // []llm.Tool exposed as `tools` global
})
// result.JSON / result.Text / result.Logs / result.Vars
```

As a tool the model invokes from the loop:

```go
jsTool := codeexec.NewTool(executor, actionSpaceTools)
```

Inside the sandbox the `tools` global exposes every passed-in `llm.Tool`:

```javascript
const r = await tools.read({path: "main.go"});
return r;
```

Key constraints inside the sandbox:
- No `require`, `import`, `fetch`, or direct filesystem access
- All external work goes through the `tools` action-space
- Inline string/template/single-quote literals are capped at 4 KB — move larger payloads into `js_execute.input` and reference them via the `input` global
- Default execution timeout is 3 minutes; override with `JSRequest.Timeout` or the tool's `timeout_ms` param

---

## RLM Primitives

The `rlm` sub-package (`github.com/proiceremo/ai-sdk/rlm`) provides pure retrieval and lambda planning primitives with no agent/runtime dependency.

### Document type

```go
type Document struct {
    ID       string
    Title    string
    URI      string
    Text     string
    Metadata any
}
```

Chunking is internal — callers pass whole documents in and get bounded `QueryResult` chunks back.

### Kernel API

```go
kernel := rlm.NewKernel()

// Index loose documents:
_, err := kernel.Index(docs, /*replace=*/ true)

// Or index a workspace on disk (respects include/exclude globs):
_, err = kernel.IndexWorkspace(ctx, "/path/to/repo", rlm.WorkspaceOptions{
    MaxFiles:     2000,
    MaxFileBytes: 256 * 1024,
})

// Retrieve top-K bounded chunks:
results, err := kernel.Retrieve("oauth refresh-token flow", 5)
for _, r := range results {
    fmt.Printf("%s:%d-%d (score=%.3f)\n%s\n\n", r.DocumentID, r.Start, r.End, r.Score, r.Text)
}

// Higher-level helpers:
res, err := kernel.Analyze(query, docs, /*topK=*/ 5, /*chunkSize=*/ 1500)
res, err = kernel.LambdaAnalyze(query, docs, 5, 1500)
```

`QueryResult` carries `DocumentID`, `ChunkIndex`, `Start`/`End`, `Score`, `Text`, and any propagated metadata — retrieval returns bounded slices, not whole documents, so the caller controls the context budget.

### Tool modes (via `mcp`-style `Input`)

When wrapped as a tool, the `Input.Mode` field selects: `index`, `index_workspace`, `retrieve`, `analyze`, `lambda_analyze`.

---

## Embeddings

`ResolveEmbedding` returns an `EmbeddingCapable` rather than a `Client`. The request takes `MessageContent` so embedding inputs can themselves be multimodal where the provider supports it; for plain text use `NewTextEmbeddingRequest`.

```go
client, model, err := llm.DefaultRegistry().ResolveEmbedding(ctx, "openai/text-embedding-3-small")
if err != nil {
    log.Fatal(err)
}

req := llm.NewTextEmbeddingRequest(model.ModelID, "Hello world", "Goodbye world")
resp, err := client.CreateEmbeddings(ctx, req)
if err != nil {
    log.Fatal(err)
}
// resp.Embeddings is []float32 (provider-concatenated); resp.Usage gives token counts.
```

Use `llm.NormalizeEmbeddingIfNeeded` or `llm.NormalizeVector` to L2-normalise vectors when the downstream consumer requires it.

---

## Content I/O

`contentio` converts URIs, files, and data-URI strings into `llm.ContentBlock` values ready to drop into a message.

```go
// From any URI — http(s), file path, or data: URI.
// mimeType is optional (nil = sniff from URI / response).
block, err := contentio.NewContentBlockFromURI("https://example.com/diagram.png", nil)

// From a local file (path-only convenience):
block, err = contentio.FileToContentBlock("/home/user/project/README.md")

// From an in-memory base64 / URL payload when you already know the MIME type:
block = contentio.ContentBlockFromMimeTypeBase64("image/png", b64data)
block = contentio.ContentBlockFromMimeTypeURL("application/pdf", "https://example.com/doc.pdf")

// Or just the data-URI string:
dataURI, mime, err := contentio.FileToDataURI("/home/user/diagram.png")
```

There is no `contentio.Load` — pick the helper that matches your source.

---

## MCP Management

The `mcp.Manager` is store-driven: server definitions come from a `ServerStore` you implement (typically backed by a JSON file or a database), and the manager opens / pools connections lazily on demand.

```go
type fileStore struct{ servers []mcp.ServerConfig }
func (s fileStore) ListServers(ctx context.Context, scopeID string) ([]mcp.ServerConfig, error) {
    return s.servers, nil
}

manager := mcp.NewManager(fileStore{servers: []mcp.ServerConfig{{
    Name:    "svelte",
    Command: "npx",
    Args:    []string{"mcp-server-svelte"},
}}})

// Every interaction goes through Execute(scopeID, Input).
out, err := manager.Execute("session-1", mcp.Input{Mode: "list_tools", Server: "svelte"})
// out.Tools is []mcp.ToolInfo

out, err = manager.Execute("session-1", mcp.Input{
    Mode:      "call_tool",
    Server:    "svelte",
    Tool:      "search-docs",
    Arguments: map[string]any{"query": "$state"},
})
```

Or wrap the manager as a tool the model can call directly:

```go
tool := mcp.NewTool(manager) // exposes the full Input/Output surface as a single "mcp" tool
```

OAuth-protected MCP servers persist tokens via `oauthx.MCPStore()` at `<PRO_HOME>/oauth/mcp/<server>.json` — same `FileStore` type the model-provider OAuth flow uses, so refresh + status + cleanup all go through one code path. See [OAuth Authentication](#oauth-authentication) for the unified credentials layout.

---

## Skills

Skills are filesystem-backed `SKILL.md` bundles. The loader walks one or more roots and the optional `ManagedRoot` (where agent-created skills land).

```go
loader := skills.NewLoader(skills.Options{
    Roots:       []string{"~/.agents/skills", "./skills"}, // default root is ~/.agents/skills
    ManagedRoot: "./.skills",
})

// List / search:
list, err := loader.List(skills.Input{Query: "go test"}, /*workingDir=*/ "")

// Read a specific skill:
out, err := loader.Read(skills.Input{Skill: "go-unit-test"})
fmt.Println(out.Name, out.Path)
fmt.Println(out.Content) // raw SKILL.md body

// Expose as a tool for the model:
tool := skills.NewTool(loader)
```

`skills.Input.Mode` ranges over `list`, `search`, `read`, `create`, `patch`, `archive`, `restore` — pick the mode and fill the relevant fields (`Skill`, `Query`, `Content`, `Pinned`). The `tools` helper exposes the same surface as a single `skills` tool.

---

## Agent Definitions And Local Tools

`ai-sdk` keeps these pieces generic:

- `agent.Definition` / `agent.Registry` describe model-facing recipes, tool configs, finish schemas, and prompts.
- `codeexec.NewTool` exposes a JavaScript action space with nested tools and `finish`.
- `plan.NewTool` accepts a sink callback; applications decide whether to persist plan state or emit UI events.
- `terminal.NewManager` and `terminal.NewTool` provide a durable local process primitive keyed by a generic scope ID.
- `mcp.NewManager` / `mcp.NewTool` manage MCP server definitions and calls.
- `skills.NewLoader` / `skills.NewTool` discover and manage `SKILL.md` directories from configured roots.

Application runtimes still own policy: permissions, session manifests, actor histories, UI replay, indexes, and cancellation fan-out.

---

## Testing

```bash
cd ai-sdk
go test ./...
```

Tests cover: core types, provider clients, tool system, registry, embeddings, RLM, content I/O, and code execution.
