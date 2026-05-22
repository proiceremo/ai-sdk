# ai-sdk

Unified Go SDK for LLM provider clients, multimodal content, streaming, tool schemas, embeddings, and pure retrieval primitives.

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

- **Multi-provider LLM client** — OpenAI (chat completions + Codex / Responses API + OAuth), Anthropic (API key + OAuth), Google/Gemini, Fireworks via a single `Client` interface
- **Streaming + non-streaming completions** with full multimodal usage tracking — input/output token counts, cache read/write, reasoning sub-counts, audio sub-counts
- **OAuth authentication** (`oauthx`) — authorization-code + Anthropic flows with refresh-token persistence under `<PRO_HOME>/oauth/{providers,mcp}/`
- **Embeddings** with concurrent batching
- **Tool system** — extensible JSON-schema tools with a typed `Tool` interface and `ToolRegistry`
- **Content I/O** — URL/file content loading helpers
- **RLM primitives** — deterministic indexing, retrieval, and lambda planning
- **Agent definitions** — small recipe structs and registries for model/tool prompts, without persistence or orchestration policy
- **QuickJS sandbox** — model-generated JavaScript execution with action-space tool access
- **Plan, terminal, MCP, and skills tools** — reusable tool implementations with dependency-injected storage, executors, and event sinks

---

## Installation

```bash
go get ai-sdk
```

---

## Core Types

### Messages

```go
type Message struct {
    Role    MessageRole    // system | user | assistant
    Content MessageContent // []ContentBlock
}

type MessageRole string

const (
    MessageRoleSystem    MessageRole = "system"
    MessageRoleUser      MessageRole = "user"
    MessageRoleAssistant MessageRole = "assistant"
)
```

### Content Blocks

```go
type ContentBlock struct {
    Type        ContentBlockType
    Text        string
    Image       *ImageSource
    Audio       *AudioSource
    Video       *VideoSource
    Document    *DocumentSource
    Thinking    *ThinkingBlock
    ToolUse     *ToolUse
    ToolResult  *ToolResult
}

type ContentBlockType string

const (
    ContentBlockTypeText       ContentBlockType = "text"
    ContentBlockTypeImage      ContentBlockType = "image"
    ContentBlockTypeAudio      ContentBlockType = "audio"
    ContentBlockTypeVideo      ContentBlockType = "video"
    ContentBlockTypeDocument   ContentBlockType = "document"
    ContentBlockTypeThinking   ContentBlockType = "thinking"
    ContentBlockTypeToolUse    ContentBlockType = "tool_use"
    ContentBlockTypeToolResult ContentBlockType = "tool_result"
)
```

### Tool Abstraction

```go
type Tool interface {
    Name() string
    Description() string
    InputSchema() JSONSchema
    Execute(ctx ToolContext, input map[string]any) ToolResult
}

type ToolContext struct {
    Context          context.Context
    SessionID        string
    WorkingDirectory string
    ParentAgent      string
    Vars             map[string]any
    ModelConfig      *ModelConfig
    Permission       PermissionHandler
    Emit             func(any)
}

type ToolResult struct {
    Output   []ContentBlock
    Error    error
    ErrorStr string
    Wait     *AgentWait
    Metadata ToolResultMetadata
}
```

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
        llm.NewImageContentBlock(llm.ImageSource{
            URL: "https://example.com/image.png",
        }),
    },
}
```

---

## Provider System

### Resolving a Client

```go
resolver := llm.DefaultRegistry()
client, model, err := resolver.Resolve(ctx, "fireworks/kimi-k2p6-turbo")
if err != nil {
    log.Fatal(err)
}
```

### Supported Catalog IDs

| Catalog ID prefix | Provider | Transport | Streaming | Embeddings | Auth |
|---|---|---|---|---|---|
| `openai/...` | OpenAI | chat completions | yes | yes | API key |
| `openai-codex/...` | OpenAI Codex | Responses API | yes | n/a | OAuth (`/login codex`) |
| `anthropic/...` | Anthropic | messages API | yes | no | API key |
| `anthropic-oauth/...` / `claude-code/...` | Anthropic Claude | messages API | yes | no | OAuth (`/login claude`) |
| `google/...` | Google Gemini | generative API | yes | yes | API key |
| `fireworks/...` | Fireworks | OpenAI-compatible | yes | no | API key |

OAuth-backed catalog IDs read tokens from `<PRO_HOME>/oauth/providers/<provider>.json` automatically (see [OAuth Authentication](#oauth-authentication)). Codex routes through the OpenAI Responses API rather than chat completions and is the only path that supports GPT-5 / o-series reasoning models with the `reasoning_tokens` sub-count surfaced.

### Provider Interface

```go
type Client interface {
    CreateCompletion(ctx context.Context, messages []Message, params InferenceParams) (*Message, error)
}

type StreamingCapable interface {
    SupportsStreaming() bool
    CreateCompletionStream(ctx context.Context, messages []Message, params InferenceParams) (Stream, error)
}
```

### Inference Parameters

```go
type InferenceParams struct {
    Model          string
    Temperature    float64
    MaxTokens      int
    TopP           float64
    StopSequences  []string
    Tools          []ToolConfig
    ToolChoice     ToolChoice
    Stream         bool
}
```

---

## Streaming

### Creating a Stream

```go
stream, err := client.(llm.StreamingCapable).CreateCompletionStream(ctx, messages, llm.InferenceParams{
    Model:     "fireworks/kimi-k2p6-turbo",
    MaxTokens: 4096,
    Stream:    true,
})
if err != nil {
    log.Fatal(err)
}
defer stream.Close()
```

### Consuming Events

```go
for {
    event, err := stream.Next()
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    switch e := event.(type) {
    case llm.StreamText:
        fmt.Print(e.Text)
    case llm.StreamToolUse:
        fmt.Printf("Tool: %s\n", e.Name)
    case llm.StreamUsage:
        fmt.Printf("Tokens: %d\n", e.TotalTokens)
    }
}
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

The `ai-sdk/oauthx` package implements OAuth flows for model providers (OpenAI Codex, Anthropic Claude) and is reused by `ai-sdk/mcp` for MCP server tokens. All credentials live under a single root so they're easy to seed, back up, and reason about:

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

### Implementing a Tool

```go
type MyTool struct{}

func (t MyTool) Name() string        { return "my_tool" }
func (t MyTool) Description() string { return "Does something useful" }
func (t MyTool) InputSchema() llm.JSONSchema {
    return llm.JSONSchema{
        Type: "object",
        Properties: map[string]llm.JSONSchema{
            "input": {Type: "string"},
        },
        Required: []string{"input"},
    }
}
func (t MyTool) Execute(ctx llm.ToolContext, input map[string]any) llm.ToolResult {
    val, _ := input["input"].(string)
    return llm.ToolResult{
        Output: []llm.ContentBlock{llm.NewTextContentBlock("Result: " + val)},
    }
}
```

### Built-in Tools

| Tool | Package | Description |
|------|---------|-------------|
| `read_file` | `ai-sdk/tools/filesystem.go` | Read files with offset/limit/range |
| `edit_file` | `ai-sdk/tools/filesystem.go` | Atomic file edits (replace, insert, write) with SHA-256 guards |
| `finish` | `ai-sdk/finish_tool.go` | Structured finish tool primitive |

---

## Tool Registry

```go
registry := llm.NewToolRegistry()
registry.Register("my_tool", llm.ToolFactory(func(ctx context.Context, buildCtx llm.ToolBuildContext, config map[string]any) (llm.Tool, error) {
    return MyTool{}, nil
}))

tools, err := registry.BuildTools(ctx, buildCtx, configs)
if err != nil {
    log.Fatal(err)
}
```

---

## Code Execution

The QuickJS executor runs model-generated JavaScript in an isolated sandbox.

```go
executor := codeexec.NewQuickJSExecutor()
result, err := executor.Execute(ctx, codeexec.JSRequest{
    Code:    `return {sum: 1 + 1};`,
    Timeout: 10 * time.Second,
})
```

Inside the sandbox, a `tools` object exposes all registered action-space tools:

```javascript
const result = await tools.read_file({path: "main.go"});
return result;
```

Key constraints:
- No `require`, `import`, `fetch`, or filesystem access
- All external work is done through the `tools` action-space
- Large strings should be passed through `js_execute.input`, not inlined in source

---

## RLM Primitives

The `ai-sdk/rlm` package provides pure retrieval and lambda planning primitives with no agent/runtime dependency.

### Document Types

```go
type Document struct {
    ID       string
    Title    string
    Text     string
    URI      string
    Metadata any
}

type Chunk struct {
    ID        string
    DocumentID string
    Text      string
    Offset    int
    Length    int
}
```

### Operations

| Mode | Description |
|------|-------------|
| `index` | Add documents to an in-memory or disk index |
| `retrieve` | BM25-ish lexical retrieval over corpus chunks |
| `analyze` | Run analysis over a document set |
| `lambda_analyze` | Deterministic lambda-RLM planning and execution |

```go
index := rlm.NewIndex()
index.AddDocuments(docs...)
results, err := index.Retrieve(ctx, "query", rlm.RetrieveOptions{TopK: 5})
```

---

## Embeddings

```go
client, _, err := resolver.Resolve(ctx, "openai/text-embedding-3-small")
if err != nil {
    log.Fatal(err)
}

embeddings, err := client.CreateEmbeddings(ctx, []string{
    "Hello world",
    "Goodbye world",
}, llm.EmbeddingParams{})
```

---

## Content I/O

```go
// Load content from a URL or local file
content, err := contentio.Load(ctx, "https://example.com/doc.md")
content, err := contentio.Load(ctx, "/home/user/project/README.md")
```

---

## MCP Management

```go
manager := mcp.NewManager()
err := manager.Connect(ctx, mcp.ServerConfig{
    Name:    "svelte",
    Command: "npx",
    Args:    []string{"mcp-server-svelte"},
})

tools, err := manager.ListTools(ctx, "svelte")
```

OAuth-protected MCP servers persist tokens via `oauthx.MCPStore()` at `<PRO_HOME>/oauth/mcp/<server>.json` — same FileStore type the model-provider OAuth flow uses, so refresh + status + cleanup all go through one code path. See [OAuth Authentication](#oauth-authentication) for the unified credentials layout.

---

## Skills

```go
loader := skills.NewLoader(skills.Options{
    Roots: []string{"~/.pro/skills", "./skills"},
    ManagedRoot: "./.skills",
})

skill, err := loader.Read(skills.Input{Skill: "go-unit-test"})
if err != nil {
    log.Fatal(err)
}
```

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
