package anthropic

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"golang.org/x/oauth2"

	llm "github.com/proiceremo/ai-sdk"
)

// API-key path: a single system block carries cache_control. We don't
// inject the Claude-Code identity prefix here — that's OAuth-only.
func TestBuildCompletionRequestAPIKeyPathCachesSystemPrompt(t *testing.T) {
	c, err := NewAnthropicClient(context.Background(), "sk-ant-test", "")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.buildCompletionRequest(
		[]llm.Message{{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("hi")}}},
		llm.InferenceParams{Model: "claude-test", SystemPrompt: "You are a helpful agent."},
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.System) != 1 {
		t.Fatalf("expected 1 system block on API-key path, got %d", len(req.System))
	}
	if req.System[0].Text != "You are a helpful agent." {
		t.Errorf("system text mismatch: %q", req.System[0].Text)
	}
	if req.System[0].CacheControl == nil || req.System[0].CacheControl.Type != "ephemeral" {
		t.Errorf("API-key path must still cache the system prompt; got CacheControl=%+v", req.System[0].CacheControl)
	}
	// No identity prefix on the API-key path.
	if strings.Contains(req.System[0].Text, "Claude Code") {
		t.Errorf("identity prefix leaked into API-key path system prompt")
	}
}

// OAuth path: identity prefix lands as the FIRST system block; user
// prompt is the SECOND block; cache_control lives on the LAST block so
// the cached prefix covers BOTH. This mirrors pi's behaviour at
// anthropic.ts:905.
func TestBuildCompletionRequestOAuthPathPrependsIdentityAndCachesLast(t *testing.T) {
	c, err := NewAnthropicClientWithTokenSource(context.Background(),
		oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "sk-ant-oat-test"}), "")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.buildCompletionRequest(
		[]llm.Message{{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("hi")}}},
		llm.InferenceParams{Model: "claude-test", SystemPrompt: "You are a benchmark agent."},
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.System) != 2 {
		t.Fatalf("expected 2 system blocks (identity + user) on OAuth path, got %d", len(req.System))
	}
	if !strings.Contains(req.System[0].Text, "Claude Code") {
		t.Errorf("first system block must be the Claude-Code identity prompt; got %q", req.System[0].Text)
	}
	if req.System[1].Text != "You are a benchmark agent." {
		t.Errorf("second system block must be the user's system prompt; got %q", req.System[1].Text)
	}
	// Cache_control on the LAST (=second) block, NOT the first. Anthropic
	// caches the prefix up to and including the marked block, so marking
	// the tail captures both blocks in one breakpoint.
	if req.System[0].CacheControl != nil {
		t.Errorf("identity block must NOT carry cache_control on its own; only the tail does")
	}
	if req.System[1].CacheControl == nil {
		t.Errorf("tail system block must carry cache_control to cache the full system prefix")
	}
}

// OAuth path with no user system prompt: identity block becomes the
// ONLY system block and carries cache_control itself.
func TestBuildCompletionRequestOAuthPathNoUserPrompt(t *testing.T) {
	c, err := NewAnthropicClientWithTokenSource(context.Background(),
		oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), "")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.buildCompletionRequest(
		[]llm.Message{{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("hi")}}},
		llm.InferenceParams{Model: "claude-test"},
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.System) != 1 {
		t.Fatalf("expected 1 system block (identity only), got %d", len(req.System))
	}
	if req.System[0].CacheControl == nil {
		t.Errorf("lone identity block must carry cache_control")
	}
}

// Last content block of the last message carries cache_control. This is
// the multi-turn cache hook: every new turn reuses the prefix
// {system}+{tools}+{messages[0..n-1]} for free.
func TestBuildCompletionRequestCachesLastMessageBlock(t *testing.T) {
	c, err := NewAnthropicClient(context.Background(), "sk-ant-test", "")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.buildCompletionRequest(
		[]llm.Message{
			{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("first turn")}},
			{Role: llm.MessageRoleAssistant, Content: llm.MessageContent{llm.NewTextContentBlock("ok")}},
			{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("second turn")}},
		},
		llm.InferenceParams{Model: "claude-test", SystemPrompt: "hi"},
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) == 0 {
		t.Fatal("no messages emitted")
	}
	last := req.Messages[len(req.Messages)-1]
	if len(last.Content) == 0 {
		t.Fatal("last message has no content")
	}
	tail := last.Content[len(last.Content)-1]
	if tail.CacheControl == nil || tail.CacheControl.Type != "ephemeral" {
		t.Errorf("last content block of last message must carry ephemeral cache_control; got %+v", tail.CacheControl)
	}
	// Only the TAIL — earlier blocks/messages stay clean.
	for i, m := range req.Messages[:len(req.Messages)-1] {
		for j, b := range m.Content {
			if b.CacheControl != nil {
				t.Errorf("unexpected cache_control on messages[%d].content[%d]", i, j)
			}
		}
	}
}

// Last tool definition carries cache_control as a map field. Tools
// schemas are stable across turns and typically 1-5k tokens, so this is
// the second-largest cache win after the system prompt.
func TestBuildCompletionRequestCachesLastTool(t *testing.T) {
	c, err := NewAnthropicClient(context.Background(), "sk-ant-test", "")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.buildCompletionRequest(
		[]llm.Message{{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("hi")}}},
		llm.InferenceParams{
			Model: "claude-test",
			Tools: []llm.Tool{stubTool{name: "tool_a"}, stubTool{name: "tool_b"}, stubTool{name: "tool_c"}},
		},
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Tools) != 3 {
		t.Fatalf("expected 3 tools, got %d", len(req.Tools))
	}
	// Last tool is cached, earlier ones are not. Anthropic caches the
	// PREFIX up to the marker, so marking the tail captures all tools.
	last, ok := req.Tools[2].(map[string]any)
	if !ok {
		t.Fatalf("last tool not a map: %T", req.Tools[2])
	}
	if _, ok := last["cache_control"]; !ok {
		t.Errorf("last tool must carry cache_control; got %v", last)
	}
	for i := 0; i < 2; i++ {
		t0, _ := req.Tools[i].(map[string]any)
		if _, ok := t0["cache_control"]; ok {
			t.Errorf("tool[%d] must NOT carry cache_control (only the last one does); got %v", i, t0)
		}
	}
}

// End-to-end: the JSON wire payload must include the identity block
// AND three cache_control breakpoints (system tail, last tool, last
// user-message block). This is the integration sanity check — if any
// piece is silently dropped during marshal, the test catches it.
func TestBuildCompletionRequestWireShape(t *testing.T) {
	c, err := NewAnthropicClientWithTokenSource(context.Background(),
		oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), "")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.buildCompletionRequest(
		[]llm.Message{{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("hello")}}},
		llm.InferenceParams{
			Model: "claude-test", SystemPrompt: "be brief",
			Tools: []llm.Tool{stubTool{name: "foo"}},
		},
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	w := string(wire)
	if !strings.Contains(w, "Claude Code") {
		t.Errorf("wire payload missing identity prompt: %s", w)
	}
	// Three cache_control breakpoints: system, tool, message.
	count := strings.Count(w, `"cache_control"`)
	if count != 3 {
		t.Errorf("expected exactly 3 cache_control markers, got %d. payload=%s", count, w)
	}
}

// stubTool is a minimal llm.Tool for exercising the request builder.
type stubTool struct{ name string }

func (s stubTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Name:        s.name,
		Description: "test tool " + s.name,
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}
}

func (s stubTool) Execute(_ llm.ToolContext, _ map[string]any) llm.ToolResult {
	return llm.ToolResult{}
}

// CacheRetention="long" translates to ttl="1h" on every cache_control
// marker the request emits. pi at anthropic.ts:62 makes the same
// translation. The Anthropic API treats the 1h tier as GA so no
// beta header is required.
func TestBuildCompletionRequestLongCacheTTL(t *testing.T) {
	c, err := NewAnthropicClient(context.Background(), "sk-ant-test", "")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.buildCompletionRequest(
		[]llm.Message{{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("hi")}}},
		llm.InferenceParams{
			Model:          "claude-test",
			SystemPrompt:   "be brief",
			Tools:          []llm.Tool{stubTool{name: "foo"}},
			CacheRetention: "long",
		},
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	// System block
	if req.System[0].CacheControl == nil || req.System[0].CacheControl.TTL != "1h" {
		t.Errorf("system cache_control TTL=%q want 1h", req.System[0].CacheControl.TTL)
	}
	// Last user message
	tail := req.Messages[len(req.Messages)-1].Content[0]
	if tail.CacheControl == nil || tail.CacheControl.TTL != "1h" {
		t.Errorf("last message cache_control TTL=%q want 1h", tail.CacheControl.TTL)
	}
	// Last tool (cache_control is a map[string]any since tools are freeform)
	last, _ := req.Tools[0].(map[string]any)
	cc, _ := last["cache_control"].(map[string]any)
	if cc["ttl"] != "1h" {
		t.Errorf("tool cache_control ttl=%v want 1h", cc["ttl"])
	}
}

// CacheRetention="" (default) leaves TTL empty, which the Anthropic
// API treats as the 5-minute tier. Wire payload must NOT carry a
// "ttl" field when retention is default — sending ttl:"" is technically
// valid but pi omits the field entirely.
func TestBuildCompletionRequestDefaultCacheNoTTL(t *testing.T) {
	c, err := NewAnthropicClient(context.Background(), "sk-ant-test", "")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.buildCompletionRequest(
		[]llm.Message{{Role: llm.MessageRoleUser, Content: llm.MessageContent{llm.NewTextContentBlock("hi")}}},
		llm.InferenceParams{Model: "claude-test", SystemPrompt: "x"},
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(wire), `"ttl"`) {
		t.Errorf("default cache must NOT serialise a ttl field; payload=%s", string(wire))
	}
}

// resolveCacheTTL is the canonical mapping from caller-facing
// CacheRetention strings to Anthropic's ttl field. The set of accepted
// values is tiny so we lock the table.
func TestResolveCacheTTL(t *testing.T) {
	cases := map[string]string{
		"":         "",
		"default":  "",
		"DEFAULT":  "",
		"none":     "", // unrecognised → safe default
		"  long  ": "1h",
		"long":     "1h",
		"LONG":     "1h",
	}
	for in, want := range cases {
		if got := resolveCacheTTL(in); got != want {
			t.Errorf("resolveCacheTTL(%q)=%q want %q", in, got, want)
		}
	}
}

// fine-grained-tool-streaming-2025-05-14 beta is sent whenever the
// request carries tools. pi gates on model compat at anthropic.ts:1161;
// our equivalent is "tools present" since all native Anthropic models
// leave the eager-streaming flag falsy. Beta is NOT sent when there
// are no tools — sending unnecessary betas is API noise.
func TestRequestHeadersFineGrainedBeta(t *testing.T) {
	cAPI, _ := NewAnthropicClient(context.Background(), "sk-ant-test", "")
	with, err := cAPI.requestHeaders(context.Background(), &anthropicMessageRequest{Tools: []any{map[string]any{"name": "foo"}}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(with["anthropic-beta"], "fine-grained-tool-streaming-2025-05-14") {
		t.Errorf("expected fine-grained beta when tools present; got %q", with["anthropic-beta"])
	}
	without, _ := cAPI.requestHeaders(context.Background(), &anthropicMessageRequest{})
	if strings.Contains(without["anthropic-beta"], "fine-grained") {
		t.Errorf("fine-grained beta must NOT be sent when no tools; got %q", without["anthropic-beta"])
	}
}

// OAuth path keeps claude-code + oauth betas FIRST and appends
// fine-grained when tools are present. Ordering matters because pi's
// gateway parses them positionally; tests pin the order.
func TestRequestHeadersOAuthBetaOrdering(t *testing.T) {
	c, err := NewAnthropicClientWithTokenSource(context.Background(),
		oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok"}), "")
	if err != nil {
		t.Fatal(err)
	}
	got, _ := c.requestHeaders(context.Background(),
		&anthropicMessageRequest{Tools: []any{map[string]any{"name": "foo"}}})
	want := "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14"
	if got["anthropic-beta"] != want {
		t.Errorf("OAuth beta header order mismatch:\n  got  %q\n  want %q", got["anthropic-beta"], want)
	}
	// Without tools the fine-grained beta is dropped but OAuth betas stay.
	got2, _ := c.requestHeaders(context.Background(), &anthropicMessageRequest{})
	if got2["anthropic-beta"] != "claude-code-20250219,oauth-2025-04-20" {
		t.Errorf("OAuth beta (no tools): got %q", got2["anthropic-beta"])
	}
}
