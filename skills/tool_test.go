package skills

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	llm "github.com/proiceremo/ai-sdk"
)

func TestSkillsToolDiscoversGlobalAndWorkspaceSkills(t *testing.T) {
	home := t.TempDir()
	wd := t.TempDir()
	t.Setenv("HOME", home)

	writeSkill(t, filepath.Join(home, ".agents", "skills", "ai-sdk"), `---
name: ai-sdk
description: Build AI SDK agents and tools
license: MIT
---
# AI SDK

Use this skill for AI SDK work.
`)
	writeSkill(t, filepath.Join(wd, ".agents", "skills", "workspace-skill"), `---
name: workspace-skill
description: Workspace-specific guidance
---
# Workspace Skill
`)

	tool := NewTool(NewLoader(Options{}))
	ctx := llm.ToolContext{Context: context.Background(), WorkingDirectory: wd}

	result := tool.Execute(ctx, map[string]any{"mode": "list"})
	if result.Error != nil {
		t.Fatalf("list failed: %v", result.Error)
	}
	list, ok := result.StructuredOutput.(ListResult)
	if !ok {
		t.Fatalf("unexpected list output type: %T", result.StructuredOutput)
	}
	if len(list.Skills) != 2 {
		t.Fatalf("expected global and workspace skills, got %#v", list.Skills)
	}
	if list.Skills[0].Name != "ai-sdk" || list.Skills[0].Description != "Build AI SDK agents and tools" {
		t.Fatalf("frontmatter was not parsed: %#v", list.Skills[0])
	}
	if list.Skills[1].Name != "workspace-skill" || list.Skills[1].Scope != "workspace" {
		t.Fatalf("workspace skill missing or wrong scope: %#v", list.Skills[1])
	}
}

func TestSkillsToolSearchesAndReadsFromCanonicalWorkspacePath(t *testing.T) {
	home := t.TempDir()
	wd := t.TempDir()
	t.Setenv("HOME", home)
	writeSkill(t, filepath.Join(wd, ".agents", "skills", "svelte"), `---
name: svelte
description: Svelte component guidance
---
# Svelte

Use the Svelte docs before editing components.
`)

	tool := NewTool(NewLoader(Options{}))
	ctx := llm.ToolContext{Context: context.Background(), WorkingDirectory: wd}

	search := tool.Execute(ctx, map[string]any{"mode": "search", "query": "component"})
	if search.Error != nil {
		t.Fatalf("search failed: %v", search.Error)
	}
	list := search.StructuredOutput.(ListResult)
	if len(list.Skills) != 1 || list.Skills[0].Name != "svelte" {
		t.Fatalf("search did not find skill: %#v", list.Skills)
	}

	read := tool.Execute(ctx, map[string]any{"mode": "read", "skill": "svelte"})
	if read.Error != nil {
		t.Fatalf("read failed: %v", read.Error)
	}
	loaded := read.StructuredOutput.(ReadResult)
	if !strings.Contains(loaded.Content, "Use the Svelte docs") {
		t.Fatalf("read returned wrong content: %#v", loaded)
	}
}

func TestSkillsToolListReturnsEmptyArrayNotNull(t *testing.T) {
	home := t.TempDir()
	wd := t.TempDir()
	t.Setenv("HOME", home)

	tool := NewTool(NewLoader(Options{}))
	result := tool.Execute(llm.ToolContext{Context: context.Background(), WorkingDirectory: wd}, map[string]any{"mode": "list"})
	if result.Error != nil {
		t.Fatalf("list failed: %v", result.Error)
	}
	list := result.StructuredOutput.(ListResult)
	if list.Skills == nil {
		t.Fatal("expected empty skills array, got nil")
	}
	if len(list.Skills) != 0 {
		t.Fatalf("expected no skills, got %#v", list.Skills)
	}
}

func TestSkillsToolSchemaAdvertisesSupportedModes(t *testing.T) {
	schema := NewTool(NewLoader(Options{})).Schema()
	properties, ok := schema.InputSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties: %#v", schema.InputSchema)
	}
	mode, ok := properties["mode"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no mode property: %#v", properties)
	}
	enum, ok := mode["enum"].([]any)
	if !ok {
		t.Fatalf("mode schema has no enum: %#v", mode)
	}
	for _, expected := range []string{"list", "search", "read", "create", "patch", "archive", "restore"} {
		if !containsEnum(enum, expected) {
			t.Fatalf("mode enum missing %q: %#v", expected, enum)
		}
	}
	for _, removed := range []string{"discover", "query", "load", "get"} {
		if containsEnum(enum, removed) {
			t.Fatalf("mode enum should not include %q: %#v", removed, enum)
		}
	}
}

func TestSkillsToolRejectsAliasModes(t *testing.T) {
	tool := NewTool(NewLoader(Options{}))
	for _, mode := range []string{"discover", "query", "load", "get", "view", "open", "find"} {
		result := tool.Execute(llm.ToolContext{Context: context.Background()}, map[string]any{"mode": mode})
		if result.Error == nil {
			t.Fatalf("expected alias mode %q to be rejected", mode)
		}
	}
}

func TestSkillsToolCreatesPatchesAndArchivesManagedSkill(t *testing.T) {
	home := t.TempDir()
	wd := t.TempDir()
	t.Setenv("HOME", home)

	tool := NewTool(NewLoader(Options{ManagedRoot: filepath.Join(wd, ".agents", "skills")}))
	ctx := llm.ToolContext{Context: context.Background(), WorkingDirectory: wd}
	create := tool.Execute(ctx, map[string]any{
		"mode":    "create",
		"skill":   "demo",
		"content": "# Demo\n\nUse this.",
	})
	if create.Error != nil {
		t.Fatalf("create failed: %v", create.Error)
	}
	patch := tool.Execute(ctx, map[string]any{
		"mode":    "patch",
		"skill":   "demo",
		"content": "# Demo\n\nUse this carefully.",
	})
	if patch.Error != nil {
		t.Fatalf("patch failed: %v", patch.Error)
	}
	read := tool.Execute(ctx, map[string]any{"mode": "read", "skill": "demo"})
	if read.Error != nil {
		t.Fatalf("read failed: %v", read.Error)
	}
	if !strings.Contains(read.StructuredOutput.(ReadResult).Content, "carefully") {
		t.Fatalf("patch did not update content: %#v", read.StructuredOutput)
	}
	archive := tool.Execute(ctx, map[string]any{"mode": "archive", "skill": "demo"})
	if archive.Error != nil {
		t.Fatalf("archive failed: %v", archive.Error)
	}
	usage, err := ReadUsage(filepath.Join(wd, ".agents", "skills", "demo"))
	if err != nil {
		t.Fatalf("ReadUsage: %v", err)
	}
	if usage.State != "archived" || usage.Provenance != "agent-created" {
		t.Fatalf("unexpected usage: %#v", usage)
	}
}

func TestSkillsToolRejectsPathTraversalSkillNames(t *testing.T) {
	tool := NewTool(NewLoader(Options{ManagedRoot: filepath.Join(t.TempDir(), ".agents", "skills")}))
	result := tool.Execute(llm.ToolContext{Context: context.Background()}, map[string]any{
		"mode":    "create",
		"skill":   "../escape",
		"content": "# Escape",
	})
	if result.Error == nil {
		t.Fatal("expected invalid skill name error")
	}
}

func writeSkill(t *testing.T, dir string, content string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(content), 0o600); err != nil {
		t.Fatalf("write skill: %v", err)
	}
}

func containsEnum(values []any, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
