package skills

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	llm "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/tools"
)

func NewTool(loader *Loader) llm.Tool {
	if loader == nil {
		loader = NewLoader(Options{})
	}
	return tools.NewGenericTool("skills", "List, search, read, and manage local agent skills", func(ctx llm.ToolContext, input Input) llm.ToolResult {
		return executeTool(ctx, loader, input)
	}).WithPermissionExtractor(func(ctx llm.ToolContext, input Input) ([]llm.PermissionGuard, error) {
		mode := strings.TrimSpace(input.Mode)
		if mode != "create" && mode != "patch" {
			return nil, nil
		}
		if strings.TrimSpace(input.Skill) == "" {
			return nil, nil
		}
		skillName, err := cleanSkillName(input.Skill)
		if err != nil {
			return nil, err
		}
		path := filepath.Join(managedSkillRoot(ctx.WorkingDirectory, loader), skillName, "SKILL.md")
		return []llm.PermissionGuard{{
			Key:        "skills.write",
			Specifiers: []string{"path:" + filepath.Clean(path)},
			MatchMode:  llm.PermissionMatchModeExact,
			Options:    tools.FileWritePermissionOptions("skills"),
		}}, nil
	})
}

func executeTool(ctx llm.ToolContext, loader *Loader, input Input) llm.ToolResult {
	mode := strings.TrimSpace(input.Mode)
	if mode == "" {
		mode = "list"
	}
	mode = strings.ToLower(mode)

	switch mode {
	case "list", "search":
		result, err := loader.List(input, ctx.WorkingDirectory)
		if err != nil {
			return tools.ErrorResult(err)
		}
		return tools.JSONResult("List skills", llm.ToolKindOther, result, "")
	case "read":
		result, err := loader.Read(input, ctx.WorkingDirectory)
		if err != nil {
			return tools.ErrorResult(err)
		}
		_ = TouchUsage(filepath.Dir(result.Path), "view")
		return tools.JSONResult("Read skill", llm.ToolKindOther, result, result.Path)
	case "create", "patch":
		result, err := writeManagedSkill(ctx.WorkingDirectory, loader, input, mode == "patch")
		if err != nil {
			return tools.ErrorResult(err)
		}
		path := ""
		if v, ok := result["path"].(string); ok {
			path = v
		}
		title := "Create skill"
		if mode == "patch" {
			title = "Patch skill"
		}
		return tools.JSONResult(title, llm.ToolKindOther, result, path)
	case "archive", "restore":
		result, err := setManagedSkillState(ctx.WorkingDirectory, loader, input.Skill, mode)
		if err != nil {
			return tools.ErrorResult(err)
		}
		title := "Archive skill"
		if mode == "restore" {
			title = "Restore skill"
		}
		return tools.JSONResult(title, llm.ToolKindOther, result, "")
	default:
		return tools.ErrorResult(fmt.Errorf("unsupported skills mode: %s", mode))
	}
}

func managedSkillRoot(workingDir string, loader *Loader) string {
	if loader != nil && strings.TrimSpace(loader.managedRoot) != "" {
		return filepath.Clean(loader.managedRoot)
	}
	if strings.TrimSpace(workingDir) != "" {
		return filepath.Join(workingDir, ".agents", "skills")
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".agents", "skills")
	}
	return filepath.Join(".agents", "skills")
}

func writeManagedSkill(workingDir string, loader *Loader, input Input, patch bool) (map[string]any, error) {
	name := strings.TrimSpace(input.Skill)
	cleanName, err := cleanSkillName(name)
	if err != nil {
		return nil, err
	}
	content := strings.TrimSpace(input.Content)
	if content == "" {
		return nil, fmt.Errorf("content is required")
	}
	root := managedSkillRoot(workingDir, loader)
	dir := filepath.Join(root, cleanName)
	if patch {
		usage, err := ReadUsage(dir)
		if err != nil || usage.Provenance != "agent-created" || usage.Pinned {
			return nil, fmt.Errorf("patch requires an unpinned agent-created skill")
		}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "SKILL.md")
	if err := os.WriteFile(path, []byte(content+"\n"), 0o600); err != nil {
		return nil, err
	}
	usage, _ := ReadUsage(dir)
	if usage.Provenance == "" {
		usage.Provenance = "agent-created"
	}
	usage.State = "active"
	if input.Pinned != nil {
		usage.Pinned = *input.Pinned
	}
	if err := WriteUsage(dir, usage); err != nil {
		return nil, err
	}
	_ = TouchUsage(dir, "patch")
	usage, _ = ReadUsage(dir)
	return map[string]any{"name": cleanName, "path": path, "usage": usage}, nil
}

func setManagedSkillState(workingDir string, loader *Loader, name string, mode string) (map[string]any, error) {
	cleanName, err := cleanSkillName(name)
	if err != nil {
		return nil, err
	}
	root := managedSkillRoot(workingDir, loader)
	dir := filepath.Join(root, cleanName)
	usage, err := ReadUsage(dir)
	if err != nil || usage.Provenance != "agent-created" {
		return nil, fmt.Errorf("%s requires an agent-created skill", mode)
	}
	if usage.Pinned {
		return nil, fmt.Errorf("skill is pinned")
	}
	if mode == "archive" {
		usage.State = "archived"
	} else {
		usage.State = "active"
	}
	if err := WriteUsage(dir, usage); err != nil {
		return nil, err
	}
	return map[string]any{"name": cleanName, "state": usage.State}, nil
}

func cleanSkillName(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return "", fmt.Errorf("skill is required")
	}
	clean := filepath.Clean(name)
	if clean == "." || clean == ".." || filepath.IsAbs(clean) || strings.Contains(clean, string(os.PathSeparator)) {
		return "", fmt.Errorf("invalid skill name")
	}
	return clean, nil
}
