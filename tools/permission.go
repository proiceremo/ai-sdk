package tools

import (
	"fmt"
	"path/filepath"
	"strings"

	llm "github.com/proiceremo/ai-sdk"
)

type PermissionExtractor struct {
	Key       string
	Fields    []PermissionField
	MatchMode llm.PermissionMatchMode
	Options   []llm.PermissionOption
}

type PermissionField struct {
	Name        string
	Transform   string
	Base        string
	DefaultFrom string
	Segments    int
}

func ExtractPermission(ctx llm.ToolContext, input map[string]any, spec PermissionExtractor) (llm.PermissionGuard, error) {
	if spec.Key == "" {
		return llm.PermissionGuard{}, fmt.Errorf("permission key is required")
	}
	mode := spec.MatchMode
	if mode == "" {
		mode = llm.PermissionMatchModeExact
	}
	guard := llm.PermissionGuard{
		Key:       spec.Key,
		MatchMode: mode,
		Options:   append([]llm.PermissionOption(nil), spec.Options...),
	}
	for _, field := range spec.Fields {
		value := permissionFieldValue(ctx, input, field)
		if value == "" {
			continue
		}
		guard.Specifiers = append(guard.Specifiers, field.Name+":"+value)
	}
	return guard, nil
}

func ExtractPermissions(ctx llm.ToolContext, input map[string]any, specs ...PermissionExtractor) ([]llm.PermissionGuard, error) {
	out := make([]llm.PermissionGuard, 0, len(specs))
	for _, spec := range specs {
		guard, err := ExtractPermission(ctx, input, spec)
		if err != nil {
			return nil, err
		}
		out = append(out, guard)
	}
	return out, nil
}

func permissionFieldValue(ctx llm.ToolContext, input map[string]any, field PermissionField) string {
	raw := ""
	if field.Name != "" {
		if value, ok := input[field.Name]; ok {
			raw = strings.TrimSpace(fmt.Sprintf("%v", value))
		}
	}
	if raw == "" && field.DefaultFrom == "cwd" {
		raw = ctx.WorkingDirectory
	}
	if raw == "" {
		return ""
	}
	return TransformPermissionValue(ctx.WorkingDirectory, raw, field.Transform, field.Base, field.Segments)
}

func TransformPermissionValue(cwd string, value string, transform string, base string, segments int) string {
	switch transform {
	case "", "self":
		return value
	case "path":
		return normalizePermissionPath(cwd, value, base)
	case "dirname":
		return filepath.Dir(normalizePermissionPath(cwd, value, base))
	case "words":
		if segments <= 0 {
			segments = 1
		}
		parts := strings.Fields(value)
		if len(parts) == 0 {
			return ""
		}
		if len(parts) < segments {
			segments = len(parts)
		}
		return strings.Join(parts[:segments], " ")
	default:
		return value
	}
}

func normalizePermissionPath(cwd string, value string, base string) string {
	if value == "" {
		return ""
	}
	if base == "cwd" && !filepath.IsAbs(value) {
		value = filepath.Join(cwd, value)
	}
	if abs, err := filepath.Abs(value); err == nil {
		return filepath.Clean(abs)
	}
	return filepath.Clean(value)
}
