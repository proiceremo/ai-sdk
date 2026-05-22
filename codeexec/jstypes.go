package codeexec

import (
	"fmt"
	"sort"
	"strings"

	llm "ai-sdk"
)

// BuildToolTypeDeclarations renders the model-facing TypeScript declarations
// for the tool surface a code-executing agent can call. The output is
// intentionally TypeScript-shaped because most frontier models recognise it as
// a type contract and stop guessing argument shapes.
func BuildToolTypeDeclarations(tools []llm.Tool) string {
	if len(tools) == 0 {
		return "declare const tools: Record<string, (input?: any) => Promise<any>>;"
	}

	type toolEntry struct {
		name        string
		description string
		inputType   string
		inputDecl   string
		outputType  string
		outputDecl  string
	}

	entries := make([]toolEntry, 0, len(tools))
	seen := map[string]bool{}
	for _, t := range tools {
		if t == nil {
			continue
		}
		schema := t.Schema()
		if schema.Name == "" || seen[schema.Name] {
			continue
		}
		seen[schema.Name] = true
		typeName := tsIdentifier(schema.Name) + "Input"
		outputType := ""
		outputDecl := ""
		if len(schema.OutputSchema) > 0 {
			outputType = tsIdentifier(schema.Name) + "Output"
			outputDecl = "type " + outputType + " = " + renderTSType(schema.OutputSchema, 0) + ";"
		}
		entries = append(entries, toolEntry{
			name:        schema.Name,
			description: schema.Description,
			inputType:   typeName,
			inputDecl:   "type " + typeName + " = " + renderTSType(schema.InputSchema, 0) + ";",
			outputType:  outputType,
			outputDecl:  outputDecl,
		})
	}

	sort.Slice(entries, func(i, j int) bool { return entries[i].name < entries[j].name })

	var b strings.Builder
	b.WriteString("type JSONPrimitive = string | number | boolean | null;\n")
	b.WriteString("type JSONValue = JSONPrimitive | { [key: string]: JSONValue } | JSONValue[];\n")
	b.WriteString("declare const cwd: string;\n")
	b.WriteString("declare const input: Readonly<Record<string, JSONValue>>;\n")
	b.WriteString("declare const env: Readonly<{ cwd: string; session_id: string; run_id: string; model: string; vars?: Record<string, JSONValue> }>;\n")
	b.WriteString("declare const vars: Record<string, JSONValue>;\n")
	b.WriteString("declare const console: { log(...args: unknown[]): void; warn(...args: unknown[]): void; error(...args: unknown[]): void };\n\n")
	for _, e := range entries {
		b.WriteString(e.inputDecl)
		b.WriteByte('\n')
		if e.outputDecl != "" {
			b.WriteString(e.outputDecl)
			b.WriteByte('\n')
		}
	}
	b.WriteString("\ndeclare const tools: {\n")
	for _, e := range entries {
		if e.description != "" {
			b.WriteString("  /** ")
			b.WriteString(strings.ReplaceAll(strings.TrimSpace(e.description), "\n", " "))
			b.WriteString(" */\n")
		}
		ret := "any"
		if e.outputType != "" {
			ret = e.outputType
		}
		fmt.Fprintf(&b, "  %s(input: %s): Promise<%s>;\n", tsPropertyName(e.name), e.inputType, ret)
	}
	b.WriteString("  list(): Array<{ name: string; description?: string }>;\n")
	b.WriteString("  schema(name: string): unknown;\n")
	b.WriteString("};\n")
	return b.String()
}

func tsIdentifier(name string) string {
	var b strings.Builder
	upper := true
	for _, r := range name {
		switch {
		case r >= 'A' && r <= 'Z', r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			if upper && r >= 'a' && r <= 'z' {
				r -= 32
			}
			b.WriteRune(r)
			upper = false
		default:
			upper = true
		}
	}
	if b.Len() == 0 {
		return "Tool"
	}
	return b.String()
}

func tsPropertyName(name string) string {
	for _, r := range name {
		if !(r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' || r == '$') {
			return fmt.Sprintf("%q", name)
		}
	}
	return name
}

// renderTSType is a small JSON-Schema → TS renderer. It only handles the
// shapes our generic tool builder produces (object/string/number/bool/array
// plus enums and unions); anything more exotic falls back to `any`.
func renderTSType(schema map[string]any, depth int) string {
	if depth > 6 || len(schema) == 0 {
		return "any"
	}
	if enum, ok := schema["enum"].([]any); ok && len(enum) > 0 {
		parts := make([]string, 0, len(enum))
		for _, v := range enum {
			parts = append(parts, jsLiteral(v))
		}
		return strings.Join(parts, " | ")
	}
	if anyOf, ok := schema["anyOf"].([]any); ok && len(anyOf) > 0 {
		parts := make([]string, 0, len(anyOf))
		for _, v := range anyOf {
			if m, ok := v.(map[string]any); ok {
				parts = append(parts, renderTSType(m, depth+1))
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, " | ")
		}
	}
	switch t := schema["type"].(type) {
	case string:
		return renderTSPrimitive(t, schema, depth)
	case []any:
		parts := make([]string, 0, len(t))
		for _, item := range t {
			if s, ok := item.(string); ok {
				parts = append(parts, renderTSPrimitive(s, schema, depth))
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, " | ")
		}
	}
	return "any"
}

func renderTSPrimitive(t string, schema map[string]any, depth int) string {
	switch t {
	case "string":
		return "string"
	case "integer", "number":
		return "number"
	case "boolean":
		return "boolean"
	case "null":
		return "null"
	case "array":
		items, _ := schema["items"].(map[string]any)
		return "Array<" + renderTSType(items, depth+1) + ">"
	case "object":
		props, _ := schema["properties"].(map[string]any)
		if len(props) == 0 {
			additional, ok := schema["additionalProperties"].(map[string]any)
			if ok {
				return "Record<string, " + renderTSType(additional, depth+1) + ">"
			}
			return "Record<string, any>"
		}
		required := map[string]bool{}
		if rs, ok := schema["required"].([]any); ok {
			for _, item := range rs {
				if s, ok := item.(string); ok {
					required[s] = true
				}
			}
		}
		names := make([]string, 0, len(props))
		for name := range props {
			names = append(names, name)
		}
		sort.Strings(names)
		indent := strings.Repeat("  ", depth+1)
		closing := strings.Repeat("  ", depth)
		var b strings.Builder
		b.WriteString("{\n")
		for _, name := range names {
			child, _ := props[name].(map[string]any)
			optional := ""
			if !required[name] {
				optional = "?"
			}
			if desc, _ := child["description"].(string); strings.TrimSpace(desc) != "" {
				b.WriteString(indent + "/** " + strings.ReplaceAll(strings.TrimSpace(desc), "\n", " ") + " */\n")
			}
			fmt.Fprintf(&b, "%s%s%s: %s;\n", indent, tsPropertyName(name), optional, renderTSType(child, depth+1))
		}
		b.WriteString(closing + "}")
		return b.String()
	}
	return "any"
}

func jsLiteral(v any) string {
	switch t := v.(type) {
	case string:
		return fmt.Sprintf("%q", t)
	case bool:
		if t {
			return "true"
		}
		return "false"
	case nil:
		return "null"
	default:
		return fmt.Sprintf("%v", t)
	}
}
