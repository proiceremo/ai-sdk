package codeexec

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	llm "github.com/proiceremo/ai-sdk"
)

const nestedToolOutputSpillThreshold = 32 * 1024

// spillLargeStringFieldThreshold is the size at which an individual string
// field on a spilled tool result is replaced with a truncation placeholder.
// Smaller string fields (path, sha256, mime_type, etc.) pass through so the
// schema shape the calling agent observes stays stable.
const spillLargeStringFieldThreshold = 1024

func safePathPart(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
			continue
		}
		b.WriteByte('_')
	}
	return b.String()
}

func spillNestedToolOutputIfNeeded(proHome, sessionID, actorID, runID, callID, callName, text string, value any, result llm.ToolResult) (any, string) {
	if len(text) <= nestedToolOutputSpillThreshold {
		return value, text
	}
	if proHome == "" {
		return value, text
	}
	file := llm.ActorToolOutputPath(proHome, sessionID, actorID, runID, callID, callName)
	dir := filepath.Dir(file)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return value, text
	}
	if err := os.WriteFile(file, []byte(text), 0o644); err != nil {
		return value, text
	}
	var b strings.Builder
	fmt.Fprintf(&b, "[tool output spilled: %d bytes]\n", len(text))
	fmt.Fprintf(&b, "saved_path: %s\n", file)
	fmt.Fprint(&b, "Use `read_range` or `grep` on saved_path; do not load the whole file unless needed.")
	// Preserve the tool's structured shape so agent code keeps working. A
	// generic spill marker used to replace the whole value, which broke
	// schema-driven access patterns like `result.content.split(...)` for
	// read. Now we keep every top-level field, only swapping out the
	// large string payload(s) for a short placeholder, and overlay the spill
	// metadata fields on top.
	shaped := shapePreservedSpill(value, file)
	shaped["spilled"] = true
	shaped["saved_path"] = file
	shaped["bytes"] = len(text)
	shaped["hint"] = "Use `read_range` or `grep` on saved_path."
	return shaped, b.String()
}

// shapePreservedSpill renders the original tool value as a map (via JSON
// round-trip) and replaces any string field above spillLargeStringFieldThreshold
// with a short placeholder that points at the saved file. Returns an empty map
// when the value isn't an object-shaped payload — callers still get the spill
// metadata layered on top.
func shapePreservedSpill(value any, savedPath string) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	data, err := json.Marshal(value)
	if err != nil {
		return map[string]any{}
	}
	var asMap map[string]any
	if err := json.Unmarshal(data, &asMap); err != nil {
		return map[string]any{}
	}
	for k, v := range asMap {
		s, ok := v.(string)
		if !ok {
			continue
		}
		if len(s) > spillLargeStringFieldThreshold {
			asMap[k] = fmt.Sprintf("[truncated %d bytes; full output saved to %s]", len(s), savedPath)
		}
	}
	return asMap
}
