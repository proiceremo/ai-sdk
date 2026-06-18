package llm

import (
	"path/filepath"
	"strings"
	"unicode/utf8"
)

func varString(ctx ToolContext, key string) string {
	if ctx.Vars == nil {
		return ""
	}
	v, _ := ctx.Vars[key].(string)
	return v
}

func ToolSessionID(ctx ToolContext) string { return varString(ctx, "session_id") }
func ToolActorID(ctx ToolContext) string   { return varString(ctx, "actor_id") }
func ToolRunID(ctx ToolContext) string     { return varString(ctx, "run_id") }
func ToolProHome(ctx ToolContext) string   { return varString(ctx, "pro_home") }

func ActorToolOutputPath(proHome, sessionID, actorID, runID, callID, callName string) string {
	return ActorToolOutputArtifactPath(proHome, sessionID, actorID, runID, callID, callName, ".txt")
}

func ActorToolOutputArtifactPath(proHome, sessionID, actorID, runID, callID, callName, ext string) string {
	if ext == "" {
		ext = ".bin"
	}
	if ext[0] != '.' {
		ext = "." + ext
	}
	return filepath.Join(
		proHome, "sessions", safePathPart(sessionID), "actors", safePathPart(actorID),
		"runs", safePathPart(runID), "tool-outputs",
		safePathPart(callID)+"-"+safePathPart(callName)+safePathPart(ext),
	)
}

func safePathPart(value string) string {
	var b []byte
	for i := 0; i < len(value); i++ {
		c := value[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' {
			b = append(b, c)
		} else {
			b = append(b, '_')
		}
	}
	if len(b) == 0 {
		return "unknown"
	}
	return string(b)
}

// SanitizeSurrogates removes unpaired Unicode surrogate characters and invalid UTF-8 bytes from a string.
func SanitizeSurrogates(s string) string {
	var sb strings.Builder
	sb.Grow(len(s))
	for i, w := 0, 0; i < len(s); i += w {
		r, size := utf8.DecodeRuneInString(s[i:])
		w = size
		if r == utf8.RuneError && size == 1 {
			// Invalid UTF-8 byte, skip it
			continue
		}
		if r >= 0xD800 && r <= 0xDFFF {
			// Unpaired surrogate code point, skip it
			continue
		}
		sb.WriteRune(r)
	}
	return sb.String()
}

