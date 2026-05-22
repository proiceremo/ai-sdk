package codeexec

import (
	"regexp"
	"strings"
)

var codeFenceRegex = regexp.MustCompile("(?s)^```(?:js|javascript|typescript|ts|tsx|jsx)?\\s*\\n?(.*?)\\n?```\\s*$")

// NormalizeCode strips markdown fences and wraps the code into an async IIFE
// that JSON-stringifies its return value. The wrapping is deliberately
// permissive: callers can pass a single expression, an arrow function literal,
// or a multi-line block.
//
// The two failure modes we explicitly handle:
//   - Bare arrow function literal: `(input) => tools.read(...)` — we
//     invoke it with the standard environment object so the result is captured.
//   - Block without an explicit `return`: we transform the last non-empty,
//     non-control statement into `return <expr>` so the agent doesn't silently
//     get `undefined` back.
func NormalizeCode(code string) string {
	code = strings.TrimSpace(code)
	if matches := codeFenceRegex.FindStringSubmatch(code); len(matches) > 1 {
		code = strings.TrimSpace(matches[1])
	}
	if code == "" {
		return "(async () => JSON.stringify(undefined))()"
	}

	if isArrowFunctionLiteral(code) {
		return "(async () => {\n" +
			"  const __user = (" + code + ");\n" +
			"  const __result = await __user({ input, cwd, env, tools, vars });\n" +
			"  return JSON.stringify(__result);\n" +
			"})()"
	}

	body := ensureTrailingReturn(code)
	return "(async () => {\n" +
		"  const __result = await (async () => {\n" + body + "\n  })();\n" +
		"  return JSON.stringify(__result);\n" +
		"})()"
}

// isArrowFunctionLiteral matches `(args) => expr` and `async (args) => expr`
// shapes that should be invoked rather than re-wrapped.
func isArrowFunctionLiteral(code string) bool {
	if strings.Contains(code, "\n") {
		return false
	}
	if !strings.Contains(code, "=>") {
		return false
	}
	trimmed := strings.TrimSpace(code)
	if strings.HasPrefix(trimmed, "(") || strings.HasPrefix(trimmed, "async") {
		// Reject blocks that explicitly use `return` — those should go through
		// the standard wrap so the inner async IIFE captures their value.
		return !strings.Contains(trimmed, "return ")
	}
	return false
}

// ensureTrailingReturn looks at the final non-empty line and, if it's a bare
// expression (no leading keyword like `return`, `if`, `const`, …), rewrites
// it as `return <expr>`. This lets models write multi-line blocks that end in
// the value they want to return without remembering the keyword.
func ensureTrailingReturn(code string) string {
	if strings.Contains(code, "return ") || strings.Contains(code, "return\n") || strings.Contains(code, "return;") {
		return code
	}
	lines := strings.Split(code, "\n")
	idx := -1
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.TrimSpace(lines[i]) != "" {
			idx = i
			break
		}
	}
	if idx < 0 {
		return code
	}
	last := lines[idx]
	trimmed := strings.TrimSpace(last)
	if trimmed == "" || trimmed == "}" || trimmed == "{" || strings.HasPrefix(trimmed, "//") {
		return code
	}
	if hasLeadingStatementKeyword(trimmed) {
		return code
	}
	indent := last[:len(last)-len(strings.TrimLeft(last, " \t"))]
	expression := strings.TrimRight(trimmed, ";")
	if expression == "" {
		return code
	}
	lines[idx] = indent + "return (" + expression + ");"
	return strings.Join(lines, "\n")
}

// statementKeywords are the leading tokens that indicate a line is a
// statement we should NOT rewrite into `return <expr>`. Notably absent:
// `await` — a trailing `await tools.foo()` is exactly the kind of value the
// model wants returned.
var statementKeywords = []string{
	"return", "throw", "if", "for", "while", "do", "switch", "try", "function",
	"class", "const", "let", "var", "import", "export", "yield",
}

func hasLeadingStatementKeyword(line string) bool {
	for _, kw := range statementKeywords {
		if strings.HasPrefix(line, kw+" ") || line == kw || strings.HasPrefix(line, kw+"(") || strings.HasPrefix(line, kw+"{") {
			return true
		}
	}
	return false
}
