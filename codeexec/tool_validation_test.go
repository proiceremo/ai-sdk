package codeexec

import (
	"strings"
	"testing"
)

func TestLongestInlineStringLiteralFindsBacktickTemplate(t *testing.T) {
	body := strings.Repeat("x", 700)
	src := "return `" + body + "`;"
	kind, n, found := longestInlineStringLiteral(src)
	if !found {
		t.Fatal("expected to find a literal")
	}
	if kind != "template" {
		t.Fatalf("expected kind=template, got %q", kind)
	}
	if n != 700 {
		t.Fatalf("expected length=700, got %d", n)
	}
}

func TestLongestInlineStringLiteralFindsDoubleQuote(t *testing.T) {
	body := strings.Repeat("a", 1024)
	src := `const s = "` + body + `";`
	kind, n, _ := longestInlineStringLiteral(src)
	if kind != "string" || n != 1024 {
		t.Fatalf("expected (string, 1024), got (%s, %d)", kind, n)
	}
}

func TestLongestInlineStringLiteralIgnoresInterpolatedCode(t *testing.T) {
	// The template body proper is short ("a" + ... + "b"). The
	// interpolation contains a long expression but it's *code*, not a
	// string literal — it must not be counted toward the template's length.
	longExpr := strings.Repeat("z+", 600)
	src := "const t = `a${" + longExpr + "0}b`;"
	kind, n, found := longestInlineStringLiteral(src)
	if !found {
		t.Fatal("expected to find at least one literal segment")
	}
	if kind != "template" {
		t.Fatalf("expected kind=template, got %q", kind)
	}
	// The visible template content split by the interpolation is "a" then
	// "b". Each segment should be ≤2 bytes.
	if n > 5 {
		t.Fatalf("expected short template literal (interpolation excluded); got %d bytes", n)
	}
}

func TestLongestInlineStringLiteralIgnoresComments(t *testing.T) {
	// A line comment full of "characters" must not be counted.
	src := "// " + strings.Repeat("x", 5000) + "\nconst y = 1;"
	_, n, _ := longestInlineStringLiteral(src)
	if n > 0 {
		t.Fatalf("expected 0 literal bytes, got %d", n)
	}
}

func TestLongestInlineStringLiteralRespectsEscapes(t *testing.T) {
	// The middle `"` is escaped — the literal is the entire quoted span.
	src := `const s = "hello \"world\" goodbye";`
	kind, n, _ := longestInlineStringLiteral(src)
	if kind != "string" || n != len(`hello \"world\" goodbye`) {
		t.Fatalf("expected (string, 23), got (%s, %d)", kind, n)
	}
}



func TestValidateJSExecuteInputAllowsShortInlineLiteral(t *testing.T) {
	err := validateJSExecuteInput(JSExecuteInput{
		Code: `const x = "hi"; return x;`,
	})
	if err != nil {
		t.Fatalf("expected short code to pass; got %v", err)
	}
}

func TestValidateJSExecuteInputAllowsLargeCodeFieldOfShortLiterals(t *testing.T) {
	// The total-code-size cap was removed: legitimate use cases can emit
	// many short literals (each well under the per-literal limit) and the
	// total payload size is bounded only by the model's own output_tokens
	// budget. This locks in that "no total cap" promise.
	parts := []string{}
	for i := 0; i < 600; i++ {
		parts = append(parts, `const a`+string(rune('A'+i%26))+` = "abcdefghij";`)
	}
	huge := strings.Join(parts, " ")
	if len(huge) < 16*1024 {
		t.Skip("test fixture didn't generate enough bytes to be meaningful")
	}
	err := validateJSExecuteInput(JSExecuteInput{Code: huge})
	if err != nil {
		t.Fatalf("expected large code with only short literals to pass; got %v", err)
	}
}

func TestValidateJSExecuteInputAllowsInputCarriedContent(t *testing.T) {
	// This is exactly the encouraged shape: the long content rides through
	// `input`, the JS body just references it. Must pass.
	err := validateJSExecuteInput(JSExecuteInput{
		Code:  `return await tools.finish({ summary: input.summary, output: input.output });`,
		Input: map[string]any{"summary": strings.Repeat("x", 5000)},
	})
	if err != nil {
		t.Fatalf("expected the input-carried shape to pass; got %v", err)
	}
}
