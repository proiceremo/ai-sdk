package codeexec

import (
	"strings"
	"testing"
)

type stubError string

func (e stubError) Error() string { return string(e) }

func TestSimplifyJSErrorPreservesEvalSourceLocation(t *testing.T) {
	raw := stubError("SyntaxError: Unexpected identifier 'Read'\n    at ai_sdk_eval.js:42:5\n")
	got := simplifyJSError(raw).Error()
	if !strings.Contains(got, "SyntaxError: Unexpected identifier 'Read'") {
		t.Fatalf("description was lost: %q", got)
	}
	if !strings.Contains(got, "ai_sdk_eval.js:42:5") {
		t.Fatalf("expected source location to be preserved, got %q", got)
	}
}

func TestSimplifyJSErrorPreservesParenthesizedFrame(t *testing.T) {
	// Runtime errors render frames as "at <fn> (<source>:<line>:<col>)".
	raw := stubError("Error: boom\n    at <eval> (ai_sdk_eval.js:1:10)\n")
	got := simplifyJSError(raw).Error()
	if !strings.Contains(got, "ai_sdk_eval.js:1:10") {
		t.Fatalf("expected parenthesized frame to be preserved, got %q", got)
	}
}

func TestSimplifyJSErrorSkipsInternalFrames(t *testing.T) {
	raw := stubError("TypeError: x is not a function\n    at internalThing (qjs/internal:9:1)\n    at ai_sdk_prelude.js:5:3\n")
	got := simplifyJSError(raw).Error()
	if !strings.Contains(got, "ai_sdk_prelude.js:5:3") {
		t.Fatalf("expected first user-source frame to be picked, got %q", got)
	}
	if strings.Contains(got, "qjs/internal") {
		t.Fatalf("internal frame leaked into simplified message: %q", got)
	}
}

func TestSimplifyJSErrorWithoutLocationFallsBackToDescription(t *testing.T) {
	raw := stubError("ReferenceError: foo is not defined")
	got := simplifyJSError(raw).Error()
	if got != "ReferenceError: foo is not defined" {
		t.Fatalf("expected pass-through for single-line error, got %q", got)
	}
}
