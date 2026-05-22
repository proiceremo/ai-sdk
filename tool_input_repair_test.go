package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRepairToolInputPassesValidJSON(t *testing.T) {
	in := json.RawMessage(`{"code": "console.log(1)"}`)
	out, kind, err := RepairToolInput(in)
	if err != nil {
		t.Fatalf("expected valid JSON to pass through; got err=%v", err)
	}
	if kind != RepairKindNone {
		t.Fatalf("expected kind=None for already-valid JSON; got %v", kind)
	}
	if string(out) != string(in) {
		t.Fatalf("expected byte-identical passthrough; got %q", out)
	}
}

func TestRepairToolInputEmptyBecomesObject(t *testing.T) {
	out, kind, err := RepairToolInput(json.RawMessage(""))
	if err != nil {
		t.Fatalf("expected empty input to be repaired; got err=%v", err)
	}
	if kind != RepairKindLossless {
		t.Fatalf("expected kind=Lossless for empty input; got %v", kind)
	}
	if kind.LostData() {
		t.Fatalf("empty-input repair must not be classified as data loss")
	}
	if string(out) != "{}" {
		t.Fatalf("expected '{}'; got %q", out)
	}
}

func TestRepairToolInputEscapesRawControlChars(t *testing.T) {
	// A raw newline inside a string is invalid JSON; the repair pass should
	// escape it without losing any content.
	in := json.RawMessage("{\"code\": \"line1\nline2\"}")
	out, kind, err := RepairToolInput(in)
	if err != nil {
		t.Fatalf("expected control-char repair to succeed; got err=%v\nin=%q\nout=%q", err, in, out)
	}
	if kind != RepairKindLossless {
		t.Fatalf("control-char escape should be Lossless; got %v", kind)
	}
	if kind.LostData() {
		t.Fatal("control-char escape must not be classified as data loss")
	}
	if !json.Valid(out) {
		t.Fatalf("repaired bytes still aren't valid JSON: %q", out)
	}
}

func TestRepairToolInputClosesTruncatedString(t *testing.T) {
	// Simulates the kimi-k2 max_tokens truncation: open string, no closing
	// quote, no closing brace. The classifier MUST flag this as truncated
	// so the run loop refuses to execute the tool.
	in := json.RawMessage(`{"code": "return await tools.finish({summary: ` + "`huge dump...`")
	out, kind, err := RepairToolInput(in)
	if err != nil {
		t.Fatalf("expected truncation repair to succeed; got err=%v\nout=%q", err, out)
	}
	if kind != RepairKindTruncated {
		t.Fatalf("dangling-string repair should be Truncated; got %v", kind)
	}
	if !kind.LostData() {
		t.Fatalf("dangling-string repair must report LostData=true")
	}
	if !json.Valid(out) {
		t.Fatalf("repaired bytes still aren't valid JSON: %q", out)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("repaired bytes don't unmarshal: %v", err)
	}
	if _, ok := parsed["code"].(string); !ok {
		t.Fatalf("expected `code` field to survive repair; got %#v", parsed)
	}
}

func TestRepairToolInputClosesTrailingComma(t *testing.T) {
	in := json.RawMessage(`{"a":1,`)
	out, kind, err := RepairToolInput(in)
	if err != nil {
		t.Fatalf("expected trailing-comma repair to succeed; got err=%v", err)
	}
	if kind != RepairKindTruncated {
		t.Fatalf("dangling-object repair should be Truncated; got %v", kind)
	}
	if !json.Valid(out) {
		t.Fatalf("repaired bytes not valid JSON: %q", out)
	}
}

func TestRepairToolInputClosesNestedArray(t *testing.T) {
	in := json.RawMessage(`{"xs":[1,2,3`)
	out, kind, err := RepairToolInput(in)
	if err != nil {
		t.Fatalf("expected nested-array repair to succeed; got err=%v", err)
	}
	if kind != RepairKindTruncated {
		t.Fatalf("dangling-array repair should be Truncated; got %v", kind)
	}
	if !json.Valid(out) {
		t.Fatalf("repaired bytes not valid JSON: %q", out)
	}
}

func TestRepairToolInputBailsOnHopelessInput(t *testing.T) {
	// Unbalanced closes can't be patched safely — we'd be guessing intent.
	in := json.RawMessage(`}}}`)
	out, kind, err := RepairToolInput(in)
	if err == nil {
		t.Fatalf("expected repair to bail; got kind=%v out=%q", kind, out)
	}
	if !strings.Contains(err.Error(), "could not be repaired") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestSafeToolInputFallbackIsValid(t *testing.T) {
	if !json.Valid(SafeToolInputFallback()) {
		t.Fatalf("SafeToolInputFallback must be valid JSON")
	}
	if string(SafeToolInputFallback()) != "{}" {
		t.Fatalf("expected '{}'; got %q", SafeToolInputFallback())
	}
}

func TestToolInputRepairKindHelpers(t *testing.T) {
	cases := []struct {
		kind          ToolInputRepairKind
		wantRepaired  bool
		wantLostData  bool
	}{
		{RepairKindNone, false, false},
		{RepairKindLossless, true, false},
		{RepairKindTruncated, true, true},
	}
	for _, c := range cases {
		if c.kind.Repaired() != c.wantRepaired {
			t.Errorf("%v.Repaired() = %v, want %v", c.kind, c.kind.Repaired(), c.wantRepaired)
		}
		if c.kind.LostData() != c.wantLostData {
			t.Errorf("%v.LostData() = %v, want %v", c.kind, c.kind.LostData(), c.wantLostData)
		}
	}
}
