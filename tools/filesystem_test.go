package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	llm "github.com/proiceremo/ai-sdk"
)

// execTool is a thin convenience wrapper that calls a tool with a typed
// input struct and decodes the structured output into `out` (if provided).
func execTool[I any, O any](t *testing.T, tool llm.Tool, cwd string, in I, out *O) llm.ToolResult {
	t.Helper()
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("input round-trip: %v", err)
	}
	result := tool.Execute(llm.ToolContext{
		Context:          context.Background(),
		WorkingDirectory: cwd,
	}, m)
	if out != nil && result.StructuredOutput != nil {
		structured, err := json.Marshal(result.StructuredOutput)
		if err != nil {
			t.Fatalf("marshal structured output: %v", err)
		}
		if err := json.Unmarshal(structured, out); err != nil {
			t.Fatalf("decode structured output: %v\njson: %s", err, structured)
		}
	}
	return result
}

func textOutput(result llm.ToolResult) string {
	var b strings.Builder
	for _, block := range result.Output {
		if block.Type == llm.ContentBlockTypeText {
			b.WriteString(block.Text)
		}
	}
	return b.String()
}

// -----------------------------------------------------------------------------
// read
// -----------------------------------------------------------------------------

func TestRead_PlainText(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "hello.txt")
	if err := os.WriteFile(path, []byte("alpha\nbeta\ngamma\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out ReadResult
	result := execTool(t, NewReadTool(FileToolsOptions{}), dir, ReadInput{Path: "hello.txt"}, &out)
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	if !strings.Contains(out.Content, "alpha") || !strings.Contains(out.Content, "gamma") {
		t.Fatalf("content missing lines: %q", out.Content)
	}
	if out.Mode != "text" {
		t.Fatalf("mode = %q, want text", out.Mode)
	}
	if out.TotalLines == 0 {
		t.Fatalf("total_lines should be set, got %d", out.TotalLines)
	}
}

func TestRead_RejectsOversizeText(t *testing.T) {
	dir := t.TempDir()
	var b strings.Builder
	for i := 1; i <= 100; i++ {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	// MaxBytes=100 — file is well over that.
	result := execTool(t, NewReadTool(FileToolsOptions{MaxBytes: 100}), dir, ReadInput{Path: "big.txt"}, (*ReadResult)(nil))
	if result.Error == nil {
		t.Fatal("read should refuse oversize files and redirect to read_range")
	}
	if !strings.Contains(result.Error.Error(), "read_range") {
		t.Fatalf("error should redirect to read_range, got: %v", result.Error)
	}
}

func TestRead_BinaryRejected(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "blob.bin"), []byte{0x89, 'P', 'N', 'G', 0x00, 0x01, 0x02}, 0o644); err != nil {
		t.Fatal(err)
	}
	result := execTool(t, NewReadTool(FileToolsOptions{}), dir, ReadInput{Path: "blob.bin"}, (*ReadResult)(nil))
	if result.Error == nil {
		t.Fatal("expected error for binary file")
	}
}

// -----------------------------------------------------------------------------
// read_range
// -----------------------------------------------------------------------------

func TestReadRange_Slice(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("L1\nL2\nL3\nL4\nL5\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out ReadResult
	execTool(t, NewReadRangeTool(FileToolsOptions{}), dir, ReadRangeInput{Path: "f.txt", Offset: 2, Limit: 2}, &out)
	if !strings.Contains(out.Content, "L2") || !strings.Contains(out.Content, "L3") {
		t.Fatalf("expected L2..L3 in content, got %q", out.Content)
	}
	if strings.Contains(out.Content, "L4") {
		t.Fatalf("L4 should be outside the requested range, got %q", out.Content)
	}
	if out.NextOffset != 4 {
		t.Fatalf("next_offset = %d, want 4", out.NextOffset)
	}
}

func TestReadRange_RequiresOffset(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("only\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result := execTool(t, NewReadRangeTool(FileToolsOptions{}), dir, ReadRangeInput{Path: "f.txt"}, (*ReadResult)(nil))
	if result.Error == nil {
		t.Fatal("read_range without offset should error")
	}
	if !strings.Contains(result.Error.Error(), "offset") {
		t.Fatalf("error should mention offset, got: %v", result.Error)
	}
}

func TestReadRange_OffsetPastEnd(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("only\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result := execTool(t, NewReadRangeTool(FileToolsOptions{}), dir, ReadRangeInput{Path: "a.txt", Offset: 99}, (*ReadResult)(nil))
	if result.Error == nil {
		t.Fatal("expected error for offset past end")
	}
}

func TestReadRange_TruncationByBytes(t *testing.T) {
	dir := t.TempDir()
	var b strings.Builder
	for i := 0; i < 600; i++ {
		b.WriteString("abcdefghij\n")
	}
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	var out ReadResult
	execTool(t, NewReadRangeTool(FileToolsOptions{MaxBytes: 100}), dir, ReadRangeInput{Path: "big.txt", Offset: 1}, &out)
	if out.Truncation == nil || !out.Truncation.Truncated {
		t.Fatalf("expected truncation, got %+v", out.Truncation)
	}
	if out.NextOffset == 0 {
		t.Fatal("expected next_offset to be set for continuation")
	}
}

// TestReadRange_SliceDoesNotPretendToBeFullFile is the regression for the
// schemas/to_model.go destruction in sess_a9c7d6a15319706c. The tool-name
// split is the structural fix — read_range can no longer be confused with
// a full read because it is a different tool. We assert the result still
// has TotalLines populated so the model can compare and notice.
func TestReadRange_PartialResultIsLabeled(t *testing.T) {
	dir := t.TempDir()
	var b strings.Builder
	for i := 1; i <= 100; i++ {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	var out ReadResult
	execTool(t, NewReadRangeTool(FileToolsOptions{}), dir, ReadRangeInput{Path: "big.txt", Offset: 1, Limit: 30}, &out)
	if out.TotalLines == 0 || out.EndLine == 0 || out.NextOffset == 0 {
		t.Fatalf("read_range result should carry total_lines/end_line/next_offset: %+v", out)
	}
	if out.TotalLines < 100 {
		t.Fatalf("expected total_lines >= 100, got %d", out.TotalLines)
	}
}

// -----------------------------------------------------------------------------
// edit
// -----------------------------------------------------------------------------

func TestEdit_UniqueReplace(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.go")
	if err := os.WriteFile(path, []byte("func Foo() {}\nfunc Bar() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out EditResult
	result := execTool(t, NewEditTool(FileToolsOptions{}), dir, EditInput{
		Path:  "x.go",
		Edits: []EditChange{{OldText: "func Foo() {}", NewText: "func Foo() error { return nil }"}},
	}, &out)
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	final, _ := os.ReadFile(path)
	if !strings.Contains(string(final), "func Foo() error") || !strings.Contains(string(final), "func Bar() {}") {
		t.Fatalf("file content mismatch: %s", final)
	}
	if out.EditsApplied != 1 {
		t.Fatalf("edits_applied = %d", out.EditsApplied)
	}
	if !strings.Contains(out.Diff, "func Foo() error") {
		t.Fatalf("diff missing new text: %q", out.Diff)
	}
}

func TestEdit_RejectsAmbiguous(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "y.go"), []byte("x := 1\ny := 1\nx := 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result := execTool(t, NewEditTool(FileToolsOptions{}), dir, EditInput{
		Path:  "y.go",
		Edits: []EditChange{{OldText: "x := 1", NewText: "x := 2"}},
	}, (*EditResult)(nil))
	if result.Error == nil {
		t.Fatal("expected ambiguity error")
	}
	if !strings.Contains(result.Error.Error(), "not unique") {
		t.Fatalf("error should mention uniqueness, got: %v", result.Error)
	}
}

func TestEdit_RejectsOverlapping(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "z.go"), []byte("hello world\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result := execTool(t, NewEditTool(FileToolsOptions{}), dir, EditInput{
		Path: "z.go",
		Edits: []EditChange{
			{OldText: "hello world", NewText: "hi world"},
			{OldText: "lo wor", NewText: "lo WOR"},
		},
	}, (*EditResult)(nil))
	if result.Error == nil {
		t.Fatal("expected overlap error")
	}
	if !strings.Contains(result.Error.Error(), "overlap") {
		t.Fatalf("error should mention overlap, got: %v", result.Error)
	}
}

func TestEdit_MultipleDisjoint(t *testing.T) {
	dir := t.TempDir()
	content := "alpha\nbeta\ngamma\ndelta\n"
	path := filepath.Join(dir, "m.txt")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	result := execTool(t, NewEditTool(FileToolsOptions{}), dir, EditInput{
		Path: "m.txt",
		Edits: []EditChange{
			{OldText: "alpha", NewText: "ALPHA"},
			{OldText: "delta", NewText: "DELTA"},
		},
	}, (*EditResult)(nil))
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	final, _ := os.ReadFile(path)
	if !strings.Contains(string(final), "ALPHA") || !strings.Contains(string(final), "DELTA") || !strings.Contains(string(final), "beta") {
		t.Fatalf("content mismatch: %s", final)
	}
}

func TestEdit_MissingFile(t *testing.T) {
	dir := t.TempDir()
	result := execTool(t, NewEditTool(FileToolsOptions{}), dir, EditInput{
		Path:  "does_not_exist.txt",
		Edits: []EditChange{{OldText: "a", NewText: "b"}},
	}, (*EditResult)(nil))
	if result.Error == nil {
		t.Fatal("expected error for missing file")
	}
}

// -----------------------------------------------------------------------------
// write
// -----------------------------------------------------------------------------

func TestWrite_CreatesParentDirs(t *testing.T) {
	dir := t.TempDir()
	var out WriteResult
	result := execTool(t, NewWriteTool(FileToolsOptions{}), dir, WriteInput{
		Path:    "nested/sub/dir/hi.txt",
		Content: "hi\n",
	}, &out)
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	if !out.Created {
		t.Fatal("created should be true for new file")
	}
	data, err := os.ReadFile(filepath.Join(dir, "nested/sub/dir/hi.txt"))
	if err != nil {
		t.Fatalf("file should exist: %v", err)
	}
	if string(data) != "hi\n" {
		t.Fatalf("content mismatch: %q", data)
	}
}

func TestWrite_Overwrites(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "old.txt")
	if err := os.WriteFile(path, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out WriteResult
	execTool(t, NewWriteTool(FileToolsOptions{}), dir, WriteInput{Path: "old.txt", Content: "new\n"}, &out)
	if out.Created {
		t.Fatal("created should be false when overwriting")
	}
	data, _ := os.ReadFile(path)
	if string(data) != "new\n" {
		t.Fatalf("expected overwrite, got %q", data)
	}
}

// -----------------------------------------------------------------------------
// grep
// -----------------------------------------------------------------------------

func TestGrep_RegexMatches(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.go"), []byte("package main\n\nfunc Foo() {}\nfunc Bar() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "b.go"), []byte("package main\n\nfunc Baz() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out GrepResult
	execTool(t, NewGrepTool(FileToolsOptions{}), dir, GrepInput{Pattern: "func B[a-z]+"}, &out)
	if out.Total < 2 {
		t.Fatalf("expected at least 2 matches, got %d (%+v)", out.Total, out.Matches)
	}
	body := textOutput(execTool(t, NewGrepTool(FileToolsOptions{}), dir, GrepInput{Pattern: "func B[a-z]+"}, (*GrepResult)(nil)))
	if !strings.Contains(body, "a.go") || !strings.Contains(body, "b.go") {
		t.Fatalf("output missing file refs: %q", body)
	}
}

func TestGrep_LiteralFlag(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("price is $9.99\nfunc f() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out GrepResult
	execTool(t, NewGrepTool(FileToolsOptions{}), dir, GrepInput{Pattern: "$9.99", Literal: true}, &out)
	if out.Total != 1 {
		t.Fatalf("expected 1 literal match, got %d", out.Total)
	}
}

func TestGrep_HonorsGitignore(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte("ignored/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "ignored"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ignored", "x.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "kept.txt"), []byte("needle\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out GrepResult
	execTool(t, NewGrepTool(FileToolsOptions{}), dir, GrepInput{Pattern: "needle"}, &out)
	for _, m := range out.Matches {
		if strings.Contains(m.Path, "ignored") {
			t.Fatalf("match leaked through .gitignore: %+v", m)
		}
	}
	// With noIgnore the ignored file should now appear.
	var withIgnored GrepResult
	execTool(t, NewGrepTool(FileToolsOptions{}), dir, GrepInput{Pattern: "needle", NoIgnore: true}, &withIgnored)
	found := false
	for _, m := range withIgnored.Matches {
		if strings.Contains(m.Path, "ignored") {
			found = true
		}
	}
	if !found {
		t.Fatal("noIgnore should expose ignored matches")
	}
}

func TestGrep_GlobFilter(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.go"), []byte("needle in go file\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "a.ts"), []byte("needle in ts file\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out GrepResult
	execTool(t, NewGrepTool(FileToolsOptions{}), dir, GrepInput{Pattern: "needle", Glob: "*.go"}, &out)
	if out.Total != 1 {
		t.Fatalf("expected 1 match limited to *.go, got %d (%+v)", out.Total, out.Matches)
	}
}

// -----------------------------------------------------------------------------
// find
// -----------------------------------------------------------------------------

func TestFind_GlobPattern(t *testing.T) {
	dir := t.TempDir()
	for _, p := range []string{"src/a.go", "src/b.go", "src/c.ts", "docs/readme.md"} {
		full := filepath.Join(dir, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var out FindResult
	execTool(t, NewFindTool(FileToolsOptions{}), dir, FindInput{Pattern: "**/*.go"}, &out)
	if out.Total != 2 {
		t.Fatalf("expected 2 .go files, got %d (%+v)", out.Total, out.Files)
	}
}

func TestFind_HonorsGitignore(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte("dist/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "dist"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "dist", "out.js"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "src.js"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out FindResult
	execTool(t, NewFindTool(FileToolsOptions{}), dir, FindInput{Pattern: "**/*.js"}, &out)
	for _, f := range out.Files {
		if strings.Contains(f, "dist/") {
			t.Fatalf("dist file should be ignored: %q", f)
		}
	}
}

// -----------------------------------------------------------------------------
// ls
// -----------------------------------------------------------------------------

func TestLs_SortsCaseInsensitiveWithSuffix(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "Aaa"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bbb.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".dot"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out LsResult
	execTool(t, NewLsTool(FileToolsOptions{}), dir, LsInput{Path: "."}, &out)
	if len(out.Entries) != 3 {
		t.Fatalf("expected 3 entries, got %d (%+v)", len(out.Entries), out.Entries)
	}
	names := []string{out.Entries[0].Name, out.Entries[1].Name, out.Entries[2].Name}
	if names[0] != ".dot" {
		t.Fatalf("expected .dot first, got %v", names)
	}
	// Aaa should sort case-insensitively before bbb.txt
	if names[1] != "Aaa" {
		t.Fatalf("expected Aaa second, got %v", names)
	}
	if !out.Entries[1].IsDir {
		t.Fatal("Aaa should be marked as dir")
	}
}

// -----------------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------------

func TestRegisterFileToolFactories_RegistersAllNames(t *testing.T) {
	reg := RegisterFileToolFactories(nil)
	configs := []llm.ToolConfig{}
	for _, name := range []string{"read", "read_range", "edit", "write", "grep", "find", "ls"} {
		configs = append(configs, llm.ToolConfig{ID: name})
	}
	got, err := reg.BuildTools(context.Background(), llm.ToolBuildContext{}, configs)
	if err != nil {
		t.Fatalf("BuildTools: %v", err)
	}
	if len(got) != len(configs) {
		t.Fatalf("expected %d tools, got %d", len(configs), len(got))
	}
}

// Grep MUST skip any single file larger than maxGrepFileSize. Without
// the pre-Stat check, an `os.ReadFile` on the 50-100 MiB proagent-eval
// binary loads it into the heap before isBinary fires — that's how
// Gemini Flash OOM-killed case-000027 with `tools.grep({path:"/"})`.
//
// The needle in the big file is something the regex would otherwise
// match, so a passing test PROVES the file was skipped (not "matched
// but truncated").
func TestGrep_SkipsLargeFiles(t *testing.T) {
	dir := t.TempDir()
	// Write a 2 MiB file containing the needle near the top. Above the
	// 1 MiB per-file cap, so grep must skip it entirely.
	big := make([]byte, 2<<20)
	copy(big, []byte("NEEDLE_GIANT_LINE\n"))
	if err := os.WriteFile(filepath.Join(dir, "huge.txt"), big, 0o644); err != nil {
		t.Fatal(err)
	}
	// Small companion file so the walk finds *something* and exercises
	// the per-match code path — confirms we don't blanket-skip.
	if err := os.WriteFile(filepath.Join(dir, "small.txt"), []byte("NEEDLE_GIANT_LINE\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out GrepResult
	execTool(t, NewGrepTool(FileToolsOptions{}), dir, GrepInput{Pattern: "NEEDLE_GIANT_LINE"}, &out)
	for _, m := range out.Matches {
		if strings.Contains(m.Path, "huge.txt") {
			t.Errorf("huge file should be skipped — got match in %q", m.Path)
		}
	}
	if !out.Truncated {
		t.Errorf("expected Truncated=true to flag the skip; got %+v", out)
	}
}

// System dirs (/usr, /lib, /proc, etc.) are skipped by walkSearchableFiles.
// This catches the Gemini Flash failure mode where `grep -r / NEEDLE`
// would read /usr/lib/* and OOM the container.
func TestGrep_SkipsSystemDirs(t *testing.T) {
	dir := t.TempDir()
	// Fake /usr/, /proc/, /sys/ subtrees under the test root. The walk
	// uses dirname matching (not absolute-path matching), so these
	// fake subtrees exercise the same code path as `/usr` would in
	// production.
	for _, sub := range []string{"usr", "lib", "proc", "sys", "var", "etc"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, sub, "secret.txt"), []byte("PROBE_NEEDLE\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Workspace file the agent legitimately searches.
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("PROBE_NEEDLE\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out GrepResult
	execTool(t, NewGrepTool(FileToolsOptions{}), dir, GrepInput{Pattern: "PROBE_NEEDLE"}, &out)
	for _, m := range out.Matches {
		for _, sub := range []string{"usr/", "lib/", "proc/", "sys/", "var/", "etc/"} {
			if strings.Contains(m.Path, sub) {
				t.Errorf("system dir %q must be skipped; matched %q", sub, m.Path)
			}
		}
	}
	// Workspace-side match should still be present — the skip list
	// must not over-exclude.
	hasMain := false
	for _, m := range out.Matches {
		if strings.Contains(m.Path, "main.go") {
			hasMain = true
			break
		}
	}
	if !hasMain {
		t.Errorf("legitimate workspace match missing; got %+v", out.Matches)
	}
}

// isNoisyDir now covers system-level Unix dirs in addition to dev
// tooling. Lock the set so future contributors don't accidentally drop
// /proc or /usr from the list and reintroduce the OOM.
func TestIsNoisyDir_CoversSystemDirs(t *testing.T) {
	required := []string{
		// Dev tooling (pre-existing).
		"node_modules", ".git", "vendor", "__pycache__",
		// System (added 2026-05 after case-000027 OOM).
		"proc", "sys", "dev", "usr", "lib", "var", "etc", "bin", "sbin", "opt",
	}
	for _, name := range required {
		if !isNoisyDir(name) {
			t.Errorf("isNoisyDir(%q) returned false — regression risk", name)
		}
	}
}
