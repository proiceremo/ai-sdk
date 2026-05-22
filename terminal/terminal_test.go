package terminal

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	llm "ai-sdk"
)

func TestHarmlessCommandLineAllowsReadAndSimpleCommands(t *testing.T) {
	allowed := []string{
		"ls -la",
		"cat README.md",
		"sed -n '1,20p' README.md",
		"find . -maxdepth 2 -type f",
		"echo hello",
		"git status --short",
		"git diff -- README.md",
		"ls -la | head -n 5",
		"pwd && ls",
		"cat README.md 2>/dev/null || echo 'No README.md'",
		"cat README.md 2> /dev/null || echo 'No README.md'",
		"cat README.md >/dev/null && echo done",
		// Quote-aware pipeline parsing: the `|` characters inside the sed
		// expression must not be treated as pipeline separators.
		"find . -type f -not -path './.git/*' | sed 's|^./||' | cut -d/ -f1 | sort | uniq -c | sort -rn",
		`echo "hello | world" | wc -l`,
		`echo 'a;b;c' | tr ';' '\n'`,
		// stdin-from-file redirect is read-only.
		"wc -c < README.md",
		"head -n 5 < /etc/hosts | tr -d ' '",
		// $(...) substitution of read-only commands is itself read-only.
		"echo $(wc -l < README.md)",
		"echo \"size: $(stat -c %s README.md)\"",
		// Nested $(...) with read-only inner commands.
		"echo $(echo $(wc -c < README.md))",
		// Real-world: catalogue every file with size + first line.
		"find . -type f -not -path './.git/*' -not -path './node_modules/*' | sort | while read f; do\n  size=$(wc -c < \"$f\" | tr -d ' ')\n  firstline=$(head -n1 \"$f\" 2>/dev/null | tr '\\n' ' ')\n  echo \"$size|$f|$firstline\"\ndone",
		// Leading variable assignment is harmless on its own.
		"FOO=bar BAZ=qux echo $FOO",
		"FOO=bar",
	}
	for _, cmd := range allowed {
		if !isHarmlessCommandLine(cmd) {
			t.Fatalf("expected %q to be harmless", cmd)
		}
	}
}

func TestHarmlessCommandLineRejectsMutatingOrDynamicCommands(t *testing.T) {
	rejected := []string{
		"rm -rf build",
		"python script.py",
		"go test ./...",
		"git reset --hard",
		"cat README.md > out.txt",
		"echo $(rm -rf build)",
		"cat README.md 2> err.log",
		// File-mutating commands previously misclassified as harmless.
		"cp README.md /tmp/README.copy",
		"mkdir build",
		"touch new.txt",
		// Destructive flags on otherwise-read-only commands.
		"sed -i 's/foo/bar/' file.txt",
		"sed --in-place 's/foo/bar/' file.txt",
		"find . -name '*.log' -delete",
		"find . -name '*.log' -exec rm {} ;",
		// Legacy backtick substitution — opaque, always rejected.
		"echo `rm -rf build`",
		"echo \"current: `whoami`\"",
		// Process substitution can spawn arbitrary subshells.
		"diff <(cat foo) <(rm bar)",
		"tee >(rm -rf build) < input",
		// $(...) substitution of a NON-harmless inner command propagates.
		"echo $(rm -rf build)",
		"echo \"oops: $(rm bad)\"",
		// Output redirect to a real path (not /dev/null) is a write.
		"echo hi > out.txt",
		"cat README.md > saved.md",
	}
	for _, cmd := range rejected {
		if isHarmlessCommandLine(cmd) {
			t.Fatalf("expected %q to require permission", cmd)
		}
	}
}

func TestTerminalPermissionGuardsSplitCompoundCommands(t *testing.T) {
	guards := terminalPermissionGuards("cat test > test.txt && rm test", "/repo")
	if len(guards) != 2 {
		t.Fatalf("expected cat and rm guards, got %#v", guards)
	}
	if guards[0].Specifiers[0] != "command:cat test > test.txt" || guards[1].Specifiers[0] != "command:rm test" {
		t.Fatalf("unexpected command specifiers: %#v", guards)
	}
	for _, guard := range guards {
		if guard.Key != "terminal.run" {
			t.Fatalf("unexpected guard key: %#v", guard)
		}
		if guard.Specifiers[1] != "working_dir:/repo" {
			t.Fatalf("unexpected working dir specifier: %#v", guard)
		}
		if len(guard.Options) == 0 {
			t.Fatalf("expected custom permission options: %#v", guard)
		}
	}
}

func TestRunReturnsOutputForSynchronousCommand(t *testing.T) {
	store := newTestStore(t)
	manager := NewManager(store)
	tool := NewTool(manager)
	result := tool.Execute(llm.ToolContext{
		Context:          context.Background(),
		WorkingDirectory: t.TempDir(),
		Vars:             map[string]any{"session_id": "sess_test"},
	}, map[string]any{
		"action":  "run",
		"command": "echo hello",
	})
	if result.Error != nil {
		t.Fatal(result.Error)
	}
	output, ok := result.StructuredOutput.(map[string]any)
	if !ok {
		t.Fatalf("expected map structured output, got %#v", result.StructuredOutput)
	}
	if strings.TrimSpace(output["output"].(string)) != "hello" {
		t.Fatalf("unexpected output: %#v", output["output"])
	}
	terminal, ok := output["terminal"].(Terminal)
	if !ok {
		t.Fatalf("expected terminal in output, got %#v", output["terminal"])
	}
	if terminal.Status != statusIdle {
		t.Fatalf("expected command to be complete and shell idle, got %s", terminal.Status)
	}
	if got, ok := output["exit_code"].(int); !ok || got != 0 {
		t.Fatalf("expected command exit_code 0, got %#v", output["exit_code"])
	}
}

func TestStatefulPersistentShellReuse(t *testing.T) {
	store := newTestStore(t)
	manager := NewManager(store)
	tool := NewTool(manager)

	ctx := llm.ToolContext{
		Context:          context.Background(),
		WorkingDirectory: t.TempDir(),
		Vars:             map[string]any{"session_id": "sess_stateful_test"},
	}

	// 1. Start a fresh persistent shell and export a variable
	result1 := tool.Execute(ctx, map[string]any{
		"action":  "run",
		"command": "export MY_TEST_VAR=antigravity_rocks",
	})
	if result1.Error != nil {
		t.Fatal(result1.Error)
	}

	output1, ok := result1.StructuredOutput.(map[string]any)
	if !ok {
		t.Fatalf("expected map, got %#v", result1.StructuredOutput)
	}
	terminalID := output1["terminal_id"].(string)

	// 2. Reuse the same shell via terminal_id and assert the exported variable is preserved!
	result2 := tool.Execute(ctx, map[string]any{
		"action":      "run",
		"command":     "echo $MY_TEST_VAR",
		"terminal_id": terminalID,
	})
	if result2.Error != nil {
		t.Fatal(result2.Error)
	}

	output2, ok := result2.StructuredOutput.(map[string]any)
	if !ok {
		t.Fatalf("expected map, got %#v", result2.StructuredOutput)
	}

	outStr := strings.TrimSpace(output2["output"].(string))
	if outStr != "antigravity_rocks" {
		t.Fatalf("expected environment variable to be preserved, got %q", outStr)
	}
}

func TestAsyncCommandBecomesIdleAfterSentinel(t *testing.T) {
	store := newTestStore(t)
	manager := NewManager(store)
	tool := NewTool(manager)
	ctx := llm.ToolContext{
		Context:          context.Background(),
		WorkingDirectory: t.TempDir(),
		Vars:             map[string]any{"session_id": "sess_async_test"},
	}

	start := tool.Execute(ctx, map[string]any{
		"action":  "run",
		"command": "echo quick-async",
		"async":   true,
	})
	if start.Error != nil {
		t.Fatal(start.Error)
	}
	terminalID := start.StructuredOutput.(map[string]any)["terminal_id"].(string)

	var term Terminal
	for deadline := time.Now().Add(2 * time.Second); time.Now().Before(deadline); {
		var err error
		term, err = manager.Status("sess_async_test", terminalID)
		if err != nil {
			t.Fatal(err)
		}
		if term.Status == statusIdle {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if term.Status != statusIdle {
		t.Fatalf("expected async command to settle back to idle, got %#v", term)
	}
	if term.ExitCode == nil || *term.ExitCode != 0 {
		t.Fatalf("expected async command exit code 0, got %#v", term.ExitCode)
	}
	out, _, err := manager.Output("sess_async_test", terminalID, 0, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stripSentinel(out), "quick-async") {
		t.Fatalf("expected async command output, got %q", out)
	}
}

func TestWaitReturnsWhenAsyncCommandCompletes(t *testing.T) {
	store := newTestStore(t)
	manager := NewManager(store)
	tool := NewTool(manager)
	ctx := llm.ToolContext{
		Context:          context.Background(),
		WorkingDirectory: t.TempDir(),
		Vars:             map[string]any{"session_id": "sess_wait_test"},
	}

	start := tool.Execute(ctx, map[string]any{
		"action":  "run",
		"command": "sleep 0.1; echo waited",
		"async":   true,
	})
	if start.Error != nil {
		t.Fatal(start.Error)
	}
	terminalID := start.StructuredOutput.(map[string]any)["terminal_id"].(string)

	wait := tool.Execute(ctx, map[string]any{
		"action":      "wait",
		"terminal_id": terminalID,
		"timeout":     2,
	})
	if wait.Error != nil {
		t.Fatal(wait.Error)
	}
	term := wait.StructuredOutput.(Terminal)
	if term.Status != statusIdle {
		t.Fatalf("expected wait to return after command completion, got %#v", term)
	}
	if term.ExitCode == nil || *term.ExitCode != 0 {
		t.Fatalf("expected exit code 0, got %#v", term.ExitCode)
	}
	out, _, err := manager.Output("sess_wait_test", terminalID, 0, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stripSentinel(out), "waited") {
		t.Fatalf("expected command output, got %q", out)
	}
}

func TestActiveTerminalsHeaderIsCapped(t *testing.T) {
	manager := NewManager(newTestStore(t))
	for i := 0; i < 8; i++ {
		id := fmt.Sprintf("term_%02d", i)
		manager.procs[id] = &procEntry{
			term: &Terminal{
				ID:      id,
				ScopeID: "sess_test",
				Command: fmt.Sprintf("sleep %d", i),
				Status:  "running",
			},
		}
	}
	manager.procs["other_scope"] = &procEntry{
		term: &Terminal{ID: "other_scope", ScopeID: "sess_other", Command: "sleep 99", Status: "running"},
	}
	manager.procs["exited"] = &procEntry{
		term: &Terminal{ID: "exited", ScopeID: "sess_test", Command: "sleep 99", Status: "exited"},
	}
	manager.procs["idle"] = &procEntry{
		term: &Terminal{ID: "idle", ScopeID: "sess_test", Command: "echo done", Status: statusIdle},
	}

	payload := map[string]any{}
	injectActiveTerminalsHeader(payload, manager, "sess_test", "term_00")

	if got := payload["active_background_session_count"]; got != 7 {
		t.Fatalf("expected 7 active background sessions, got %#v", got)
	}
	if got := payload["active_background_sessions_truncated"]; got != true {
		t.Fatalf("expected truncated marker, got %#v", got)
	}
	if got := payload["active_background_sessions_omitted"]; got != 2 {
		t.Fatalf("expected 2 omitted sessions, got %#v", got)
	}
	preview, ok := payload["active_background_sessions"].(string)
	if !ok {
		t.Fatalf("expected active_background_sessions string, got %#v", payload["active_background_sessions"])
	}
	if strings.Contains(preview, "term_00") || strings.Contains(preview, "other_scope") || strings.Contains(preview, "exited") || strings.Contains(preview, "idle") {
		t.Fatalf("preview included excluded sessions: %q", preview)
	}
	if strings.Count(preview, "term_") != activeTerminalPreviewLimit {
		t.Fatalf("expected capped preview of %d sessions, got %q", activeTerminalPreviewLimit, preview)
	}
}

type testStore struct {
	root string
}

func newTestStore(t *testing.T) *testStore {
	t.Helper()
	return &testStore{root: t.TempDir()}
}

func (s *testStore) SaveManifest(ctx context.Context, term Terminal) error {
	return nil
}

func (s *testStore) LoadManifest(ctx context.Context, scopeID, terminalID string) (Terminal, error) {
	return Terminal{ID: terminalID, ScopeID: scopeID, Status: "exited"}, nil
}

func (s *testStore) ListRunning(ctx context.Context) ([]Terminal, error) {
	return nil, nil
}

func (s *testStore) LogPath(scopeID, terminalID string) string {
	return filepath.Join(s.root, scopeID, terminalID+".log")
}

var _ Store = (*testStore)(nil)
