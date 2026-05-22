package terminal

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	llm "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/tools"
)

type Terminal struct {
	ID        string     `json:"id"`
	ScopeID   string     `json:"scope_id"`
	Command   string     `json:"command"`
	Cwd       string     `json:"cwd"`
	PID       int        `json:"pid"`
	PGID      int        `json:"pgid"`
	Status    string     `json:"status"`
	ExitCode  *int       `json:"exit_code,omitempty"`
	StartedAt time.Time  `json:"started_at"`
	EndedAt   *time.Time `json:"ended_at,omitempty"`
}

type Store interface {
	SaveManifest(ctx context.Context, term Terminal) error
	LoadManifest(ctx context.Context, scopeID, terminalID string) (Terminal, error)
	ListRunning(ctx context.Context) ([]Terminal, error)
	LogPath(scopeID, terminalID string) string
}

type Manager struct {
	store Store
	mu    sync.Mutex
	procs map[string]*procEntry
}

type procEntry struct {
	term   *Terminal
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.Reader
	done   chan struct{}
}

const activeTerminalPreviewLimit = 5

const (
	statusIdle        = "idle"
	statusRunning     = "running"
	statusExited      = "exited"
	statusInterrupted = "interrupted"
	statusCancelled   = "cancelled"
)

func NewManager(store Store) *Manager {
	return &Manager{store: store, procs: map[string]*procEntry{}}
}

func (m *Manager) Reconcile(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	terms, err := m.store.ListRunning(ctx)
	if err != nil {
		return err
	}
	for _, t := range terms {
		if !isProcessAlive(t.PID) {
			t.Status = statusInterrupted
			now := time.Now()
			t.EndedAt = &now
			_ = m.store.SaveManifest(ctx, t)
		}
	}
	return nil
}

func isProcessAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	return syscall.Kill(pid, 0) == nil
}

type StartRequest struct {
	ScopeID string
	Command string
	Cwd     string
	Env     []string
	Async   bool
	Timeout time.Duration
}

var ErrCommandTimeout = errors.New("timeout waiting for command to finish")

func randHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func resolveShell() string {
	if runtime.GOOS == "windows" {
		if comSpec := os.Getenv("COMSPEC"); comSpec != "" {
			if _, err := os.Stat(comSpec); err == nil {
				return comSpec
			}
		}
		if path, err := exec.LookPath("cmd.exe"); err == nil {
			return path
		}
		if path, err := exec.LookPath("powershell.exe"); err == nil {
			return path
		}
		return "cmd.exe"
	}

	if shell := os.Getenv("SHELL"); shell != "" {
		if _, err := os.Stat(shell); err == nil {
			return shell
		}
	}
	if path, err := exec.LookPath("bash"); err == nil {
		return path
	}
	if path, err := exec.LookPath("zsh"); err == nil {
		return path
	}
	if path, err := exec.LookPath("sh"); err == nil {
		return path
	}
	return "/bin/sh"
}

func (m *Manager) Start(req StartRequest) (Terminal, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	id := "term_" + randHex(8)
	logPath := m.store.LogPath(req.ScopeID, id)
	_ = os.MkdirAll(filepath.Dir(logPath), 0o755)

	// Create and clear log file
	_ = os.WriteFile(logPath, []byte{}, 0o644)

	shell := resolveShell()
	cmd := exec.Command(shell)
	cmd.Dir = req.Cwd
	cmd.Env = append(os.Environ(), req.Env...)

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return Terminal{}, err
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return Terminal{}, err
	}
	cmd.Stderr = cmd.Stdout
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		return Terminal{}, err
	}

	term := Terminal{
		ID:        id,
		ScopeID:   req.ScopeID,
		Command:   req.Command,
		Cwd:       req.Cwd,
		PID:       cmd.Process.Pid,
		PGID:      cmd.Process.Pid,
		Status:    statusIdle,
		StartedAt: time.Now(),
	}

	entry := &procEntry{
		term:   &term,
		cmd:    cmd,
		stdin:  stdinPipe,
		stdout: stdoutPipe,
		done:   make(chan struct{}),
	}
	m.procs[id] = entry
	_ = m.store.SaveManifest(context.Background(), term)

	// Background reader goroutine to continually stream all shell output directly to the log file live!
	go func() {
		buf := make([]byte, 4096)
		for {
			n, err := stdoutPipe.Read(buf)
			if n > 0 {
				logFile, oErr := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
				if oErr == nil {
					_, _ = logFile.Write(buf[:n])
					logFile.Close()
				}
			}
			if err != nil {
				break
			}
		}
	}()

	// Background exit waiter goroutine
	go func() {
		_ = cmd.Wait()
		m.mu.Lock()
		defer m.mu.Unlock()
		if term.Status != statusCancelled && term.Status != statusInterrupted {
			term.Status = statusExited
		}
		code := 0
		if cmd.ProcessState != nil {
			code = cmd.ProcessState.ExitCode()
			term.ExitCode = &code
		}
		now := time.Now()
		term.EndedAt = &now
		_ = m.store.SaveManifest(context.Background(), term)
		delete(m.procs, id)
		close(entry.done)
	}()

	// Execute initial command if provided
	if strings.TrimSpace(req.Command) != "" {
		m.markCommandStartedLocked(entry, req.Command)
		if req.Async {
			sentinel := randTerminalSentinel()
			if _, err := stdinPipe.Write([]byte(shellCommandWithSentinel(shell, req.Command, sentinel))); err != nil {
				entry.term.Status = statusIdle
				_ = m.store.SaveManifest(context.Background(), *entry.term)
				return *entry.term, err
			}
			go m.monitorSentinel(req.ScopeID, id, logPath, sentinel)
		} else {
			// Write command + sentinel and block until completed
			sentinel := randTerminalSentinel()
			cmdStr := shellCommandWithSentinel(shell, req.Command, sentinel)
			go m.monitorSentinel(req.ScopeID, id, logPath, sentinel)
			_, err = stdinPipe.Write([]byte(cmdStr))
			if err != nil {
				entry.term.Status = statusIdle
				_ = m.store.SaveManifest(context.Background(), *entry.term)
				return *entry.term, err
			}
			m.mu.Unlock()
			timeout := req.Timeout
			if timeout <= 0 {
				timeout = 15 * time.Second
			}
			exitCode, _ := m.WaitForSentinel(context.Background(), logPath, sentinel, timeout)
			m.mu.Lock()
			if exitCode != nil {
				m.markCommandCompleteLocked(entry, *exitCode)
			}
		}
	}

	return *entry.term, nil
}

func randTerminalSentinel() string {
	return "---PTY-DONE-UUID-" + randHex(16) + "---"
}

func shellCommandWithSentinel(shell, command, sentinel string) string {
	if strings.Contains(shell, "powershell") {
		return command + "; $proStatus = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { 0 }; Write-Output \"\"; Write-Output \"" + sentinel + ":$proStatus\"\r\n"
	}
	if strings.Contains(shell, "cmd") {
		return command + " & echo. & echo " + sentinel + ":%ERRORLEVEL%\r\n"
	}
	return command + "\n__proagent_status=$?\nprintf '\\n%s:%s\\n' '" + sentinel + "' \"$__proagent_status\"\n"
}

func parseSentinelExitCode(data []byte, sentinel string) (*int, bool) {
	idx := bytes.Index(data, []byte(sentinel))
	if idx < 0 {
		return nil, false
	}
	start := idx + len(sentinel)
	if start >= len(data) || data[start] != ':' {
		code := 0
		return &code, true
	}
	start++
	end := start
	for end < len(data) && data[end] >= '0' && data[end] <= '9' {
		end++
	}
	code := 0
	if end > start {
		if parsed, err := strconv.Atoi(string(data[start:end])); err == nil {
			code = parsed
		}
	}
	return &code, true
}

func (m *Manager) WaitForSentinel(ctx context.Context, logPath, sentinel string, timeout time.Duration) (*int, error) {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()

	deadline := time.Now().Add(timeout)
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			if time.Now().After(deadline) {
				return nil, ErrCommandTimeout
			}
			data, err := os.ReadFile(logPath)
			if err == nil {
				if exitCode, ok := parseSentinelExitCode(data, sentinel); ok {
					return exitCode, nil
				}
			}
		}
	}
}

func (m *Manager) monitorSentinel(scopeID, tid, logPath, sentinel string) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		m.mu.Lock()
		entry, ok := m.procs[tid]
		done := (<-chan struct{})(nil)
		if ok {
			done = entry.done
		}
		m.mu.Unlock()
		if !ok {
			return
		}
		select {
		case <-done:
			return
		case <-ticker.C:
			data, err := os.ReadFile(logPath)
			if err != nil {
				continue
			}
			exitCode, found := parseSentinelExitCode(data, sentinel)
			if !found || exitCode == nil {
				continue
			}
			m.mu.Lock()
			if entry, ok := m.procs[tid]; ok && entry.term.ScopeID == scopeID {
				m.markCommandCompleteLocked(entry, *exitCode)
			}
			m.mu.Unlock()
			return
		}
	}
}

func (m *Manager) markCommandStartedLocked(entry *procEntry, command string) {
	if entry == nil || entry.term == nil {
		return
	}
	entry.term.Command = command
	entry.term.Status = statusRunning
	entry.term.ExitCode = nil
	entry.term.EndedAt = nil
	_ = m.store.SaveManifest(context.Background(), *entry.term)
}

func (m *Manager) markCommandCompleteLocked(entry *procEntry, exitCode int) {
	if entry == nil || entry.term == nil || entry.term.Status != statusRunning {
		return
	}
	entry.term.Status = statusIdle
	entry.term.ExitCode = &exitCode
	now := time.Now()
	entry.term.EndedAt = &now
	_ = m.store.SaveManifest(context.Background(), *entry.term)
}

func (m *Manager) ExecuteCommand(ctx context.Context, scopeID, tid, command string, async bool, timeout time.Duration) (Terminal, error) {
	m.mu.Lock()
	p, ok := m.procs[tid]
	m.mu.Unlock()
	if !ok {
		return Terminal{}, fmt.Errorf("terminal session %s is not active", tid)
	}

	logPath := m.store.LogPath(scopeID, tid)
	shell := resolveShell()

	if async {
		m.mu.Lock()
		m.markCommandStartedLocked(p, command)
		m.mu.Unlock()
		sentinel := randTerminalSentinel()
		_, err := p.stdin.Write([]byte(shellCommandWithSentinel(shell, command, sentinel)))
		if err != nil {
			m.mu.Lock()
			p.term.Status = statusIdle
			_ = m.store.SaveManifest(context.Background(), *p.term)
			m.mu.Unlock()
			return *p.term, err
		}
		go m.monitorSentinel(scopeID, tid, logPath, sentinel)
		return *p.term, nil
	}

	m.mu.Lock()
	m.markCommandStartedLocked(p, command)
	m.mu.Unlock()
	sentinel := randTerminalSentinel()
	cmdStr := shellCommandWithSentinel(shell, command, sentinel)
	go m.monitorSentinel(scopeID, tid, logPath, sentinel)

	_, err := p.stdin.Write([]byte(cmdStr))
	if err != nil {
		m.mu.Lock()
		p.term.Status = statusIdle
		_ = m.store.SaveManifest(context.Background(), *p.term)
		m.mu.Unlock()
		return *p.term, err
	}

	exitCode, err := m.WaitForSentinel(ctx, logPath, sentinel, timeout)
	if err != nil {
		if errors.Is(err, ErrCommandTimeout) {
			return *p.term, nil
		}
		return *p.term, err
	}
	if exitCode != nil {
		m.mu.Lock()
		m.markCommandCompleteLocked(p, *exitCode)
		m.mu.Unlock()
	}

	return *p.term, nil
}

func (m *Manager) List(scopeID string) ([]Terminal, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result []Terminal
	for _, p := range m.procs {
		if p.term.ScopeID == scopeID {
			result = append(result, *p.term)
		}
	}

	if m.store != nil {
		dbTerms, err := m.store.ListRunning(context.Background())
		if err == nil {
			seen := make(map[string]bool)
			for _, t := range result {
				seen[t.ID] = true
			}
			for _, t := range dbTerms {
				if t.ScopeID == scopeID && !seen[t.ID] {
					result = append(result, t)
				}
			}
		}
	}
	return result, nil
}

func (m *Manager) Status(scopeID, tid string) (Terminal, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if p, ok := m.procs[tid]; ok {
		return *p.term, nil
	}
	return m.store.LoadManifest(context.Background(), scopeID, tid)
}

func (m *Manager) Wait(ctx context.Context, scopeID, tid string, timeout time.Duration) (Terminal, error) {
	m.mu.Lock()
	p, ok := m.procs[tid]
	m.mu.Unlock()
	if !ok {
		return m.store.LoadManifest(ctx, scopeID, tid)
	}

	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		term, err := m.Status(scopeID, tid)
		if err != nil {
			return term, err
		}
		if term.Status != statusRunning {
			return term, nil
		}
		select {
		case <-p.done:
			return m.Status(scopeID, tid)
		case <-ticker.C:
		case <-timer.C:
			return term, nil
		case <-ctx.Done():
			return term, ctx.Err()
		}
	}
}

func (m *Manager) Output(scopeID, tid string, offset, limit int) (string, int, error) {
	logPath := m.store.LogPath(scopeID, tid)
	data, err := os.ReadFile(logPath)
	if err != nil {
		return "", 0, err
	}
	if offset >= len(data) {
		return "", len(data), nil
	}
	end := offset + limit
	if end > len(data) {
		end = len(data)
	}
	return string(data[offset:end]), len(data), nil
}

func (m *Manager) Cancel(scopeID, tid string) error {
	m.mu.Lock()
	p, ok := m.procs[tid]
	if ok && p.term != nil {
		p.term.Status = statusCancelled
		now := time.Now()
		p.term.EndedAt = &now
		_ = m.store.SaveManifest(context.Background(), *p.term)
	}
	m.mu.Unlock()
	if !ok {
		return nil
	}
	_ = p.stdin.Close()
	if p.term.PGID > 0 {
		_ = syscall.Kill(-p.term.PGID, syscall.SIGTERM)
		time.AfterFunc(2*time.Second, func() {
			_ = syscall.Kill(-p.term.PGID, syscall.SIGKILL)
		})
	}
	return nil
}

func (m *Manager) Release(scopeID, tid string) error {
	m.mu.Lock()
	delete(m.procs, tid)
	m.mu.Unlock()
	return nil
}

type ToolInput struct {
	Action     string `json:"action,omitempty" jsonschema:"description=What to do. Defaults to 'run' (which is what you want for almost every call). Use 'list' to see active shells. Use 'status'/'output'/'wait'/'cancel'/'release' to manage a terminal.,enum=run,enum=list,enum=status,enum=output,enum=wait,enum=cancel,enum=release,default=run"`
	Command    string `json:"command,omitempty" jsonschema:"description=Shell command line to execute. Required when action='run'. Runs statefully inside the persistent shell process. Honours pipes, redirects, $(...) substitution, &&/||."`
	TerminalID string `json:"terminal_id,omitempty" jsonschema:"description=Identifier returned by a previous 'run'. Supply this to run a command inside the SAME stateful persistent shell session (reusing working directory and environment variables). Never required for a fresh shell."`
	Async      bool   `json:"async,omitempty" jsonschema:"description=When true the call returns immediately with a terminal_id and the command keeps running in the background. Use for long jobs (servers, watchers, builds >30s). Default false: synchronous, wait for the command to finish."`
	Timeout    int    `json:"timeout,omitempty" jsonschema:"description=Seconds to wait for a sync 'run' or a 'wait' to complete before giving up. Default 10 for run, 30 for wait. The command keeps running in the background after timeout — poll status/output to keep watching."`
	Offset     int    `json:"offset,omitempty" jsonschema:"description=Byte offset to start reading the terminal log from. Use 0 (default) for a fresh read; supply the previous call's 'size' to tail just the new output."`
	Limit      int    `json:"limit,omitempty" jsonschema:"description=Maximum bytes of terminal output to return in one call. Default 24KB. Larger reads are paginated via offset+limit; the structured response includes total 'size' so you can resume."`
}

// terminalToolDescription is the model-facing contract for the terminal tool.
const terminalToolDescription = `Run shell commands and manage stateful, persistent shell sessions.

When to use:
- Inspecting the environment (ls, git status, env, which, …).
- Building, testing, linting (go build, npm test, pytest, …).
- Spawning and managing persistent shell sessions.
- REUSING persistent shells: supply "terminal_id" with action="run" to execute commands inside the SAME persistent shell process. This preserves working directories, environment variables, aliases, and running subprocesses!

Default usage — synchronous run:
` + "```" + `
tools.terminal({ command: "go test ./..." })
` + "```" + `
Returns ` + "`{ terminal, output, size, offset, limit }`" + ` once the command exits.

Stateful Persistent Shell Reuse:
` + "```" + `
// Start fresh and export a variable
const t = await tools.terminal({ command: "export MY_VAR=hello" });
// Reuse that SAME shell (MY_VAR is preserved!)
const t2 = await tools.terminal({ command: "echo $MY_VAR", terminal_id: t.terminal_id });
` + "```" + `

Long-running jobs — async run + poll:
` + "```" + `
const t = await tools.terminal({ command: "npm run dev", async: true });
// list running sessions:
const active = await tools.terminal({ action: "list" });
` + "```" + `

Actions:
- ` + "`run`" + `      — start a fresh persistent shell OR execute inside an active one by supplying ` + "`terminal_id`" + `. Sync by default; pass ` + "`async: true`" + ` to return immediately.
- ` + "`list`" + `     — list all active persistent shell sessions for the current session.
- ` + "`status`" + `   — fetch ` + "`{ id, status, exit_code, started_at, ended_at }`" + ` for a known terminal.
- ` + "`output`" + `   — read more output from a known terminal. Honours ` + "`offset`" + ` / ` + "`limit`" + `.
- ` + "`wait`" + `     — block until the terminal exits or ` + "`timeout`" + ` fires.
- ` + "`cancel`" + `   — terminate a persistent shell process group immediately.
- ` + "`release`" + `  — stop tracking a terminal in the in-memory map.

Reliability tips:
- Always check ` + "`result.terminal.exit_code`" + ` and ` + "`result.output`" + ` before claiming success.`

func stripSentinel(out string) string {
	for {
		idx := strings.Index(out, "---PTY-DONE-UUID-")
		if idx == -1 {
			break
		}
		endIdx := idx
		for endIdx < len(out) && out[endIdx] != '\n' && out[endIdx] != '\r' {
			endIdx++
		}
		for endIdx < len(out) && (out[endIdx] == '\n' || out[endIdx] == '\r') {
			endIdx++
		}
		out = out[:idx] + out[endIdx:]
	}
	return strings.TrimRight(out, "\r\n ")
}

func injectActiveTerminalsHeader(payload map[string]any, manager *Manager, scopeID string, currentTID string) {
	manager.mu.Lock()
	var active []string
	for _, p := range manager.procs {
		if p.term.ScopeID == scopeID && p.term.Status == statusRunning && p.term.ID != currentTID {
			words := strings.Fields(p.term.Command)
			cmdName := "shell"
			if len(words) > 0 {
				cmdName = words[0]
				if len(words) > 1 {
					cmdName += " " + words[1]
				}
			}
			active = append(active, fmt.Sprintf("%s (%s)", p.term.ID, cmdName))
		}
	}
	manager.mu.Unlock()

	if len(active) > 0 {
		sort.Strings(active)
		payload["active_background_session_count"] = len(active)
		if len(active) > activeTerminalPreviewLimit {
			payload["active_background_sessions"] = strings.Join(active[:activeTerminalPreviewLimit], ", ")
			payload["active_background_sessions_truncated"] = true
			payload["active_background_sessions_omitted"] = len(active) - activeTerminalPreviewLimit
			return
		}
		payload["active_background_sessions"] = strings.Join(active, ", ")
	}
}

func NewTool(manager *Manager) llm.Tool {
	return tools.NewGenericTool("terminal", terminalToolDescription, func(ctx llm.ToolContext, input ToolInput) llm.ToolResult {
		action := strings.TrimSpace(input.Action)
		if action == "" {
			action = "run"
		}
		switch action {
		case "run":
			sessionID := ""
			if v, ok := ctx.Vars["session_id"].(string); ok {
				sessionID = v
			}

			timeout := input.Timeout
			if timeout <= 0 {
				timeout = 10
			}

			var term Terminal
			var err error

			if input.TerminalID != "" {
				term, err = manager.ExecuteCommand(ctx.Context, sessionID, input.TerminalID, input.Command, input.Async, time.Duration(timeout)*time.Second)
				if err != nil {
					return tools.ErrorResult(err)
				}
			} else {
				term, err = manager.Start(StartRequest{
					ScopeID: sessionID,
					Command: input.Command,
					Cwd:     ctx.WorkingDirectory,
					Async:   input.Async,
					Timeout: time.Duration(timeout) * time.Second,
				})
				if err != nil {
					return tools.ErrorResult(err)
				}
			}

			if input.Async {
				payload := map[string]any{
					"terminal_id": term.ID,
					"status":      term.Status,
					"terminal":    term,
				}
				injectActiveTerminalsHeader(payload, manager, sessionID, term.ID)
				return tools.JSONResult(input.Command, llm.ToolKindExecute, payload, "")
			}

			limit := input.Limit
			if limit <= 0 {
				limit = 24 * 1024
			}
			out, size, err := manager.Output(sessionID, term.ID, input.Offset, limit)
			if err != nil {
				return tools.ErrorResult(err)
			}
			out = stripSentinel(out)

			payload := map[string]any{
				"terminal_id": term.ID,
				"status":      term.Status,
				"output":      out,
				"size":        size,
				"offset":      input.Offset,
				"limit":       limit,
				"truncated":   size > input.Offset+len(out),
				"terminal":    term,
			}
			if term.ExitCode != nil {
				payload["exit_code"] = *term.ExitCode
			}
			injectActiveTerminalsHeader(payload, manager, sessionID, term.ID)
			return tools.JSONResult(input.Command, llm.ToolKindExecute, payload, "")

		case "list":
			sessionID := ""
			if v, ok := ctx.Vars["session_id"].(string); ok {
				sessionID = v
			}
			terms, err := manager.List(sessionID)
			if err != nil {
				return tools.ErrorResult(err)
			}
			return tools.JSONResult("Active terminal sessions", llm.ToolKindOther, terms, "")

		case "status":
			sessionID := ""
			if v, ok := ctx.Vars["session_id"].(string); ok {
				sessionID = v
			}
			term, err := manager.Status(sessionID, input.TerminalID)
			if err != nil {
				return tools.ErrorResult(err)
			}
			return tools.JSONResult("Terminal status", llm.ToolKindOther, term, "")

		case "output":
			sessionID := ""
			if v, ok := ctx.Vars["session_id"].(string); ok {
				sessionID = v
			}
			limit := input.Limit
			if limit <= 0 {
				limit = 24 * 1024
			}
			out, size, err := manager.Output(sessionID, input.TerminalID, input.Offset, limit)
			if err != nil {
				return tools.ErrorResult(err)
			}
			out = stripSentinel(out)
			payload := map[string]any{
				"terminal_id": input.TerminalID,
				"output":      out,
				"size":        size,
				"offset":      input.Offset,
				"limit":       limit,
				"truncated":   size > input.Offset+len(out),
			}
			if term, err := manager.Status(sessionID, input.TerminalID); err == nil {
				payload["status"] = term.Status
				if term.ExitCode != nil {
					payload["exit_code"] = *term.ExitCode
				}
			}
			injectActiveTerminalsHeader(payload, manager, sessionID, input.TerminalID)
			return tools.JSONResult("Terminal output", llm.ToolKindOther, payload, "")

		case "wait":
			sessionID := ""
			if v, ok := ctx.Vars["session_id"].(string); ok {
				sessionID = v
			}
			timeout := input.Timeout
			if timeout <= 0 {
				timeout = 30
			}
			term, err := manager.Wait(ctx.Context, sessionID, input.TerminalID, time.Duration(timeout)*time.Second)
			if err != nil {
				return tools.ErrorResult(err)
			}
			return tools.JSONResult("Terminal wait", llm.ToolKindOther, term, "")

		case "cancel":
			sessionID := ""
			if v, ok := ctx.Vars["session_id"].(string); ok {
				sessionID = v
			}
			_ = manager.Cancel(sessionID, input.TerminalID)
			return tools.JSONResult("Cancel terminal", llm.ToolKindOther, map[string]any{}, "")

		case "release":
			sessionID := ""
			if v, ok := ctx.Vars["session_id"].(string); ok {
				sessionID = v
			}
			_ = manager.Release(sessionID, input.TerminalID)
			return tools.JSONResult("Release terminal", llm.ToolKindOther, map[string]any{}, "")
		default:
			return tools.ErrorResult(fmt.Errorf("unsupported terminal action: %s", action))
		}
	}).WithPermissionExtractor(func(ctx llm.ToolContext, input ToolInput) ([]llm.PermissionGuard, error) {
		if strings.TrimSpace(input.Action) != "run" && input.Action != "" {
			return nil, nil
		}
		cmd := strings.TrimSpace(input.Command)
		if isHarmlessCommandLine(cmd) {
			return nil, nil
		}
		return terminalPermissionGuards(cmd, ctx.WorkingDirectory), nil
	})
}

// harmlessCommands lists root commands that — with their *safe* flag set —
// only read from the filesystem and the environment. A few entries (sed,
// find) accept destructive flags and need an extra per-flag check below.
//
// cp, mkdir, touch were intentionally removed: they all mutate the
// filesystem (overwriting an existing path, creating directories,
// touching timestamps) and should always go through the permission flow.
var harmlessCommands = map[string]bool{
	"ls": true, "cd": true, "pwd": true, "cat": true, "sed": true,
	"head": true, "tail": true, "grep": true, "find": true, "echo": true,
	"printenv": true, "which": true, "wc": true, "sort": true, "uniq": true,
	"cut": true, "tr": true, "diff": true, "date": true, "whoami": true,
	"du": true, "df": true, "stat": true, "file": true, "basename": true,
	"dirname": true, "realpath": true, "readlink": true, "env": true,
	"true": true, "false": true, "printf": true, "seq": true, "id": true,
	"uname": true, "hostname": true, "uptime": true, "pgrep": true,
	"ps": true, "top": true, "htop": true, "free": true, "vmstat": true,
	"iostat": true, "netstat": true, "ss": true, "lsof": true, "history": true,
	"alias": true, "type": true, "command": true, "builtin": true,
	"declare": true, "export": true, "local": true, "readonly": true,
	"set": true, "shopt": true, "test": true, "[": true,
}

var harmlessGitSubcommands = map[string]bool{
	"status": true, "log": true, "show": true, "diff": true, "branch": true,
	"rev-parse": true, "ls-files": true, "grep": true, "remote": true,
}

// bashControlKeywords are shell language constructs (not external commands)
// that show up as the leading token in pipeline segments after we split on
// `|`/`;`/newlines. They never invoke anything by themselves, so once we
// have validated every external command in the pipeline they should pass.
var bashControlKeywords = map[string]bool{
	"if": true, "then": true, "else": true, "elif": true, "fi": true,
	"case": true, "esac": true, "in": true,
	"for": true, "while": true, "until": true, "do": true, "done": true,
	"function": true, "select": true, "time": true, "!": true,
	"{": true, "}": true,
}

func isHarmlessCommandLine(cmd string) bool {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return true
	}
	// Legacy backtick command substitution and bash process substitution
	// (<(...) / >(...)) can both invoke arbitrary subshells. There's no
	// safe way to validate them from the outside, so they're hard rejects.
	if strings.ContainsRune(cmd, '`') {
		return false
	}
	if hasProcessSubstitution(cmd) {
		return false
	}
	// Validate $(...) substitutions recursively. Each inner pipeline must be
	// harmless on its own, and we replace the substitution with a placeholder
	// so subsequent scans don't trip on the embedded characters.
	stripped, ok := stripHarmlessSubstitutions(cmd)
	if !ok {
		return false
	}
	cmd = stripHarmlessRedirects(stripped)
	// After redirect stripping, any remaining `>` or `<` is either a write
	// to an arbitrary path or an exotic FD operation we don't model. Force
	// the permission flow.
	if strings.ContainsAny(cmd, "><") {
		return false
	}
	parts := splitShellCommands(cmd)
	if len(parts) == 0 {
		return false
	}
	for _, part := range parts {
		if !isHarmlessSimpleCommand(strings.TrimSpace(part)) {
			return false
		}
	}
	return true
}

// hasProcessSubstitution returns true when cmd contains a `<(` or `>(` token
// outside of single quotes. It does NOT respect double quotes — bash treats
// `<(` and `>(` as active inside double quotes only in some edge cases, and a
// conservative reject here is fine since real-world commands rarely embed the
// glyphs inside strings.
func hasProcessSubstitution(cmd string) bool {
	inSingle := false
	for i := 0; i < len(cmd)-1; i++ {
		c := cmd[i]
		if c == '\\' && i+1 < len(cmd) {
			i++
			continue
		}
		if c == '\'' {
			inSingle = !inSingle
			continue
		}
		if inSingle {
			continue
		}
		if (c == '<' || c == '>') && cmd[i+1] == '(' {
			return true
		}
	}
	return false
}

// stripHarmlessSubstitutions walks cmd, finds every top-level `$(...)`
// substitution, recursively validates the inner pipeline via
// isHarmlessCommandLine, and replaces validated substitutions with a token
// the outer scan can safely ignore. Returns ok=false on any unbalanced
// substitution or non-harmless inner command. Single-quoted spans are
// skipped because $(...) inside single quotes is literal.
func stripHarmlessSubstitutions(cmd string) (string, bool) {
	var out strings.Builder
	out.Grow(len(cmd))
	inSingle := false
	i := 0
	for i < len(cmd) {
		c := cmd[i]
		if c == '\\' && i+1 < len(cmd) {
			out.WriteByte(c)
			out.WriteByte(cmd[i+1])
			i += 2
			continue
		}
		if c == '\'' && !inSingle {
			inSingle = true
			out.WriteByte(c)
			i++
			continue
		}
		if c == '\'' && inSingle {
			inSingle = false
			out.WriteByte(c)
			i++
			continue
		}
		if inSingle {
			out.WriteByte(c)
			i++
			continue
		}
		if c == '$' && i+1 < len(cmd) && cmd[i+1] == '(' {
			end, ok := findMatchingParen(cmd, i+1)
			if !ok {
				return "", false
			}
			inner := cmd[i+2 : end]
			if !isHarmlessCommandLine(inner) {
				return "", false
			}
			out.WriteString("__cmdsub__")
			i = end + 1
			continue
		}
		out.WriteByte(c)
		i++
	}
	if inSingle {
		// Unbalanced quote — let the rest of the pipeline make the call.
		return out.String(), true
	}
	return out.String(), true
}

// findMatchingParen locates the index of the `)` that closes the `(` at
// position open, respecting nested parens and skipping over single-quoted
// spans. Returns ok=false on imbalance.
func findMatchingParen(s string, open int) (int, bool) {
	if open >= len(s) || s[open] != '(' {
		return 0, false
	}
	depth := 1
	inSingle := false
	for i := open + 1; i < len(s); i++ {
		c := s[i]
		if c == '\\' && i+1 < len(s) {
			i++
			continue
		}
		if c == '\'' {
			inSingle = !inSingle
			continue
		}
		if inSingle {
			continue
		}
		switch c {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i, true
			}
		}
	}
	return 0, false
}

func stripHarmlessRedirects(cmd string) string {
	fields := strings.Fields(cmd)
	out := make([]string, 0, len(fields))
	for i := 0; i < len(fields); i++ {
		field := fields[i]
		switch {
		case isDevNullRedirectToken(field):
			continue
		case isRedirectOperator(field) && i+1 < len(fields) && fields[i+1] == "/dev/null":
			i++
			continue
		// stdin-from-file is a read-only redirect (`wc -c < file`,
		// `<file`). It can't mutate anything: data flows FROM the file
		// into the command's stdin. Treat it the same as the explicit
		// /dev/null cases. Process substitution `<(…)` was already
		// rejected upstream, so seeing `<` here is always a file form.
		case strings.HasPrefix(field, "<") && len(field) > 1 && field != "<<" && !strings.HasPrefix(field, "<("):
			continue
		case field == "<" && i+1 < len(fields):
			i++
			continue
		default:
			out = append(out, field)
		}
	}
	return strings.Join(out, " ")
}

func isRedirectOperator(field string) bool {
	switch field {
	case ">", "1>", "2>", "&>", ">>", "1>>", "2>>", "<", "0<":
		return true
	default:
		return false
	}
}

func isDevNullRedirectToken(field string) bool {
	if field == "2>&1" || field == "1>&2" {
		return true
	}
	for _, prefix := range []string{">", "1>", "2>", "&>", ">>", "1>>", "2>>", "<", "0<"} {
		if field == prefix+"/dev/null" {
			return true
		}
	}
	return false
}

// splitShellCommands splits a command line on the shell control operators
// `|`, `;`, `&`, `(`, `)`, and unquoted newlines — but only when they appear
// outside of single or double quotes (and not directly after a backslash).
// Without quote awareness, an argument like `sed 's|^./||'` is shredded into
// bogus tokens (`sed 's`, `^./`, `'`); without newline awareness, multi-line
// shell scripts pack the entire `do … done` body into one opaque segment
// that defeats the harmless-command check.
func splitShellCommands(cmd string) []string {
	var parts []string
	start := 0
	inSingle := false
	inDouble := false
	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if !inSingle && !inDouble && c == '\\' && i+1 < len(cmd) {
			i++
			continue
		}
		if !inDouble && c == '\'' {
			inSingle = !inSingle
			continue
		}
		if !inSingle && c == '"' {
			inDouble = !inDouble
			continue
		}
		if inSingle || inDouble {
			continue
		}
		switch c {
		case '|', ';', '&', '(', ')', '\n':
			if start < i {
				parts = append(parts, cmd[start:i])
			}
			// Collapse `&&` and `||` so we don't emit an empty segment.
			if (c == '&' || c == '|') && i+1 < len(cmd) && cmd[i+1] == c {
				i++
			}
			start = i + 1
		}
	}
	if start < len(cmd) {
		parts = append(parts, cmd[start:])
	}
	return parts
}

func isHarmlessSimpleCommand(cmd string) bool {
	fields := strings.Fields(cmd)
	// Strip leading variable assignments (FOO=bar BAZ=qux real-cmd …). A
	// segment that is *only* assignments (e.g. `size=__cmdsub__`) is itself
	// harmless — it doesn't invoke a command, just rebinds a shell var.
	for len(fields) > 0 && isVariableAssignment(fields[0]) {
		fields = fields[1:]
	}
	if len(fields) == 0 {
		return true
	}
	root := fields[0]
	// Bash control-flow keywords (while, do, done, if, then, …) are language
	// constructs, not external commands. They show up as the leading token
	// of pipeline segments after we split on `;`/`|`/newlines.
	if bashControlKeywords[root] {
		return true
	}
	if root == "git" && len(fields) > 1 {
		return harmlessGitSubcommands[fields[1]]
	}
	if !harmlessCommands[root] {
		return false
	}
	// A handful of otherwise-read-only commands accept flags that turn them
	// destructive. Force the permission flow when those flags appear so we
	// don't quietly let through an in-place file mutation.
	switch root {
	case "sed":
		for _, f := range fields[1:] {
			if f == "-i" || f == "--in-place" || strings.HasPrefix(f, "-i") || strings.HasPrefix(f, "--in-place=") {
				return false
			}
		}
	case "find":
		for _, f := range fields[1:] {
			switch f {
			case "-delete", "-exec", "-execdir", "-ok", "-okdir":
				return false
			}
		}
	}
	return true
}

// isVariableAssignment recognises tokens of the form NAME=value, where NAME
// is a valid shell identifier (letter/underscore start, then letters, digits,
// or underscores). The value half is intentionally not validated — bash
// allows just about anything there, including the empty string and quoted
// substrings.
func isVariableAssignment(token string) bool {
	eq := strings.IndexByte(token, '=')
	if eq <= 0 {
		return false
	}
	for i := 0; i < eq; i++ {
		c := token[i]
		switch {
		case c == '_':
		case c >= 'A' && c <= 'Z':
		case c >= 'a' && c <= 'z':
		case c >= '0' && c <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

func terminalPermissionGuards(cmd string, cwd string) []llm.PermissionGuard {
	cmd = stripHarmlessRedirects(strings.TrimSpace(cmd))
	parts := splitShellCommands(cmd)
	if len(parts) == 0 {
		parts = []string{cmd}
	}
	seen := map[string]bool{}
	var guards []llm.PermissionGuard
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" || isHarmlessCommandLine(part) {
			continue
		}
		key := part + "\x00" + cwd
		if seen[key] {
			continue
		}
		seen[key] = true
		guards = append(guards, newTerminalPermissionGuard(part, cwd))
	}
	if len(guards) == 0 {
		guards = append(guards, newTerminalPermissionGuard(cmd, cwd))
	}
	return guards
}

func isCompoundCommand(cmd string) bool {
	for _, r := range cmd {
		switch r {
		case '&', '|', ';', '(', ')', '`', '$':
			return true
		}
	}
	return false
}

func rootCommand(cmd string) string {
	fields := strings.Fields(cmd)
	if len(fields) == 0 {
		return ""
	}
	return fields[0]
}

func newTerminalPermissionGuard(command string, cwd string) llm.PermissionGuard {
	specifiers := []string{"command:" + command}
	if strings.TrimSpace(cwd) != "" {
		specifiers = append(specifiers, "working_dir:"+filepath.Clean(cwd))
	}
	return llm.PermissionGuard{
		Key:        "terminal.run",
		Specifiers: specifiers,
		MatchMode:  llm.PermissionMatchModePrefix,
		Options: []llm.PermissionOption{
			llm.AllowOnceOption(),
			llm.AllowAlwaysOption("allow_always_command_session", "Always allow this command (session)", llm.PermissionScopeSession, llm.PermissionTargetCommand, llm.PermissionGrant{Field: "command", MatchMode: llm.PermissionMatchModePrefix}),
			llm.AllowAlwaysOption("allow_always_command_global", "Always allow this command (global)", llm.PermissionScopeGlobal, llm.PermissionTargetCommand, llm.PermissionGrant{Field: "command", MatchMode: llm.PermissionMatchModePrefix}),
			llm.AllowAlwaysOption("allow_always_root_command_session", "Always allow this command name (session)", llm.PermissionScopeSession, llm.PermissionTargetCommand, llm.PermissionGrant{Field: "command", MatchMode: llm.PermissionMatchModePrefix, Transform: "words", Segments: 1}),
			llm.AllowAlwaysOption("allow_always_subcommand_session", "Always allow this subcommand (session)", llm.PermissionScopeSession, llm.PermissionTargetCommand, llm.PermissionGrant{Field: "command", MatchMode: llm.PermissionMatchModePrefix, Transform: "words", Segments: 2}),
			llm.AllowAlwaysOption("allow_always_directory_session", "Always allow this directory (session)", llm.PermissionScopeSession, llm.PermissionTargetFolder, llm.PermissionGrant{Field: "working_dir", MatchMode: llm.PermissionMatchModePath}),
			llm.AllowAlwaysOption("allow_always_tool_session", "Always allow terminal (session)", llm.PermissionScopeSession, llm.PermissionTargetTool),
			llm.AllowAlwaysOption("allow_always_tool_global", "Always allow terminal (global)", llm.PermissionScopeGlobal, llm.PermissionTargetTool),
			llm.RejectOnceOption(),
			llm.RejectAlwaysOption("reject_always_command_session", "Always reject this command (session)", llm.PermissionScopeSession, llm.PermissionTargetCommand, llm.PermissionGrant{Field: "command", MatchMode: llm.PermissionMatchModePrefix}),
			llm.RejectAlwaysOption("reject_always_command_global", "Always reject this command (global)", llm.PermissionScopeGlobal, llm.PermissionTargetCommand, llm.PermissionGrant{Field: "command", MatchMode: llm.PermissionMatchModePrefix}),
			llm.RejectAlwaysOption("reject_always_tool_session", "Always reject terminal (session)", llm.PermissionScopeSession, llm.PermissionTargetTool),
			llm.RejectAlwaysOption("reject_always_tool_global", "Always reject terminal (global)", llm.PermissionScopeGlobal, llm.PermissionTargetTool),
		},
	}
}
