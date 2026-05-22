package codeexec

import (
	"context"
	"encoding/json"
	"testing"

	aitools "ai-sdk/tools"
	llm "ai-sdk"
)

// TestVarsPersistenceAcrossMultipleCalls verifies that vars persist across multiple
// js_execute calls within the same session
func TestVarsPersistenceAcrossMultipleCalls(t *testing.T) {
	executor := NewQuickJSExecutor()
	tool := aitools.NewGenericTool("echo", "Echo", func(ctx llm.ToolContext, input struct {
		Value string `json:"value"`
	}) llm.ToolResult {
		return llm.ToolResult{StructuredOutput: map[string]any{"value": input.Value}}
	})
	executor.SetTools([]llm.Tool{tool})

	sessionID := "test-session-vars-persistence"

	// First call: set initial vars
	req1 := JSRequest{
		Code:      "vars.counter = 1; vars.name = 'first'; return vars;",
		ScopeSessionID: sessionID,
	}
	resp1, err := executor.Execute(context.Background(), req1)
	if err != nil {
		t.Fatalf("first execution failed: %v", err)
	}

	var result1 map[string]any
	if err := json.Unmarshal([]byte(resp1.JSON), &result1); err != nil {
		t.Fatalf("failed to parse first result: %v", err)
	}
	if result1["counter"] != float64(1) || result1["name"] != "first" {
		t.Errorf("expected counter=1 and name=first, got %v", result1)
	}

	// Second call: vars should persist and we can increment
	req2 := JSRequest{
		Code:      "vars.counter = (vars.counter || 0) + 1; vars.name = vars.name + '-second'; return vars;",
		ScopeSessionID: sessionID,
	}
	resp2, err := executor.Execute(context.Background(), req2)
	if err != nil {
		t.Fatalf("second execution failed: %v", err)
	}

	var result2 map[string]any
	if err := json.Unmarshal([]byte(resp2.JSON), &result2); err != nil {
		t.Fatalf("failed to parse second result: %v", err)
	}
	if result2["counter"] != float64(2) {
		t.Errorf("expected counter=2 (incremented), got %v", result2["counter"])
	}
	if result2["name"] != "first-second" {
		t.Errorf("expected name='first-second', got %v", result2["name"])
	}

	// Third call: verify all vars still persist
	req3 := JSRequest{
		Code:      "return {finalCounter: vars.counter, finalName: vars.name};",
		ScopeSessionID: sessionID,
	}
	resp3, err := executor.Execute(context.Background(), req3)
	if err != nil {
		t.Fatalf("third execution failed: %v", err)
	}

	var result3 map[string]any
	if err := json.Unmarshal([]byte(resp3.JSON), &result3); err != nil {
		t.Fatalf("failed to parse third result: %v", err)
	}
	if result3["finalCounter"] != float64(2) || result3["finalName"] != "first-second" {
		t.Errorf("vars did not persist correctly: %v", result3)
	}
}

// TestVarsIsolationBetweenSessions verifies that vars are isolated between different sessions
func TestVarsIsolationBetweenSessions(t *testing.T) {
	executor := NewQuickJSExecutor()

	sessionA := "session-a"
	sessionB := "session-b"

	// Set vars in session A
	reqA := JSRequest{
		Code:      "vars.data = 'session-a-data'; return vars;",
		ScopeSessionID: sessionA,
	}
	_, err := executor.Execute(context.Background(), reqA)
	if err != nil {
		t.Fatalf("session A execution failed: %v", err)
	}

	// Set different vars in session B
	reqB := JSRequest{
		Code:      "vars.data = 'session-b-data'; return vars;",
		ScopeSessionID: sessionB,
	}
	_, err = executor.Execute(context.Background(), reqB)
	if err != nil {
		t.Fatalf("session B execution failed: %v", err)
	}

	// Verify session A still has its own data
	reqACheck := JSRequest{
		Code:      "return vars.data;",
		ScopeSessionID: sessionA,
	}
	respA, err := executor.Execute(context.Background(), reqACheck)
	if err != nil {
		t.Fatalf("session A check failed: %v", err)
	}

	var resultA string
	if err := json.Unmarshal([]byte(respA.JSON), &resultA); err != nil {
		resultA = respA.JSON
	}
	if resultA != "session-a-data" {
		t.Errorf("session A data was corrupted or shared: got %v, want session-a-data", resultA)
	}

	// Verify session B still has its own data
	reqBCheck := JSRequest{
		Code:      "return vars.data;",
		ScopeSessionID: sessionB,
	}
	respB, err := executor.Execute(context.Background(), reqBCheck)
	if err != nil {
		t.Fatalf("session B check failed: %v", err)
	}

	var resultB string
	if err := json.Unmarshal([]byte(respB.JSON), &resultB); err != nil {
		resultB = respB.JSON
	}
	if resultB != "session-b-data" {
		t.Errorf("session B data was corrupted or shared: got %v, want session-b-data", resultB)
	}
}
