package codeexec

import llm "github.com/proiceremo/ai-sdk"

func withToolRuntimeVars(ctx llm.ToolContext, sessionID, actorID, runID, proHome string) llm.ToolContext {
	if ctx.Vars == nil {
		ctx.Vars = map[string]any{}
	}
	setIfEmpty(ctx.Vars, "session_id", sessionID)
	setIfEmpty(ctx.Vars, "actor_id", actorID)
	setIfEmpty(ctx.Vars, "run_id", runID)
	setIfEmpty(ctx.Vars, "pro_home", proHome)
	return ctx
}

func setIfEmpty(vars map[string]any, key, value string) {
	if value == "" {
		return
	}
	if _, ok := vars[key]; ok {
		return
	}
	vars[key] = value
}
