package llm

import (
	"context"
	"encoding/json"
	"testing"
)

type testRegistryTool struct {
	schema ToolSchema
}

func (t testRegistryTool) Schema() ToolSchema {
	return t.schema
}

func (t testRegistryTool) Execute(ctx ToolContext, input map[string]any) ToolResult {
	_ = ctx
	_ = input
	return ToolResult{}
}

func TestToolRegistryBuildsConfiguredTools(t *testing.T) {
	registry := NewToolRegistry()
	registry.MustRegister("echo", func(ctx context.Context, build ToolBuildContext, raw json.RawMessage) (Tool, error) {
		var cfg struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, err
		}
		return testRegistryTool{schema: ToolSchema{Name: cfg.Name}}, nil
	})

	tools, err := registry.BuildTools(context.Background(), ToolBuildContext{}, []ToolConfig{{
		ID:     "echo",
		Config: json.RawMessage(`{"name":"echo_value"}`),
	}})
	if err != nil {
		t.Fatalf("BuildTools returned error: %v", err)
	}
	if len(tools) != 1 || tools[0].Schema().Name != "echo_value" {
		t.Fatalf("unexpected tools: %#v", tools)
	}
}

func TestToolRegistryRejectsUnknownAndDuplicateTools(t *testing.T) {
	registry := NewToolRegistry()
	registry.MustRegister("known", func(context.Context, ToolBuildContext, json.RawMessage) (Tool, error) {
		return testRegistryTool{schema: ToolSchema{Name: "known"}}, nil
	})
	if _, err := registry.BuildTools(context.Background(), ToolBuildContext{}, []ToolConfig{{ID: "missing"}}); err == nil {
		t.Fatal("expected unknown tool error")
	}
	if _, err := registry.BuildTools(context.Background(), ToolBuildContext{}, []ToolConfig{{ID: "known"}, {ID: "known"}}); err == nil {
		t.Fatal("expected duplicate tool error")
	}
}
