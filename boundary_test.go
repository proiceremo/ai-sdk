package llm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAISDKDoesNotExposeAgentRuntimeSurface(t *testing.T) {
	forbidden := []string{
		"func RunAgent",
		"type AgentDefinition",
		"type AgentRunConfig",
		"type AgentRunState",
		"type RuntimeConfig",
		"func GenerateSessionID",
		"func GenerateRunID",
		"func SessionDir",
		"func RunDir",
		"SessionID        string",
		"RunID            string",
		"ProHome          string",
		"ParentAgent      string",
	}
	err := filepath.WalkDir(".", func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") {
			return err
		}
		if strings.HasSuffix(path, "_test.go") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(data)
		for _, needle := range forbidden {
			if strings.Contains(text, needle) {
				t.Fatalf("%s still contains forbidden runtime surface %q", path, needle)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scan ai-sdk boundary: %v", err)
	}
}
