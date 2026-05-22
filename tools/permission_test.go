package tools

import (
	"testing"

	llm "github.com/proiceremo/ai-sdk"
)

func TestExtractPermissionNormalizesPathFields(t *testing.T) {
	guard, err := ExtractPermission(llm.ToolContext{WorkingDirectory: "/repo"}, map[string]any{
		"path": "./src/main.go",
	}, PermissionExtractor{
		Key: "fs.edit",
		Fields: []PermissionField{{
			Name:      "path",
			Transform: "path",
			Base:      "cwd",
		}},
		MatchMode: llm.PermissionMatchModePath,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(guard.Specifiers) != 1 || guard.Specifiers[0] != "path:/repo/src/main.go" {
		t.Fatalf("unexpected specifiers: %#v", guard.Specifiers)
	}
}

func TestTransformPermissionValueWords(t *testing.T) {
	got := TransformPermissionValue("", "git commit -m test", "words", "", 2)
	if got != "git commit" {
		t.Fatalf("unexpected words transform: %q", got)
	}
}
