package rlm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorkspaceIndexRetrieveFindsCurrentWorkingDirectoryFiles(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "README.md"), "Project notes mention spectral-index-token and planner behavior.")
	writeFile(t, filepath.Join(root, "node_modules", "ignored.md"), "spectral-index-token should not be indexed here")

	kernel := NewKernel()
	index, err := kernel.IndexWorkspace(context.Background(), root, WorkspaceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if index.WorkspaceFiles != 1 {
		t.Fatalf("expected one indexed workspace file, got %#v", index)
	}
	results, err := kernel.RetrieveWithDocuments(RetrieveInput{Query: "spectral-index-token", IncludeIndexed: true, TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Title != "README.md" {
		t.Fatalf("unexpected results: %#v", results)
	}
}

func TestRetrieveReturnsBoundedChunksNotWholeDocument(t *testing.T) {
	kernel := NewKernel()
	large := "BEGIN_SENTINEL " + strings.Repeat("alpha beta gamma ", 300) + " END_SENTINEL"
	if _, err := kernel.Index([]Document{{ID: "doc", Text: large}}, true); err != nil {
		t.Fatal(err)
	}
	results, err := kernel.RetrieveWithDocuments(RetrieveInput{Query: "alpha", IncludeIndexed: true, TopK: 2, ChunkSize: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected retrieved chunks")
	}
	combined := ""
	for _, result := range results {
		if len([]rune(result.Text)) > 240 {
			t.Fatalf("chunk exceeded bounded size: %d", len([]rune(result.Text)))
		}
		combined += result.Text
	}
	if strings.Contains(combined, "BEGIN_SENTINEL") && strings.Contains(combined, "END_SENTINEL") {
		t.Fatalf("retrieval exposed entire document, output length=%d", len(combined))
	}
}

func TestLambdaAnalyzeProducesTypedBoundedPlan(t *testing.T) {
	kernel := NewKernel()
	doc := Document{ID: "doc", Text: strings.Repeat("question evidence answer ", 500)}
	result, err := kernel.LambdaAnalyze("question?", []Document{doc}, 3, 300)
	if err != nil {
		t.Fatal(err)
	}
	if result.Mode != "lambda_rlm" || result.Plan == nil {
		t.Fatalf("expected lambda plan, got %#v", result)
	}
	if result.Plan.Branching < 1 || result.Plan.LeafChunkSize > 300 {
		t.Fatalf("unexpected plan bounds: %#v", result.Plan)
	}
	if len(result.Results) > 3 {
		t.Fatalf("expected top_k bound, got %d", len(result.Results))
	}
}

func writeFile(t *testing.T, path string, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}
