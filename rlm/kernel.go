package rlm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

const (
	defaultChunkSize    = 1200
	defaultTopK         = 8
	defaultMaxFiles     = 2000
	defaultMaxFileBytes = int64(512 * 1024)
)

var defaultExcludedDirs = map[string]bool{
	".git": true, ".hg": true, ".svn": true, ".proagent": true,
	"node_modules": true, "vendor": true, "dist": true, "build": true,
	".next": true, ".svelte-kit": true, "coverage": true, ".turbo": true,
}

// Kernel provides document and workspace indexing with bounded retrieval.
type Kernel struct {
	mu sync.RWMutex

	docs       map[string]Document
	workspaces map[string]workspaceState
}

type workspaceState struct {
	Root      string
	Indexed   bool
	Signature string
	FileCount int
}

// NewKernel creates a new RLM kernel.
func NewKernel() *Kernel {
	return &Kernel{
		docs:       make(map[string]Document),
		workspaces: make(map[string]workspaceState),
	}
}

// Index adds documents to the kernel.
func (k *Kernel) Index(docs []Document, replace bool) (IndexResult, error) {
	if k == nil {
		return IndexResult{}, fmt.Errorf("rlm kernel is not configured")
	}
	if len(docs) == 0 {
		return IndexResult{}, fmt.Errorf("documents are required")
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if replace {
		k.docs = make(map[string]Document)
		k.workspaces = make(map[string]workspaceState)
	}
	indexed := make([]string, 0, len(docs))
	for _, doc := range docs {
		if strings.TrimSpace(doc.Text) == "" {
			continue
		}
		if strings.TrimSpace(doc.ID) == "" {
			doc.ID = stableDocID(doc)
		}
		k.docs[doc.ID] = doc
		indexed = append(indexed, doc.ID)
	}
	sort.Strings(indexed)
	return IndexResult{IndexedIDs: indexed, IndexedDocuments: len(k.docs)}, nil
}

// IndexWorkspace indexes text-like files under root. It skips common generated
// directories and keeps only bounded file content so retrieval remains cheap.
func (k *Kernel) IndexWorkspace(ctx context.Context, root string, opts WorkspaceOptions) (IndexResult, error) {
	if k == nil {
		return IndexResult{}, fmt.Errorf("rlm kernel is not configured")
	}
	if strings.TrimSpace(root) == "" {
		root = opts.Root
	}
	if strings.TrimSpace(root) == "" {
		return IndexResult{}, fmt.Errorf("workspace root is required")
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return IndexResult{}, err
	}
	if opts.Refresh {
		k.clearWorkspace(absRoot)
	}
	if state, ok := k.workspaceState(absRoot); ok && state.Indexed && !opts.Refresh {
		return IndexResult{IndexedDocuments: len(k.SnapshotDocs()), WorkspaceRoot: absRoot, WorkspaceFiles: state.FileCount}, nil
	}

	maxFiles := opts.MaxFiles
	if maxFiles <= 0 {
		maxFiles = defaultMaxFiles
	}
	maxFileBytes := opts.MaxFileBytes
	if maxFileBytes <= 0 {
		maxFileBytes = defaultMaxFileBytes
	}

	docs := make([]Document, 0)
	skipped := 0
	err = filepath.WalkDir(absRoot, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			skipped++
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if path != absRoot && shouldSkipDir(name, path, absRoot, opts.ExcludeGlobs) {
				return filepath.SkipDir
			}
			return nil
		}
		if len(docs) >= maxFiles {
			skipped++
			return nil
		}
		if !shouldIndexFile(path, absRoot, opts.IncludeGlobs, opts.ExcludeGlobs) {
			skipped++
			return nil
		}
		info, err := d.Info()
		if err != nil || info.Size() == 0 || info.Size() > maxFileBytes {
			skipped++
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil || !looksText(data) {
			skipped++
			return nil
		}
		rel, _ := filepath.Rel(absRoot, path)
		text := strings.TrimSpace(string(data))
		if text == "" {
			skipped++
			return nil
		}
		id := "workspace:" + filepath.ToSlash(rel)
		docs = append(docs, Document{
			ID:    id,
			Title: filepath.ToSlash(rel),
			URI:   path,
			Text:  text,
			Metadata: map[string]any{
				"kind": "workspace_file",
				"root": absRoot,
				"path": path,
			},
		})
		return nil
	})
	if err != nil {
		return IndexResult{}, err
	}
	result, err := k.Index(docs, false)
	if err != nil && len(docs) > 0 {
		return IndexResult{}, err
	}
	k.mu.Lock()
	k.workspaces[absRoot] = workspaceState{Root: absRoot, Indexed: true, Signature: workspaceSignature(docs), FileCount: len(docs)}
	k.mu.Unlock()
	result.WorkspaceRoot = absRoot
	result.WorkspaceFiles = len(docs)
	result.SkippedFiles = skipped
	return result, nil
}

// Retrieve retrieves bounded chunks matching the query.
func (k *Kernel) Retrieve(query string, topK int) ([]QueryResult, error) {
	return k.RetrieveWithDocuments(RetrieveInput{Query: query, TopK: topK, IncludeIndexed: true})
}

type RetrieveInput struct {
	Query          string
	Documents      []Document
	TopK           int
	ChunkSize      int
	IncludeIndexed bool
}

func (k *Kernel) RetrieveWithDocuments(input RetrieveInput) ([]QueryResult, error) {
	if strings.TrimSpace(input.Query) == "" {
		return nil, fmt.Errorf("query is required")
	}
	if input.TopK <= 0 {
		input.TopK = defaultTopK
	}
	if input.ChunkSize <= 0 {
		input.ChunkSize = defaultChunkSize
	}
	docs := append([]Document(nil), input.Documents...)
	if k != nil && (input.IncludeIndexed || len(docs) == 0) {
		k.mu.RLock()
		for _, doc := range k.docs {
			docs = append(docs, doc)
		}
		k.mu.RUnlock()
	}
	return RetrieveContext(input.Query, docs, input.TopK, input.ChunkSize)
}

func RetrieveContext(query string, docs []Document, topK int, chunkSize int) ([]QueryResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("query is required")
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("documents are required")
	}
	if topK <= 0 {
		topK = defaultTopK
	}
	if chunkSize <= 0 {
		chunkSize = defaultChunkSize
	}
	terms := queryTerms(query)
	results := make([]QueryResult, 0)
	for _, doc := range docs {
		for _, chunk := range chunkDocument(doc, chunkSize) {
			matches, score := scoreChunk(terms, doc.Title, doc.URI, chunk.Text)
			if score <= 0 {
				continue
			}
			results = append(results, QueryResult{
				DocumentID: doc.ID,
				Title:      doc.Title,
				URI:        doc.URI,
				ChunkIndex: chunk.Index,
				Start:      chunk.Start,
				End:        chunk.End,
				Score:      score,
				Text:       chunk.Text,
				Terms:      matches,
				Metadata:   MetadataMap(doc.Metadata),
			})
		}
	}
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].DocumentID < results[j].DocumentID
		}
		return results[i].Score > results[j].Score
	})
	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

// Analyze performs deterministic synthesis over retrieved chunks.
func (k *Kernel) Analyze(query string, docs []Document, topK int, chunkSize int) (Result, error) {
	results, err := k.RetrieveWithDocuments(RetrieveInput{Query: query, Documents: docs, TopK: topK, ChunkSize: chunkSize, IncludeIndexed: true})
	if err != nil {
		return Result{}, err
	}
	return analysisResult(query, "rlm", results, nil), nil
}

// LambdaAnalyze applies a typed bounded decomposition plan before retrieval.
func (k *Kernel) LambdaAnalyze(query string, docs []Document, topK int, chunkSize int) (Result, error) {
	if chunkSize <= 0 {
		chunkSize = defaultChunkSize
	}
	allDocs := append([]Document(nil), docs...)
	if k != nil && len(docs) == 0 {
		k.mu.RLock()
		for _, doc := range k.docs {
			allDocs = append(allDocs, doc)
		}
		k.mu.RUnlock()
	}
	total := 0
	for _, doc := range allDocs {
		total += utf8.RuneCountInString(doc.Text)
	}
	plan := LambdaPlanFor(query, total, chunkSize)
	results, err := k.RetrieveWithDocuments(RetrieveInput{Query: query, Documents: docs, TopK: topK, ChunkSize: plan.LeafChunkSize, IncludeIndexed: true})
	if err != nil {
		return Result{}, err
	}
	return analysisResult(query, "lambda_rlm", results, &plan), nil
}

func LambdaPlanFor(query string, n int, contextWindow int) LambdaPlan {
	if contextWindow <= 0 {
		contextWindow = defaultChunkSize
	}
	task, compose := classifyTask(query)
	if n <= contextWindow {
		return LambdaPlan{TaskType: task, ComposeOp: compose, Branching: 1, LeafChunkSize: max(n, 1), Depth: 0, CostEstimate: float64(max(n, 1))}
	}
	branching := int(math.Ceil(math.Sqrt(float64(n) / float64(contextWindow))))
	branching = min(max(branching, 2), 20)
	depth := int(math.Ceil(math.Log(float64(n)/float64(contextWindow)) / math.Log(float64(branching))))
	depth = max(depth, 1)
	leaf := min(contextWindow, max(1, int(math.Ceil(float64(n)/float64(branching)))))
	return LambdaPlan{
		TaskType:      task,
		ComposeOp:     compose,
		Branching:     branching,
		LeafChunkSize: leaf,
		Depth:         depth,
		CostEstimate:  float64(branching*depth*leaf + 500),
	}
}

func analysisResult(query, mode string, results []QueryResult, plan *LambdaPlan) Result {
	citations := make([]string, 0, len(results))
	var b strings.Builder
	for i, result := range results {
		citation := result.DocumentID
		if result.URI != "" {
			citation = result.URI
		}
		citations = append(citations, citation)
		if i >= 4 {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		fmt.Fprintf(&b, "[%s chunk %d]\n%s", citation, result.ChunkIndex, result.Text)
	}
	return Result{Mode: mode, Results: results, Analysis: b.String(), Citations: citations, Plan: plan}
}

type chunk struct {
	Index int
	Start int
	End   int
	Text  string
}

func chunkDocument(doc Document, size int) []chunk {
	text := strings.TrimSpace(doc.Text)
	if text == "" {
		return nil
	}
	runes := []rune(text)
	if len(runes) <= size {
		return []chunk{{Index: 0, Start: 0, End: len(runes), Text: text}}
	}
	overlap := min(size/8, 200)
	out := make([]chunk, 0, int(math.Ceil(float64(len(runes))/float64(size))))
	for start := 0; start < len(runes); {
		end := min(start+size, len(runes))
		out = append(out, chunk{Index: len(out), Start: start, End: end, Text: strings.TrimSpace(string(runes[start:end]))})
		if end == len(runes) {
			break
		}
		start = max(end-overlap, 0)
	}
	return out
}

func queryTerms(query string) []string {
	parts := strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '-' && r != '.'
	})
	seen := map[string]bool{}
	terms := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.Trim(part, "._-")
		if len(part) < 2 || seen[part] || stopWord(part) {
			continue
		}
		seen[part] = true
		terms = append(terms, part)
	}
	return terms
}

func scoreChunk(terms []string, title, uri, text string) ([]string, float64) {
	if len(terms) == 0 {
		return nil, 0
	}
	body := strings.ToLower(text)
	head := strings.ToLower(title + "\n" + uri)
	score := 0.0
	matches := make([]string, 0, len(terms))
	for _, term := range terms {
		bodyCount := strings.Count(body, term)
		headCount := strings.Count(head, term)
		if bodyCount+headCount == 0 {
			continue
		}
		matches = append(matches, term)
		score += 1 + math.Log1p(float64(bodyCount)) + 2*math.Log1p(float64(headCount))
	}
	return matches, score / math.Sqrt(float64(utf8.RuneCountInString(text))/300+1)
}

func classifyTask(query string) (string, string) {
	q := strings.ToLower(query)
	switch {
	case strings.Contains(q, "extract") || strings.Contains(q, "find all"):
		return "extraction", "concat_unique"
	case strings.Contains(q, "classif") || strings.Contains(q, "decide"):
		return "classification", "majority_vote"
	case strings.Contains(q, "summar"):
		return "summarization", "reduce_summaries"
	case strings.Contains(q, "?") || strings.Contains(q, "where") || strings.Contains(q, "how"):
		return "qa", "select_relevant"
	default:
		return "general", "merge_evidence"
	}
}

func shouldSkipDir(name, path, root string, exclude []string) bool {
	if defaultExcludedDirs[name] {
		return true
	}
	rel, _ := filepath.Rel(root, path)
	return matchAny(filepath.ToSlash(rel), exclude)
}

func shouldIndexFile(path, root string, include, exclude []string) bool {
	rel, _ := filepath.Rel(root, path)
	rel = filepath.ToSlash(rel)
	if matchAny(rel, exclude) {
		return false
	}
	if len(include) > 0 && !matchAny(rel, include) {
		return false
	}
	ext := strings.ToLower(filepath.Ext(path))
	if ext == "" {
		return true
	}
	switch ext {
	case ".go", ".ts", ".tsx", ".js", ".jsx", ".svelte", ".py", ".rs", ".java", ".c", ".h", ".cpp", ".hpp",
		".md", ".mdx", ".txt", ".json", ".yaml", ".yml", ".toml", ".sql", ".css", ".html", ".xml", ".sh":
		return true
	default:
		return false
	}
}

func matchAny(path string, patterns []string) bool {
	for _, pattern := range patterns {
		pattern = filepath.ToSlash(strings.TrimSpace(pattern))
		if pattern == "" {
			continue
		}
		if ok, _ := filepath.Match(pattern, path); ok {
			return true
		}
		if strings.HasPrefix(path, strings.TrimSuffix(pattern, "/")+"/") {
			return true
		}
	}
	return false
}

func looksText(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	sample := data
	if len(sample) > 4096 {
		sample = sample[:4096]
	}
	if !utf8.Valid(sample) {
		return false
	}
	nul := 0
	for _, b := range sample {
		if b == 0 {
			nul++
		}
	}
	return nul == 0
}

func MetadataMap(value any) map[string]any {
	if value == nil {
		return nil
	}
	if m, ok := value.(map[string]any); ok {
		return m
	}
	return map[string]any{"value": value}
}

func stableDocID(doc Document) string {
	key := doc.URI + "\x00" + doc.Title + "\x00" + doc.Text
	sum := sha256.Sum256([]byte(key))
	return "doc:" + hex.EncodeToString(sum[:8])
}

func workspaceSignature(docs []Document) string {
	h := sha256.New()
	for _, doc := range docs {
		h.Write([]byte(doc.ID))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (k *Kernel) workspaceState(root string) (workspaceState, bool) {
	k.mu.RLock()
	defer k.mu.RUnlock()
	state, ok := k.workspaces[root]
	return state, ok
}

func (k *Kernel) clearWorkspace(root string) {
	k.mu.Lock()
	defer k.mu.Unlock()
	for id, doc := range k.docs {
		if strings.HasPrefix(id, "workspace:") {
			if meta := MetadataMap(doc.Metadata); meta != nil && meta["root"] == root {
				delete(k.docs, id)
			}
		}
	}
	delete(k.workspaces, root)
}

func (k *Kernel) SnapshotDocs() []Document {
	k.mu.RLock()
	defer k.mu.RUnlock()
	out := make([]Document, 0, len(k.docs))
	for _, doc := range k.docs {
		out = append(out, doc)
	}
	return out
}

func stopWord(s string) bool {
	switch s {
	case "the", "and", "for", "with", "that", "this", "from", "what", "when", "where", "why", "how", "into", "about":
		return true
	default:
		return false
	}
}
