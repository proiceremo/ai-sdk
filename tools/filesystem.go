// Package tools — pi-style filesystem tools.
//
// Tools exposed to the model: read, edit, write, grep, find, ls.
//
// Design notes:
//   - No snapshot IDs and no line-ref tokens. Edits identify the target
//     region by exact text match (`oldText`), which must be unique in the
//     current file. This eliminates the misplaced/corrupted-edit failure
//     mode of the previous snapshot+ref design.
//   - grep uses re2 via wasm (github.com/wasilibs/go-re2) — no cgo, no
//     external binary dependency, linear-time regex.
//   - find and grep both honor .gitignore (via go-dotignore) and a
//     curated list of noisy directories (node_modules, .git, …).
//   - Each file write/edit is serialized through a per-path mutex so
//     concurrent calls to the same file don't interleave.
package tools

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	llm "ai-sdk"

	dotignore "github.com/codeglyph/go-dotignore/v2"
	"github.com/gen2brain/go-fitz"
	regexp "github.com/wasilibs/go-re2"
)

// -----------------------------------------------------------------------------
// Shared options + limits
// -----------------------------------------------------------------------------

// FileToolsOptions is kept for API back-compat with callers that pass it.
// MaxBytes overrides the default byte-truncation cap for read.
type FileToolsOptions struct {
	MaxBytes int64 `json:"max_bytes,omitempty"`
}

const (
	defaultMaxLines         = 2000
	defaultMaxBytes   int64 = 50 * 1024
	grepMaxLineLength       = 500
	grepDefaultLimit        = 100
	findDefaultLimit        = 1000
	lsDefaultLimit          = 500
)

// -----------------------------------------------------------------------------
// Path resolution
// -----------------------------------------------------------------------------

func resolveToolPath(cwd, path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("path is required")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwd, path)
	}
	return filepath.Clean(path), nil
}

// -----------------------------------------------------------------------------
// Truncation
// -----------------------------------------------------------------------------

type truncationResult struct {
	Content              string `json:"-"`
	Truncated            bool   `json:"truncated"`
	TruncatedBy          string `json:"truncated_by,omitempty"` // "lines" or "bytes"
	TotalLines           int    `json:"total_lines"`
	TotalBytes           int    `json:"total_bytes"`
	OutputLines          int    `json:"output_lines"`
	OutputBytes          int    `json:"output_bytes"`
	FirstLineExceedsByte bool   `json:"first_line_exceeds_byte,omitempty"`
	MaxLines             int    `json:"max_lines"`
	MaxBytes             int    `json:"max_bytes"`
}

// truncateHead keeps the first N lines / bytes (whichever cap fires first).
// Never returns a partial line (except via firstLineExceedsByte).
func truncateHead(content string, maxLines int, maxBytes int) truncationResult {
	if maxLines <= 0 {
		maxLines = defaultMaxLines
	}
	if maxBytes <= 0 {
		maxBytes = int(defaultMaxBytes)
	}
	totalBytes := len(content)
	lines := strings.Split(content, "\n")
	totalLines := len(lines)

	if totalLines <= maxLines && totalBytes <= maxBytes {
		return truncationResult{
			Content: content, Truncated: false,
			TotalLines: totalLines, TotalBytes: totalBytes,
			OutputLines: totalLines, OutputBytes: totalBytes,
			MaxLines: maxLines, MaxBytes: maxBytes,
		}
	}

	if len(lines[0]) > maxBytes {
		return truncationResult{
			Content: "", Truncated: true, TruncatedBy: "bytes",
			TotalLines: totalLines, TotalBytes: totalBytes,
			FirstLineExceedsByte: true,
			MaxLines:             maxLines, MaxBytes: maxBytes,
		}
	}

	out := make([]string, 0, maxLines)
	used := 0
	by := "lines"
	for i, line := range lines {
		if i >= maxLines {
			break
		}
		extra := len(line)
		if i > 0 {
			extra++
		}
		if used+extra > maxBytes {
			by = "bytes"
			break
		}
		out = append(out, line)
		used += extra
	}
	if len(out) >= maxLines && used <= maxBytes {
		by = "lines"
	}
	joined := strings.Join(out, "\n")
	return truncationResult{
		Content: joined, Truncated: true, TruncatedBy: by,
		TotalLines: totalLines, TotalBytes: totalBytes,
		OutputLines: len(out), OutputBytes: len(joined),
		MaxLines: maxLines, MaxBytes: maxBytes,
	}
}

func formatSize(n int) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%dB", n)
	case n < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(n)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(n)/(1024*1024))
	}
}

func truncateLine(line string, maxChars int) (string, bool) {
	if utf8.RuneCountInString(line) <= maxChars {
		return line, false
	}
	runes := []rune(line)
	return string(runes[:maxChars]) + "... [truncated]", true
}

// -----------------------------------------------------------------------------
// File mutation queue — serialize writes/edits to the same absolute path.
// -----------------------------------------------------------------------------

var fileMutationLocks = struct {
	sync.Mutex
	m map[string]*sync.Mutex
}{m: map[string]*sync.Mutex{}}

func withFileMutation(absPath string, fn func() llm.ToolResult) llm.ToolResult {
	key, err := filepath.EvalSymlinks(absPath)
	if err != nil {
		key = absPath
	}
	fileMutationLocks.Lock()
	mu, ok := fileMutationLocks.m[key]
	if !ok {
		mu = &sync.Mutex{}
		fileMutationLocks.m[key] = mu
	}
	fileMutationLocks.Unlock()
	mu.Lock()
	defer mu.Unlock()
	return fn()
}

// -----------------------------------------------------------------------------
// Binary detection + mime sniffing (small helpers reused below)
// -----------------------------------------------------------------------------

func isBinary(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	sniff := data
	if len(sniff) > 8000 {
		sniff = sniff[:8000]
	}
	if bytes.IndexByte(sniff, 0) != -1 {
		return true
	}
	if !utf8.Valid(sniff) {
		return true
	}
	return false
}

func detectMime(path string, data []byte) string {
	sniff := data
	if len(sniff) > 512 {
		sniff = sniff[:512]
	}
	return http.DetectContentType(sniff)
}

// -----------------------------------------------------------------------------
// Ignore + glob matching
// -----------------------------------------------------------------------------

func newRepoMatcher(root string, noIgnore bool) *dotignore.RepositoryMatcher {
	if noIgnore {
		return nil
	}
	cfg := dotignore.DefaultRepositoryConfig()
	cfg.SkipFolders = noisyDirNames()
	m, _ := dotignore.NewRepositoryMatcherWithConfig(root, cfg)
	return m
}

func matcherIgnored(matcher *dotignore.RepositoryMatcher, rel string, isDir bool) bool {
	if matcher == nil {
		return false
	}
	ignored, err := matcher.Matches(rel)
	if err == nil && ignored {
		return true
	}
	if isDir {
		ignored, err = matcher.Matches(rel + "/")
		return err == nil && ignored
	}
	return false
}

func isNoisyDir(name string) bool {
	for _, n := range noisyDirNames() {
		if name == n {
			return true
		}
	}
	return false
}

func noisyDirNames() []string {
	return []string{
		// Dev tooling — never useful for an agent to walk.
		".git", ".hg", ".svn", ".idea", ".vscode", ".cache",
		"node_modules", "vendor", "target", "dist", "build", ".next",
		".turbo", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
		".pytest_cache", ".ruff_cache",
		// Unix system dirs — walking these from container root reads
		// the host kernel filesystem (/proc, /sys), all binaries
		// (/usr/lib, /usr/bin), kernel headers, /var/log, etc.
		// Gemini Flash on case-000027 demonstrated this by hallucinating
		// `tools.grep({pattern:"SO-01", path:"/"})` — without the skip
		// list the walk loaded the proagent-eval binary (50-100MB) +
		// every glibc shared object + /proc fake-file slurpage into
		// memory and OOM-killed the container at exit 137.
		"proc", "sys", "dev", "run", "boot",
		"usr", "lib", "lib64", "bin", "sbin",
		"var", "etc", "mnt", "media", "opt",
	}
}

// maxGrepFileSize is the per-file ceiling for `os.ReadFile` during a
// grep walk. Any file LARGER than this is silently skipped — the agent
// has no business grepping a 100MB binary, and the cost of pre-Stat'ing
// every file is a few microseconds versus tens of GB of heap pressure
// when the walk lands on a binary blob.
//
// 1 MiB strikes a balance: covers every source file, config, and
// medium-sized JSON dataset in the workspace, but stops dead before
// reading shared libraries or compiled binaries.
const maxGrepFileSize int64 = 1 << 20 // 1 MiB

// maxGrepWalkBytes caps the TOTAL bytes read across the entire grep
// walk. Even if every individual file is small, walking a deep tree
// of small files (e.g. /usr/share with thousands of locales) still
// accumulates. Bail out cleanly once we've read this much, with a
// truncated=true flag so the agent knows the result is partial.
const maxGrepWalkBytes int64 = 50 << 20 // 50 MiB

type compiledGlob struct {
	pattern string
	re      *regexp.Regexp
}

func compileGlobs(patterns []string) ([]compiledGlob, error) {
	out := make([]compiledGlob, 0, len(patterns))
	for _, pattern := range patterns {
		pattern = strings.TrimSpace(filepath.ToSlash(pattern))
		if pattern == "" {
			continue
		}
		re, err := regexp.Compile(globToRegex(pattern))
		if err != nil {
			return nil, fmt.Errorf("invalid glob %q: %w", pattern, err)
		}
		out = append(out, compiledGlob{pattern: pattern, re: re})
	}
	return out, nil
}

func globToRegex(pattern string) string {
	pattern = strings.TrimPrefix(filepath.ToSlash(pattern), "./")
	if strings.HasSuffix(pattern, "/") {
		pattern += "**"
	}
	if !strings.Contains(pattern, "/") {
		pattern = "**/" + pattern
	}
	var b strings.Builder
	b.WriteByte('^')
	for i := 0; i < len(pattern); {
		ch := pattern[i]
		switch ch {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				i += 2
				if i < len(pattern) && pattern[i] == '/' {
					b.WriteString("(?:.*/)?")
					i++
				} else {
					b.WriteString(".*")
				}
			} else {
				b.WriteString("[^/]*")
				i++
			}
		case '?':
			b.WriteString("[^/]")
			i++
		case '/':
			b.WriteByte('/')
			i++
		default:
			b.WriteString(regexp.QuoteMeta(string(ch)))
			i++
		}
	}
	b.WriteByte('$')
	return b.String()
}

func matchGlob(globs []compiledGlob, rel string, isDir bool) bool {
	if len(globs) == 0 {
		return false
	}
	rel = strings.TrimPrefix(filepath.ToSlash(rel), "./")
	candidates := []string{rel}
	if isDir {
		candidates = append(candidates, rel+"/")
	}
	for _, g := range globs {
		for _, c := range candidates {
			if g.re.MatchString(c) {
				return true
			}
		}
	}
	return false
}

// -----------------------------------------------------------------------------
// read / read_range
//
// The two tools share a single ReadResult shape so the model never has to
// reason about a variant return type. They are split so that a partial read
// is *intent-visible* on the wire and in the system prompt: the model has
// to consciously call read_range, and reviewers see read_range in the
// trace. read itself can never return a slice — for oversize text files it
// errors with a redirect, eliminating the "read partial → write back"
// footgun structurally.
//
// Modality-aware (both tools):
//   - Text files: read returns the full file (errors if over the cap);
//     read_range returns a line slice from offset for limit lines.
//   - Images / audio / video: read returns the matching modality content
//     block when the model supports it, otherwise a text note. read_range
//     does not apply to raw media — call read.
//   - PDF / EPUB / DOCX / XLSX / PPTX / MOBI / XPS / CBZ / FB2: read returns
//     up to the first 5 pages via go-fitz; read_range pages with offset/limit
//     (1-indexed page numbers).
// -----------------------------------------------------------------------------

type ReadInput struct {
	Path string `json:"path" jsonschema:"description=Path to the file to read (relative or absolute),minLength=1"`
	Mode string `json:"mode,omitempty" jsonschema:"description=Forces a read mode; default auto picks text/raw/document by mime,enum=auto,enum=text,enum=raw"`
}

// ReadRangeInput requests a partial slice of a file. For text files,
// offset/limit are 1-indexed line numbers. For documents, offset/limit
// are 1-indexed page numbers.
type ReadRangeInput struct {
	Path   string `json:"path" jsonschema:"description=Path to the file to read (relative or absolute),minLength=1"`
	Offset int    `json:"offset" jsonschema:"description=1-indexed start line (text) or start page (documents). Required.,minimum=1"`
	Limit  int    `json:"limit,omitempty" jsonschema:"description=Maximum lines (text) or pages (documents) to return. Omit to read from offset to the next cap."`
	Mode   string `json:"mode,omitempty" jsonschema:"description=Forces a read mode; default auto picks text/raw/document by mime,enum=auto,enum=text"`
}

type ReadArtifact struct {
	Path     string `json:"path,omitempty"`
	MimeType string `json:"mime_type"`
	Bytes    int    `json:"bytes"`
	SHA256   string `json:"sha256"`
	Kind     string `json:"kind"`
}

type ReadDocumentPage struct {
	Number   int           `json:"number"`
	Text     string        `json:"text,omitempty"`
	Image    *ReadArtifact `json:"image,omitempty"`
	TextErr  string        `json:"text_error,omitempty"`
	ImageErr string        `json:"image_error,omitempty"`
}

type ReadResult struct {
	Path        string             `json:"path"`
	Mode        string             `json:"mode"` // "text" | "raw" | "document"
	MimeType    string             `json:"mime_type,omitempty"`
	BytesTotal  int                `json:"bytes_total"`
	SHA256      string             `json:"sha256,omitempty"`
	Content     string             `json:"content"`
	StartLine   int                `json:"start_line,omitempty"`
	EndLine     int                `json:"end_line,omitempty"`
	TotalLines  int                `json:"total_lines,omitempty"`
	NextOffset  int                `json:"next_offset,omitempty"`
	Truncation  *truncationResult  `json:"truncation,omitempty"`
	Notice      string             `json:"notice,omitempty"`
	RawModality string             `json:"raw_modality,omitempty"`
	Artifact    *ReadArtifact      `json:"artifact,omitempty"`
	Pages       []ReadDocumentPage `json:"pages,omitempty"`
	PageCount   int                `json:"page_count,omitempty"`
}

const readDescription = `Read the full contents of a file. Text files return the complete file (errors if it exceeds 2000 lines or 50KB — use read_range or grep instead). Images, audio, and video are returned as the matching content block when the current model supports that modality. PDFs, EPUBs, and Office docs (DOCX/XLSX/PPTX/...) return the first 5 pages via go-fitz (use read_range for more pages). Output of read is safe to round-trip through write/edit because it is always complete.`

const readRangeDescription = `Read a slice of a file. For text files, returns lines [offset, offset+limit). For PDFs/EPUBs/DOCX/..., returns pages [offset, offset+limit) via go-fitz. Use this for inspecting parts of large files. The returned content is intentionally partial — do NOT pass it into write, and do NOT use it as oldText for edit (oldText must be exact unique text from the current file; a slice usually is unique already, but make sure you're not silently losing the surrounding file). To modify a file, use edit with a unique oldText/newText pair, or call read for the whole file before write.`

const maxDocumentPagesPerCall = 5

func NewReadTool(opts FileToolsOptions) llm.Tool {
	maxBytes := int(defaultMaxBytes)
	if opts.MaxBytes > 0 {
		maxBytes = int(opts.MaxBytes)
	}
	return NewGenericTool("read", readDescription, func(ctx llm.ToolContext, input ReadInput) llm.ToolResult {
		path, data, mimeType, sha, mode, err := openForRead(ctx, input.Path, input.Mode)
		if err != nil {
			return ErrorResult(err)
		}

		// Documents go through go-fitz unless caller forces text mode. read
		// returns the first maxDocumentPagesPerCall pages.
		if mode != "text" && isFitzDocument(path, mimeType) {
			return readDocumentViaFitz(ctx, path, mimeType, int64(len(data)), sha, 1, maxDocumentPagesPerCall, false)
		}

		// Raw media: image/audio/video.
		if mode != "text" {
			if res, ok := readRawModality(ctx, path, input.Path, mimeType, sha, data); ok {
				return res
			}
			if isBinary(data) {
				return ErrorResult(fmt.Errorf("refusing to read non-text file %s (%s); current model does not support that modality", input.Path, mimeType))
			}
		}

		// Text path. read NEVER returns a slice — over the cap we error
		// and redirect the model at read_range/grep.
		if isBinary(data) {
			return ErrorResult(fmt.Errorf("refusing to read binary file %s (%s); use a shell tool to inspect bytes", input.Path, mimeType))
		}
		text := string(data)
		lines := strings.Split(text, "\n")
		totalLines := len(lines)
		if totalLines > defaultMaxLines {
			return ErrorResult(fmt.Errorf("read %s: file has %d lines (cap %d); use read_range with offset/limit or grep to inspect a slice", input.Path, totalLines, defaultMaxLines))
		}
		if len(data) > maxBytes {
			return ErrorResult(fmt.Errorf("read %s: file is %s (cap %s); use read_range with offset/limit or grep to inspect a slice", input.Path, formatSize(len(data)), formatSize(maxBytes)))
		}

		result := ReadResult{
			Path: path, Mode: "text",
			MimeType: mimeType, BytesTotal: len(data), SHA256: sha,
			Content:   text,
			StartLine: 1, EndLine: totalLines,
			TotalLines: totalLines,
		}
		block := llm.NewTextContentBlock(text)
		return llm.ToolResult{
			Output:           []llm.ContentBlock{block},
			StructuredOutput: result,
			Metadata: llm.ToolMetadata{
				Title:     "Read " + input.Path,
				Kind:      llm.ToolKindRead,
				Locations: []llm.ToolCallLocation{{Path: path}},
				Content:   []llm.ToolCallContent{llm.NewToolCallContentBlock(block)},
			},
		}
	}).WithOutputSchema(ReflectValue(ReadResult{}))
}

func NewReadRangeTool(opts FileToolsOptions) llm.Tool {
	maxBytes := int(defaultMaxBytes)
	if opts.MaxBytes > 0 {
		maxBytes = int(opts.MaxBytes)
	}
	return NewGenericTool("read_range", readRangeDescription, func(ctx llm.ToolContext, input ReadRangeInput) llm.ToolResult {
		if input.Offset < 1 {
			return ErrorResult(fmt.Errorf("read_range requires offset >= 1 (use read for the full file)"))
		}
		path, data, mimeType, sha, mode, err := openForRead(ctx, input.Path, input.Mode)
		if err != nil {
			return ErrorResult(err)
		}

		// Documents → page slice via fitz.
		if mode != "text" && isFitzDocument(path, mimeType) {
			limit := input.Limit
			if limit <= 0 {
				limit = maxDocumentPagesPerCall
			}
			return readDocumentViaFitz(ctx, path, mimeType, int64(len(data)), sha, input.Offset, limit, true)
		}

		// Raw media: read_range doesn't apply.
		if mode != "text" {
			if strings.HasPrefix(mimeType, "image/") || strings.HasPrefix(mimeType, "audio/") || strings.HasPrefix(mimeType, "video/") {
				return ErrorResult(fmt.Errorf("read_range does not apply to %s; use `read` for raw media", mimeType))
			}
		}

		if isBinary(data) {
			return ErrorResult(fmt.Errorf("refusing to read binary file %s (%s); use a shell tool to inspect bytes", input.Path, mimeType))
		}

		text := string(data)
		lines := strings.Split(text, "\n")
		totalLines := len(lines)
		startLine := input.Offset
		if startLine > totalLines {
			return ErrorResult(fmt.Errorf("offset %d is past end of file (%d lines)", startLine, totalLines))
		}
		startIdx := startLine - 1
		var selected string
		var userEnd int
		if input.Limit > 0 {
			endIdx := startIdx + input.Limit
			if endIdx > len(lines) {
				endIdx = len(lines)
			}
			selected = strings.Join(lines[startIdx:endIdx], "\n")
			userEnd = endIdx
		} else {
			selected = strings.Join(lines[startIdx:], "\n")
		}
		trunc := truncateHead(selected, defaultMaxLines, maxBytes)

		var notice string
		nextOffset := 0
		endLine := startLine + trunc.OutputLines - 1
		switch {
		case trunc.FirstLineExceedsByte:
			notice = fmt.Sprintf("Line %d is %s, exceeds %s limit. Use grep to find specific content.",
				startLine, formatSize(len(lines[startIdx])), formatSize(maxBytes))
		case trunc.Truncated:
			nextOffset = endLine + 1
			if trunc.TruncatedBy == "lines" {
				notice = fmt.Sprintf("Showing lines %d-%d of %d. Use offset=%d to continue.",
					startLine, endLine, totalLines, nextOffset)
			} else {
				notice = fmt.Sprintf("Showing lines %d-%d of %d (%s limit). Use offset=%d to continue.",
					startLine, endLine, totalLines, formatSize(maxBytes), nextOffset)
			}
		case input.Limit > 0 && userEnd < totalLines:
			nextOffset = userEnd + 1
			notice = fmt.Sprintf("%d more lines in file. Use offset=%d to continue.", totalLines-userEnd, nextOffset)
		}

		result := ReadResult{
			Path: path, Mode: "text",
			MimeType: mimeType, BytesTotal: len(data), SHA256: sha,
			Content:   trunc.Content,
			StartLine: startLine, EndLine: endLine,
			TotalLines: totalLines, NextOffset: nextOffset,
			Notice: notice,
		}
		if trunc.Truncated {
			t := trunc
			result.Truncation = &t
		}

		bodyText := trunc.Content
		if notice != "" {
			bodyText = trunc.Content + "\n\n[" + notice + "]"
		}
		block := llm.NewTextContentBlock(bodyText)
		return llm.ToolResult{
			Output:           []llm.ContentBlock{block},
			StructuredOutput: result,
			Metadata: llm.ToolMetadata{
				Title:     "Read " + input.Path,
				Kind:      llm.ToolKindRead,
				Locations: []llm.ToolCallLocation{{Path: path}},
				Content:   []llm.ToolCallContent{llm.NewToolCallContentBlock(block)},
			},
		}
	}).WithOutputSchema(ReflectValue(ReadResult{}))
}

// openForRead is the shared path/stat/read/mime/mode prelude. It returns
// the resolved absolute path, the file bytes, the detected mime type, the
// content sha256, and the effective mode ("auto" | "text" | "raw").
func openForRead(ctx llm.ToolContext, inputPath, modeRaw string) (path string, data []byte, mimeType, sha, mode string, err error) {
	path, err = resolveToolPath(ctx.WorkingDirectory, inputPath)
	if err != nil {
		return "", nil, "", "", "", err
	}
	stat, statErr := os.Stat(path)
	if statErr != nil {
		return "", nil, "", "", "", fmt.Errorf("read %s: %w", inputPath, statErr)
	}
	if stat.IsDir() {
		return "", nil, "", "", "", fmt.Errorf("read %s: is a directory; use `ls` instead", inputPath)
	}
	data, err = os.ReadFile(path)
	if err != nil {
		return "", nil, "", "", "", fmt.Errorf("read %s: %w", inputPath, err)
	}
	mimeType = detectMime(path, data)
	sum := sha256.Sum256(data)
	sha = hex.EncodeToString(sum[:])
	mode = strings.ToLower(strings.TrimSpace(modeRaw))
	if mode == "" {
		mode = "auto"
	}
	return path, data, mimeType, sha, mode, nil
}

// readRawModality emits an image/audio/video content block when the current
// model supports the matching modality. Returns (result, true) on a match;
// otherwise (zero, false) so the caller can fall back.
func readRawModality(ctx llm.ToolContext, path, inputPath, mimeType, sha string, data []byte) (llm.ToolResult, bool) {
	block, modality, ok := rawContentBlockForModel(ctx.ModelConfig, mimeType, data)
	if !ok {
		return llm.ToolResult{}, false
	}
	artifact := writeReadArtifact(ctx, "read-"+modality, mimeType, modality, data)
	result := ReadResult{
		Path: path, Mode: "raw", MimeType: mimeType,
		BytesTotal: len(data), SHA256: sha,
		RawModality: modality, Artifact: artifact,
	}
	note := fmt.Sprintf("Read %s file (%s, %s).", modality, mimeType, formatSize(len(data)))
	if artifact != nil && artifact.Path != "" {
		note += " Durable artifact: " + artifact.Path
	}
	result.Content = note
	textBlock := llm.NewTextContentBlock(note)
	return llm.ToolResult{
		Output:           []llm.ContentBlock{textBlock, block},
		StructuredOutput: result,
		Metadata: llm.ToolMetadata{
			Title:     "Read " + inputPath,
			Kind:      llm.ToolKindRead,
			Locations: []llm.ToolCallLocation{{Path: path}},
			Content:   []llm.ToolCallContent{llm.NewToolCallContentBlock(textBlock), llm.NewToolCallContentBlock(block)},
		},
	}, true
}

// readDocumentViaFitz pages through a fitz-supported document. offset and
// limit are 1-indexed page numbers; a single call returns at most
// maxDocumentPagesPerCall pages to keep the response compact. When
// strictOffset is true, an offset past the end of the document errors
// (read_range semantics); when false the caller is opportunistically
// asking for "the first N pages" (read semantics).
func readDocumentViaFitz(ctx llm.ToolContext, path, mimeType string, size int64, sha string, offset, limit int, strictOffset bool) llm.ToolResult {
	doc, err := fitz.New(path)
	if err != nil {
		return ErrorResult(fmt.Errorf("open document %s: %w", path, err))
	}
	defer doc.Close()

	pageCount := doc.NumPage()
	startPage := offset
	if startPage < 1 {
		startPage = 1
	}
	if startPage > pageCount {
		if strictOffset {
			return ErrorResult(fmt.Errorf("offset %d is past document (%d pages)", startPage, pageCount))
		}
		startPage = pageCount
	}
	pages := limit
	if pages <= 0 {
		pages = maxDocumentPagesPerCall
	}
	if pages > maxDocumentPagesPerCall {
		pages = maxDocumentPagesPerCall
	}
	endPage := startPage + pages - 1
	if endPage > pageCount {
		endPage = pageCount
	}

	result := ReadResult{
		Path: path, Mode: "document",
		MimeType: mimeType, BytesTotal: int(size), SHA256: sha,
		StartLine: startPage, EndLine: endPage,
		PageCount: pageCount,
	}
	imageSupported := ctx.ModelConfig != nil && ctx.ModelConfig.SupportsModality(llm.ModalityImage)

	var textBuf strings.Builder
	output := []llm.ContentBlock{}
	content := []llm.ToolCallContent{}

	for p := startPage; p <= endPage; p++ {
		idx := p - 1
		page := ReadDocumentPage{Number: p}
		if pageText, err := doc.Text(idx); err != nil {
			page.TextErr = err.Error()
		} else {
			page.Text = strings.TrimSpace(pageText)
		}
		if page.Text != "" {
			fmt.Fprintf(&textBuf, "\n\nPage %d:\n%s", p, page.Text)
		}
		if imageSupported {
			if png, err := doc.ImagePNG(idx, 120); err != nil {
				page.ImageErr = err.Error()
			} else {
				pageSum := sha256.Sum256(png)
				pageSha := hex.EncodeToString(pageSum[:])
				page.Image = writeReadArtifact(ctx, fmt.Sprintf("read-page-%d-%s", p, pageSha[:12]), "image/png", "image", png)
				block := llm.NewImageContentBlockFromBase64(base64.StdEncoding.EncodeToString(png), "image/png")
				output = append(output, block)
				content = append(content, llm.NewToolCallContentBlock(block))
			}
		}
		result.Pages = append(result.Pages, page)
	}

	if endPage < pageCount {
		result.NextOffset = endPage + 1
		result.Notice = fmt.Sprintf("Showing pages %d-%d of %d. Use offset=%d to continue.", startPage, endPage, pageCount, result.NextOffset)
	}
	result.Content = strings.TrimSpace(textBuf.String())

	body := result.Content
	if body == "" {
		body = fmt.Sprintf("(no extractable text on pages %d-%d)", startPage, endPage)
	}
	if result.Notice != "" {
		body += "\n\n[" + result.Notice + "]"
	}
	summary := llm.NewTextContentBlock(body)
	output = append([]llm.ContentBlock{summary}, output...)
	content = append([]llm.ToolCallContent{llm.NewToolCallContentBlock(summary)}, content...)

	return llm.ToolResult{
		Output:           output,
		StructuredOutput: result,
		Metadata: llm.ToolMetadata{
			Title:     "Read " + path,
			Kind:      llm.ToolKindRead,
			Locations: []llm.ToolCallLocation{{Path: path}},
			Content:   content,
		},
	}
}

func rawContentBlockForModel(model *llm.ModelConfig, mimeType string, data []byte) (llm.ContentBlock, string, bool) {
	payload := base64.StdEncoding.EncodeToString(data)
	switch {
	case strings.HasPrefix(mimeType, "image/") && model != nil && model.SupportsModality(llm.ModalityImage) && llm.SupportedImageBytes(data):
		return llm.NewImageContentBlockFromBase64(payload, mimeType), "image", true
	case strings.HasPrefix(mimeType, "audio/") && model != nil && model.SupportsModality(llm.ModalityAudio):
		return llm.NewAudioContentBlockFromBase64(payload, mimeType), "audio", true
	case strings.HasPrefix(mimeType, "video/") && model != nil && model.SupportsModality(llm.ModalityVideo):
		return llm.NewVideoContentBlockFromBase64(payload, mimeType), "video", true
	default:
		return llm.ContentBlock{}, "", false
	}
}

func isFitzDocument(path, mimeType string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf", ".epub", ".mobi", ".docx", ".xlsx", ".pptx", ".xps", ".cbz", ".fb2":
		return true
	}
	switch strings.ToLower(mimeType) {
	case "application/pdf",
		"application/epub+zip",
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		"application/vnd.openxmlformats-officedocument.presentationml.presentation":
		return true
	}
	return false
}

func writeReadArtifact(ctx llm.ToolContext, name, mimeType, kind string, data []byte) *ReadArtifact {
	sum := sha256.Sum256(data)
	sha := hex.EncodeToString(sum[:])
	artifact := &ReadArtifact{MimeType: mimeType, Bytes: len(data), SHA256: sha, Kind: kind}
	proHome := llm.ToolProHome(ctx)
	sessionID := llm.ToolSessionID(ctx)
	actorID := llm.ToolActorID(ctx)
	runID := llm.ToolRunID(ctx)
	if proHome == "" || sessionID == "" || actorID == "" || runID == "" {
		return artifact
	}
	ext := ".bin"
	if exts, err := mime.ExtensionsByType(mimeType); err == nil && len(exts) > 0 {
		ext = exts[0]
	}
	path := llm.ActorToolOutputArtifactPath(proHome, sessionID, actorID, runID, sha[:12], name, ext)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return artifact
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return artifact
	}
	artifact.Path = path
	return artifact
}

// -----------------------------------------------------------------------------
// edit
// -----------------------------------------------------------------------------

type EditChange struct {
	OldText string `json:"oldText" jsonschema:"description=Exact text from the original file to replace. Must be unique across the file and must not overlap with other edits in this call."`
	NewText string `json:"newText" jsonschema:"description=Replacement text."`
}

type EditInput struct {
	Path  string       `json:"path" jsonschema:"description=Path to the file to edit (relative or absolute),minLength=1"`
	Edits []EditChange `json:"edits" jsonschema:"description=One or more targeted replacements. Each oldText must appear exactly once in the original file and must not overlap with other edits."`
}

type EditResult struct {
	Path           string `json:"path"`
	EditsApplied   int    `json:"edits_applied"`
	Diff           string `json:"diff"`
	FirstChangedLn int    `json:"first_changed_line,omitempty"`
}

const editDescription = `Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. When changing multiple separate spots in the same file, prefer one edit call with multiple entries in edits[] rather than multiple edit calls. Each edits[].oldText is matched against the original file, not after earlier edits. Do not include overlapping or nested edits. Keep oldText as small as possible while still being unique.`

func NewEditTool(_ FileToolsOptions) llm.Tool {
	return NewGenericTool("edit", editDescription, func(ctx llm.ToolContext, input EditInput) llm.ToolResult {
		if len(input.Edits) == 0 {
			return ErrorResult(fmt.Errorf("edits must contain at least one replacement"))
		}
		path, err := resolveToolPath(ctx.WorkingDirectory, input.Path)
		if err != nil {
			return ErrorResult(err)
		}
		return withFileMutation(path, func() llm.ToolResult {
			data, err := os.ReadFile(path)
			if err != nil {
				return ErrorResult(fmt.Errorf("edit %s: %w", input.Path, err))
			}
			if isBinary(data) {
				return ErrorResult(fmt.Errorf("refusing to edit binary file %s", input.Path))
			}
			orig := string(data)
			bom, body := stripBOM(orig)
			ending := detectLineEnding(body)
			normalized := strings.ReplaceAll(strings.ReplaceAll(body, "\r\n", "\n"), "\r", "\n")

			matches := make([]matchSpan, 0, len(input.Edits))
			for i, e := range input.Edits {
				if e.OldText == "" {
					return ErrorResult(fmt.Errorf("edits[%d].oldText is empty", i))
				}
				first := strings.Index(normalized, e.OldText)
				if first == -1 {
					return ErrorResult(fmt.Errorf("edits[%d].oldText not found in %s", i, input.Path))
				}
				if strings.Index(normalized[first+1:], e.OldText) != -1 {
					return ErrorResult(fmt.Errorf("edits[%d].oldText is not unique in %s — include more surrounding context to disambiguate", i, input.Path))
				}
				matches = append(matches, matchSpan{
					index:   i,
					start:   first,
					end:     first + len(e.OldText),
					newText: e.NewText,
				})
			}
			sort.Slice(matches, func(i, j int) bool { return matches[i].start < matches[j].start })
			for i := 1; i < len(matches); i++ {
				if matches[i].start < matches[i-1].end {
					return ErrorResult(fmt.Errorf("edits[%d] and edits[%d] overlap — merge them into a single edit",
						matches[i-1].index, matches[i].index))
				}
			}

			var b strings.Builder
			b.Grow(len(normalized))
			cur := 0
			for _, m := range matches {
				b.WriteString(normalized[cur:m.start])
				b.WriteString(m.newText)
				cur = m.end
			}
			b.WriteString(normalized[cur:])
			newBody := b.String()
			final := bom + restoreLineEnding(newBody, ending)

			if err := writeFileAtomic(path, []byte(final), 0o644); err != nil {
				return ErrorResult(fmt.Errorf("write %s: %w", input.Path, err))
			}

			diff, firstChanged := simpleUnifiedDiff(input.Path, normalized, newBody)
			result := EditResult{
				Path:           path,
				EditsApplied:   len(input.Edits),
				Diff:           diff,
				FirstChangedLn: firstChanged,
			}
			locations := []llm.ToolCallLocation{{Path: path}}
			if firstChanged > 0 {
				line := firstChanged
				locations = append(locations, llm.ToolCallLocation{Path: path, Line: &line})
			}
			msg := fmt.Sprintf("Applied %d edit(s) to %s.\n\n%s", len(input.Edits), input.Path, diff)
			block := llm.NewTextContentBlock(msg)
			oldText := normalized
			return llm.ToolResult{
				Output:           []llm.ContentBlock{block},
				StructuredOutput: result,
				Metadata: llm.ToolMetadata{
					Title:     "Edit " + input.Path,
					Kind:      llm.ToolKindEdit,
					Locations: locations,
					Content:   []llm.ToolCallContent{llm.NewToolCallDiff(path, newBody, &oldText)},
				},
			}
		})
	}).WithOutputSchema(ReflectValue(EditResult{})).WithPermission(PermissionExtractor{
		Key:       "edit",
		MatchMode: llm.PermissionMatchModeExact,
		Fields:    []PermissionField{{Name: "path", Transform: "path", Base: "cwd"}},
		Options:   FileWritePermissionOptions("edit"),
	})
}

type matchSpan struct {
	index   int
	start   int
	end     int
	newText string
}

// -----------------------------------------------------------------------------
// write
// -----------------------------------------------------------------------------

type WriteInput struct {
	Path    string `json:"path" jsonschema:"description=Path to the file to write (relative or absolute),minLength=1"`
	Content string `json:"content" jsonschema:"description=Full file content. Overwrites if the file exists."`
}

type WriteResult struct {
	Path       string `json:"path"`
	BytesWrote int    `json:"bytes"`
	Created    bool   `json:"created"`
}

const writeDescription = `Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories. Prefer edit for changes to existing files — use write only for new files or complete rewrites.`

func NewWriteTool(_ FileToolsOptions) llm.Tool {
	return NewGenericTool("write", writeDescription, func(ctx llm.ToolContext, input WriteInput) llm.ToolResult {
		path, err := resolveToolPath(ctx.WorkingDirectory, input.Path)
		if err != nil {
			return ErrorResult(err)
		}
		return withFileMutation(path, func() llm.ToolResult {
			_, existedErr := os.Stat(path)
			existed := existedErr == nil
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return ErrorResult(fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err))
			}
			if err := writeFileAtomic(path, []byte(input.Content), 0o644); err != nil {
				return ErrorResult(fmt.Errorf("write %s: %w", input.Path, err))
			}
			result := WriteResult{Path: path, BytesWrote: len(input.Content), Created: !existed}
			verb := "Wrote"
			if !existed {
				verb = "Created"
			}
			msg := fmt.Sprintf("%s %d bytes to %s", verb, len(input.Content), input.Path)
			block := llm.NewTextContentBlock(msg)
			return llm.ToolResult{
				Output:           []llm.ContentBlock{block},
				StructuredOutput: result,
				Metadata: llm.ToolMetadata{
					Title:     verb + " " + input.Path,
					Kind:      llm.ToolKindEdit,
					Locations: []llm.ToolCallLocation{{Path: path}},
					Content:   []llm.ToolCallContent{llm.NewToolCallDiff(path, input.Content, nil)},
				},
			}
		})
	}).WithOutputSchema(ReflectValue(WriteResult{})).WithPermission(PermissionExtractor{
		Key:       "write",
		MatchMode: llm.PermissionMatchModeExact,
		Fields:    []PermissionField{{Name: "path", Transform: "path", Base: "cwd"}},
		Options:   FileWritePermissionOptions("write"),
	})
}

// -----------------------------------------------------------------------------
// grep
// -----------------------------------------------------------------------------

type GrepInput struct {
	Pattern    string `json:"pattern" jsonschema:"description=Search pattern (regex by default; set literal:true for plain text),minLength=1"`
	Path       string `json:"path,omitempty" jsonschema:"description=Directory or file to search (default: working directory)"`
	Glob       string `json:"glob,omitempty" jsonschema:"description=Filter files by glob pattern e.g. '*.go' or '**/*_test.go'"`
	IgnoreCase bool   `json:"ignoreCase,omitempty" jsonschema:"description=Case-insensitive search"`
	Literal    bool   `json:"literal,omitempty" jsonschema:"description=Treat pattern as a literal string rather than a regex"`
	Context    int    `json:"context,omitempty" jsonschema:"description=Lines of context to show before and after each match"`
	Limit      int    `json:"limit,omitempty" jsonschema:"description=Maximum number of matches to return (default: 100)"`
	NoIgnore   bool   `json:"noIgnore,omitempty" jsonschema:"description=If true, do not honor .gitignore"`
}

type GrepMatch struct {
	Path   string `json:"path"`
	Line   int    `json:"line"`
	Text   string `json:"text"`
	Marker string `json:"marker,omitempty"` // ":" for match, "-" for context
}

type GrepResult struct {
	Pattern      string      `json:"pattern"`
	Path         string      `json:"path"`
	Matches      []GrepMatch `json:"matches"`
	Total        int         `json:"total"`
	LimitReached bool        `json:"limit_reached,omitempty"`
	Truncated    bool        `json:"truncated,omitempty"`
}

const grepDescription = `Search file contents for a pattern. Returns lines as 'relative/path:line: text'. Respects .gitignore unless noIgnore:true. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars — read the file if you need the full line. Regex is re2-compatible; set literal:true for plain text search. Walk skips system dirs (/proc, /sys, /usr, /lib, /var, /etc, /bin, /sbin, /opt, /tmp, /dev, /run, /boot) and any single file larger than 1 MiB; total walked content is capped at 50 MiB. If you point grep at a wide path like "/" expect mostly-empty results — narrow the path to a project directory (e.g. /benchmark) for useful searches.`

func NewGrepTool(_ FileToolsOptions) llm.Tool {
	return NewGenericTool("grep", grepDescription, func(ctx llm.ToolContext, input GrepInput) llm.ToolResult {
		if strings.TrimSpace(input.Pattern) == "" {
			return ErrorResult(fmt.Errorf("pattern is required"))
		}
		root := ctx.WorkingDirectory
		if input.Path != "" {
			r, err := resolveToolPath(ctx.WorkingDirectory, input.Path)
			if err != nil {
				return ErrorResult(err)
			}
			root = r
		}
		limit := input.Limit
		if limit <= 0 {
			limit = grepDefaultLimit
		}

		re, err := compileGrepPattern(input.Pattern, input.Literal, input.IgnoreCase)
		if err != nil {
			return ErrorResult(err)
		}

		var globs []compiledGlob
		if input.Glob != "" {
			gs, err := compileGlobs([]string{input.Glob})
			if err != nil {
				return ErrorResult(err)
			}
			globs = gs
		}

		info, err := os.Stat(root)
		if err != nil {
			return ErrorResult(fmt.Errorf("grep %s: %w", root, err))
		}
		var files []string
		if info.IsDir() {
			files, err = walkSearchableFiles(ctx.Context, root, input.NoIgnore, globs)
			if err != nil {
				return ErrorResult(err)
			}
		} else {
			files = []string{root}
		}

		matches := make([]GrepMatch, 0, limit)
		total := 0
		limitReached := false
		bytesUsed := 0
		bytesCap := int(defaultMaxBytes)
		truncated := false
		linesTruncated := false

		var walkBytesRead int64
	walk:
		for _, file := range files {
			if ctx.Context != nil {
				if err := ctx.Context.Err(); err != nil {
					return ErrorResult(err)
				}
			}
			// PRE-READ size check. Without this an `os.ReadFile` on a
			// large binary (the proagent-eval Go binary is 50-100MB)
			// fills the heap before the isBinary check fires. Stat is
			// O(1) and harmless; skip the file entirely if it's bigger
			// than the per-file cap.
			if stat, statErr := os.Stat(file); statErr == nil && stat.Size() > maxGrepFileSize {
				truncated = true
				continue
			}
			// Total walk budget: stop dead once we've read enough total
			// bytes. Defends against deep trees of small files (locale
			// dirs, /var/log fragments) accumulating into OOM territory.
			if walkBytesRead >= maxGrepWalkBytes {
				truncated = true
				break
			}
			data, err := os.ReadFile(file)
			if err != nil {
				continue
			}
			walkBytesRead += int64(len(data))
			if isBinary(data) {
				continue
			}
			rel := relPath(root, file, info.IsDir())
			lines := strings.Split(string(data), "\n")
			for idx, line := range lines {
				if !re.MatchString(line) {
					continue
				}
				total++
				start := idx - input.Context
				if start < 0 {
					start = 0
				}
				end := idx + input.Context
				if end >= len(lines) {
					end = len(lines) - 1
				}
				for j := start; j <= end; j++ {
					trText, trWas := truncateLine(lines[j], grepMaxLineLength)
					if trWas {
						linesTruncated = true
					}
					marker := ":"
					if j != idx {
						marker = "-"
					}
					gm := GrepMatch{Path: rel, Line: j + 1, Text: trText, Marker: marker}
					formatted := fmt.Sprintf("%s%s%d%s %s", gm.Path, marker, gm.Line, marker, gm.Text)
					if bytesUsed+len(formatted)+1 > bytesCap {
						truncated = true
						break walk
					}
					matches = append(matches, gm)
					bytesUsed += len(formatted) + 1
				}
				if total >= limit {
					limitReached = true
					break walk
				}
			}
		}

		var sb strings.Builder
		for _, m := range matches {
			fmt.Fprintf(&sb, "%s%s%d%s %s\n", m.Path, m.Marker, m.Line, m.Marker, m.Text)
		}
		body := strings.TrimRight(sb.String(), "\n")
		if total == 0 {
			body = "No matches found"
		}
		var notices []string
		if limitReached {
			notices = append(notices, fmt.Sprintf("%d matches limit reached. Use limit=%d for more, or refine pattern", limit, limit*2))
		}
		if truncated {
			notices = append(notices, fmt.Sprintf("%s limit reached", formatSize(bytesCap)))
		}
		if linesTruncated {
			notices = append(notices, fmt.Sprintf("Some lines truncated to %d chars. Use read to see full lines", grepMaxLineLength))
		}
		if len(notices) > 0 {
			body += "\n\n[" + strings.Join(notices, ". ") + "]"
		}

		result := GrepResult{
			Pattern: input.Pattern, Path: root,
			Matches: matches, Total: total,
			LimitReached: limitReached, Truncated: truncated,
		}
		locations := []llm.ToolCallLocation{{Path: root}}
		for _, m := range matches {
			if m.Marker == ":" {
				line := m.Line
				locations = append(locations, llm.ToolCallLocation{Path: filepath.Join(root, m.Path), Line: &line})
			}
		}
		block := llm.NewTextContentBlock(body)
		return llm.ToolResult{
			Output:           []llm.ContentBlock{block},
			StructuredOutput: result,
			Metadata: llm.ToolMetadata{
				Title:     "Grep " + input.Pattern,
				Kind:      llm.ToolKindSearch,
				Locations: locations,
				Content:   []llm.ToolCallContent{llm.NewToolCallContentBlock(block)},
			},
		}
	}).WithOutputSchema(ReflectValue(GrepResult{}))
}

func compileGrepPattern(pattern string, literal, ignoreCase bool) (*regexp.Regexp, error) {
	expr := pattern
	if literal {
		expr = regexp.QuoteMeta(expr)
	}
	if ignoreCase {
		expr = "(?i)" + expr
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern %q: %w", pattern, err)
	}
	return re, nil
}

func walkSearchableFiles(ctx context.Context, root string, noIgnore bool, includeGlobs []compiledGlob) ([]string, error) {
	matcher := newRepoMatcher(root, noIgnore)
	var files []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			if os.IsPermission(walkErr) || os.IsNotExist(walkErr) {
				if d != nil && d.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
			return walkErr
		}
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if path == root {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			if !noIgnore && isNoisyDir(d.Name()) {
				return fs.SkipDir
			}
			if matcherIgnored(matcher, rel, true) {
				return fs.SkipDir
			}
			return nil
		}
		if matcherIgnored(matcher, rel, false) {
			return nil
		}
		if len(includeGlobs) > 0 && !matchGlob(includeGlobs, rel, false) {
			return nil
		}
		files = append(files, path)
		return nil
	})
	sort.Slice(files, func(i, j int) bool { return files[i] < files[j] })
	return files, err
}

func relPath(root, path string, rootIsDir bool) string {
	if !rootIsDir {
		return filepath.Base(path)
	}
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return filepath.Base(path)
	}
	return filepath.ToSlash(rel)
}

// -----------------------------------------------------------------------------
// find
// -----------------------------------------------------------------------------

type FindInput struct {
	Pattern  string `json:"pattern" jsonschema:"description=Glob pattern to match files e.g. '*.go' or 'cmd/**/*.ts',minLength=1"`
	Path     string `json:"path,omitempty" jsonschema:"description=Directory to search in (default: working directory)"`
	Limit    int    `json:"limit,omitempty" jsonschema:"description=Maximum number of results (default: 1000)"`
	NoIgnore bool   `json:"noIgnore,omitempty" jsonschema:"description=If true, do not honor .gitignore"`
}

type FindResult struct {
	Pattern      string   `json:"pattern"`
	Path         string   `json:"path"`
	Files        []string `json:"files"`
	Total        int      `json:"total"`
	LimitReached bool     `json:"limit_reached,omitempty"`
	Truncated    bool     `json:"truncated,omitempty"`
}

const findDescription = `Search for files by glob pattern. Returns matching paths relative to the search directory. Respects .gitignore. Output is truncated to 1000 results or 50KB (whichever is hit first). Use '**/' prefix to match across directories.`

func NewFindTool(_ FileToolsOptions) llm.Tool {
	return NewGenericTool("find", findDescription, func(ctx llm.ToolContext, input FindInput) llm.ToolResult {
		if strings.TrimSpace(input.Pattern) == "" {
			return ErrorResult(fmt.Errorf("pattern is required"))
		}
		root := ctx.WorkingDirectory
		if input.Path != "" {
			r, err := resolveToolPath(ctx.WorkingDirectory, input.Path)
			if err != nil {
				return ErrorResult(err)
			}
			root = r
		}
		limit := input.Limit
		if limit <= 0 {
			limit = findDefaultLimit
		}
		globs, err := compileGlobs([]string{input.Pattern})
		if err != nil {
			return ErrorResult(err)
		}
		files, err := walkSearchableFiles(ctx.Context, root, input.NoIgnore, globs)
		if err != nil {
			return ErrorResult(err)
		}
		total := len(files)
		limitReached := false
		if len(files) > limit {
			files = files[:limit]
			limitReached = true
		}
		rels := make([]string, 0, len(files))
		for _, f := range files {
			rels = append(rels, relPath(root, f, true))
		}
		raw := strings.Join(rels, "\n")
		trunc := truncateHead(raw, len(rels)+1, int(defaultMaxBytes))
		body := trunc.Content
		if total == 0 {
			body = "No files found matching pattern"
		}
		var notices []string
		if limitReached {
			notices = append(notices, fmt.Sprintf("%d results limit reached. Use limit=%d for more, or refine pattern", limit, limit*2))
		}
		if trunc.Truncated {
			notices = append(notices, fmt.Sprintf("%s limit reached", formatSize(int(defaultMaxBytes))))
		}
		if len(notices) > 0 {
			body += "\n\n[" + strings.Join(notices, ". ") + "]"
		}
		result := FindResult{
			Pattern: input.Pattern, Path: root,
			Files: rels, Total: total,
			LimitReached: limitReached, Truncated: trunc.Truncated,
		}
		block := llm.NewTextContentBlock(body)
		return llm.ToolResult{
			Output:           []llm.ContentBlock{block},
			StructuredOutput: result,
			Metadata: llm.ToolMetadata{
				Title:     "Find " + input.Pattern,
				Kind:      llm.ToolKindSearch,
				Locations: []llm.ToolCallLocation{{Path: root}},
				Content:   []llm.ToolCallContent{llm.NewToolCallContentBlock(block)},
			},
		}
	}).WithOutputSchema(ReflectValue(FindResult{}))
}

// -----------------------------------------------------------------------------
// ls
// -----------------------------------------------------------------------------

type LsInput struct {
	Path  string `json:"path,omitempty" jsonschema:"description=Directory to list (default: working directory)"`
	Limit int    `json:"limit,omitempty" jsonschema:"description=Maximum number of entries to return (default: 500)"`
}

type LsEntry struct {
	Name  string `json:"name"`
	IsDir bool   `json:"is_dir,omitempty"`
}

type LsResult struct {
	Path         string    `json:"path"`
	Entries      []LsEntry `json:"entries"`
	Total        int       `json:"total"`
	LimitReached bool      `json:"limit_reached,omitempty"`
}

const lsDescription = `List directory contents. Returns entries sorted case-insensitively, with '/' suffix for directories. Includes dotfiles. Truncated to 500 entries by default.`

func NewLsTool(_ FileToolsOptions) llm.Tool {
	return NewGenericTool("ls", lsDescription, func(ctx llm.ToolContext, input LsInput) llm.ToolResult {
		dir := ctx.WorkingDirectory
		if input.Path != "" {
			r, err := resolveToolPath(ctx.WorkingDirectory, input.Path)
			if err != nil {
				return ErrorResult(err)
			}
			dir = r
		}
		limit := input.Limit
		if limit <= 0 {
			limit = lsDefaultLimit
		}
		info, err := os.Stat(dir)
		if err != nil {
			return ErrorResult(fmt.Errorf("ls %s: %w", dir, err))
		}
		if !info.IsDir() {
			return ErrorResult(fmt.Errorf("not a directory: %s", dir))
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			return ErrorResult(fmt.Errorf("ls %s: %w", dir, err))
		}
		sort.Slice(entries, func(i, j int) bool {
			return strings.ToLower(entries[i].Name()) < strings.ToLower(entries[j].Name())
		})
		total := len(entries)
		limitReached := false
		if len(entries) > limit {
			entries = entries[:limit]
			limitReached = true
		}
		result := LsResult{Path: dir, Total: total, LimitReached: limitReached}
		var sb strings.Builder
		for _, e := range entries {
			suffix := ""
			if e.IsDir() {
				suffix = "/"
			}
			result.Entries = append(result.Entries, LsEntry{Name: e.Name(), IsDir: e.IsDir()})
			sb.WriteString(e.Name())
			sb.WriteString(suffix)
			sb.WriteByte('\n')
		}
		body := strings.TrimRight(sb.String(), "\n")
		if total == 0 {
			body = "(empty directory)"
		}
		if limitReached {
			body += fmt.Sprintf("\n\n[%d entries limit reached. Use limit=%d for more]", limit, limit*2)
		}
		block := llm.NewTextContentBlock(body)
		return llm.ToolResult{
			Output:           []llm.ContentBlock{block},
			StructuredOutput: result,
			Metadata: llm.ToolMetadata{
				Title:     "Ls " + dir,
				Kind:      llm.ToolKindOther,
				Locations: []llm.ToolCallLocation{{Path: dir}},
				Content:   []llm.ToolCallContent{llm.NewToolCallContentBlock(block)},
			},
		}
	}).WithOutputSchema(ReflectValue(LsResult{}))
}

// -----------------------------------------------------------------------------
// Registry + permission options
// -----------------------------------------------------------------------------

func RegisterFileToolFactories(reg *llm.ToolRegistry) *llm.ToolRegistry {
	if reg == nil {
		reg = llm.NewToolRegistry()
	}
	unmarshalOpts := func(raw json.RawMessage) (FileToolsOptions, error) {
		var opts FileToolsOptions
		if len(raw) > 0 {
			if err := json.Unmarshal(raw, &opts); err != nil {
				return opts, err
			}
		}
		return opts, nil
	}
	reg.MustRegister("read", func(_ context.Context, _ llm.ToolBuildContext, raw json.RawMessage) (llm.Tool, error) {
		o, err := unmarshalOpts(raw)
		if err != nil {
			return nil, err
		}
		return NewReadTool(o), nil
	})
	reg.MustRegister("read_range", func(_ context.Context, _ llm.ToolBuildContext, raw json.RawMessage) (llm.Tool, error) {
		o, err := unmarshalOpts(raw)
		if err != nil {
			return nil, err
		}
		return NewReadRangeTool(o), nil
	})
	reg.MustRegister("edit", func(_ context.Context, _ llm.ToolBuildContext, raw json.RawMessage) (llm.Tool, error) {
		o, err := unmarshalOpts(raw)
		if err != nil {
			return nil, err
		}
		return NewEditTool(o), nil
	})
	reg.MustRegister("write", func(_ context.Context, _ llm.ToolBuildContext, raw json.RawMessage) (llm.Tool, error) {
		o, err := unmarshalOpts(raw)
		if err != nil {
			return nil, err
		}
		return NewWriteTool(o), nil
	})
	reg.MustRegister("grep", func(_ context.Context, _ llm.ToolBuildContext, raw json.RawMessage) (llm.Tool, error) {
		o, err := unmarshalOpts(raw)
		if err != nil {
			return nil, err
		}
		return NewGrepTool(o), nil
	})
	reg.MustRegister("find", func(_ context.Context, _ llm.ToolBuildContext, raw json.RawMessage) (llm.Tool, error) {
		o, err := unmarshalOpts(raw)
		if err != nil {
			return nil, err
		}
		return NewFindTool(o), nil
	})
	reg.MustRegister("ls", func(_ context.Context, _ llm.ToolBuildContext, raw json.RawMessage) (llm.Tool, error) {
		o, err := unmarshalOpts(raw)
		if err != nil {
			return nil, err
		}
		return NewLsTool(o), nil
	})
	return reg
}

func FileWritePermissionOptions(toolName string) []llm.PermissionOption {
	return []llm.PermissionOption{
		llm.AllowOnceOption(),
		llm.AllowAlwaysOption("allow_always_file_session", "Always allow this file (session)", llm.PermissionScopeSession, llm.PermissionTargetFile, llm.PermissionGrant{Field: "path", MatchMode: llm.PermissionMatchModeExact}),
		llm.AllowAlwaysOption("allow_always_file_global", "Always allow this file (global)", llm.PermissionScopeGlobal, llm.PermissionTargetFile, llm.PermissionGrant{Field: "path", MatchMode: llm.PermissionMatchModeExact}),
		llm.AllowAlwaysOption("allow_always_folder_session", "Always allow this folder (session)", llm.PermissionScopeSession, llm.PermissionTargetFolder, llm.PermissionGrant{Field: "path", MatchMode: llm.PermissionMatchModePath, Transform: "dirname"}),
		llm.AllowAlwaysOption("allow_always_folder_global", "Always allow this folder (global)", llm.PermissionScopeGlobal, llm.PermissionTargetFolder, llm.PermissionGrant{Field: "path", MatchMode: llm.PermissionMatchModePath, Transform: "dirname"}),
		llm.AllowAlwaysOption("allow_always_tool_session", "Always allow "+toolName+" (session)", llm.PermissionScopeSession, llm.PermissionTargetTool),
		llm.AllowAlwaysOption("allow_always_tool_global", "Always allow "+toolName+" (global)", llm.PermissionScopeGlobal, llm.PermissionTargetTool),
		llm.RejectOnceOption(),
		llm.RejectAlwaysOption("reject_always_file_session", "Always reject this file (session)", llm.PermissionScopeSession, llm.PermissionTargetFile, llm.PermissionGrant{Field: "path", MatchMode: llm.PermissionMatchModeExact}),
		llm.RejectAlwaysOption("reject_always_file_global", "Always reject this file (global)", llm.PermissionScopeGlobal, llm.PermissionTargetFile, llm.PermissionGrant{Field: "path", MatchMode: llm.PermissionMatchModeExact}),
		llm.RejectAlwaysOption("reject_always_folder_session", "Always reject this folder (session)", llm.PermissionScopeSession, llm.PermissionTargetFolder, llm.PermissionGrant{Field: "path", MatchMode: llm.PermissionMatchModePath, Transform: "dirname"}),
		llm.RejectAlwaysOption("reject_always_folder_global", "Always reject this folder (global)", llm.PermissionScopeGlobal, llm.PermissionTargetFolder, llm.PermissionGrant{Field: "path", MatchMode: llm.PermissionMatchModePath, Transform: "dirname"}),
		llm.RejectAlwaysOption("reject_always_tool_session", "Always reject "+toolName+" (session)", llm.PermissionScopeSession, llm.PermissionTargetTool),
		llm.RejectAlwaysOption("reject_always_tool_global", "Always reject "+toolName+" (global)", llm.PermissionScopeGlobal, llm.PermissionTargetTool),
	}
}

// -----------------------------------------------------------------------------
// Line endings, BOM, atomic write, diff
// -----------------------------------------------------------------------------

const utf8BOM = "\ufeff"

func stripBOM(s string) (bom, body string) {
	if strings.HasPrefix(s, utf8BOM) {
		return utf8BOM, strings.TrimPrefix(s, utf8BOM)
	}
	return "", s
}

func detectLineEnding(s string) string {
	if strings.Contains(s, "\r\n") {
		return "\r\n"
	}
	return "\n"
}

func restoreLineEnding(s, ending string) string {
	if ending == "\r\n" {
		return strings.ReplaceAll(s, "\n", "\r\n")
	}
	return s
}

func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	if err := os.Chmod(tmp.Name(), perm); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// simpleUnifiedDiff renders a minimal context-3 unified diff. It is not
// optimized for big files but is enough for the tool's "show what changed"
// output. firstChanged returns the new-file line number of the first
// differing line (1-indexed), or 0 if there is no diff.
func simpleUnifiedDiff(label, oldText, newText string) (string, int) {
	if oldText == newText {
		return "", 0
	}
	oldLines := strings.Split(oldText, "\n")
	newLines := strings.Split(newText, "\n")
	hunks := diffHunks(oldLines, newLines, 3)
	if len(hunks) == 0 {
		return "", 0
	}
	var b strings.Builder
	fmt.Fprintf(&b, "--- a/%s\n+++ b/%s\n", label, label)
	firstChanged := 0
	for _, h := range hunks {
		fmt.Fprintf(&b, "@@ -%d,%d +%d,%d @@\n", h.oldStart+1, h.oldLen, h.newStart+1, h.newLen)
		newLine := h.newStart
		for _, line := range h.lines {
			b.WriteString(line)
			b.WriteByte('\n')
			if firstChanged == 0 && (strings.HasPrefix(line, "+") || strings.HasPrefix(line, "-")) {
				if strings.HasPrefix(line, "+") {
					firstChanged = newLine + 1
				} else {
					firstChanged = newLine + 1
				}
			}
			if !strings.HasPrefix(line, "-") {
				newLine++
			}
		}
	}
	return b.String(), firstChanged
}

type diffHunk struct {
	oldStart, oldLen int
	newStart, newLen int
	lines            []string
}

// diffHunks runs a minimal LCS diff and emits hunks with `context` lines
// of surrounding unchanged context. It is O(n*m) — fine for the file
// sizes the edit tool handles (50KB cap on reads upstream).
func diffHunks(a, b []string, context int) []diffHunk {
	n, m := len(a), len(b)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if a[i] == b[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else if dp[i+1][j] >= dp[i][j+1] {
				dp[i][j] = dp[i+1][j]
			} else {
				dp[i][j] = dp[i][j+1]
			}
		}
	}
	type op struct {
		kind byte // ' ', '-', '+'
		text string
		ai   int // index into a (for - and ' ')
		bi   int // index into b (for + and ' ')
	}
	var ops []op
	i, j := 0, 0
	for i < n && j < m {
		if a[i] == b[j] {
			ops = append(ops, op{' ', a[i], i, j})
			i++
			j++
		} else if dp[i+1][j] >= dp[i][j+1] {
			ops = append(ops, op{'-', a[i], i, -1})
			i++
		} else {
			ops = append(ops, op{'+', b[j], -1, j})
			j++
		}
	}
	for ; i < n; i++ {
		ops = append(ops, op{'-', a[i], i, -1})
	}
	for ; j < m; j++ {
		ops = append(ops, op{'+', b[j], -1, j})
	}

	var hunks []diffHunk
	k := 0
	for k < len(ops) {
		if ops[k].kind == ' ' {
			k++
			continue
		}
		start := k - context
		if start < 0 {
			start = 0
		}
		end := k
		gap := 0
		for end < len(ops) {
			if ops[end].kind == ' ' {
				gap++
				if gap > context*2 {
					break
				}
			} else {
				gap = 0
			}
			end++
		}
		trim := end
		for trim > 0 && ops[trim-1].kind == ' ' && (trim-k) > context {
			trim--
		}
		end = trim

		h := diffHunk{}
		var lines []string
		var firstA, firstB = -1, -1
		var lastA, lastB = -1, -1
		for _, o := range ops[start:end] {
			lines = append(lines, string(o.kind)+o.text)
			switch o.kind {
			case ' ':
				if firstA < 0 {
					firstA = o.ai
				}
				if firstB < 0 {
					firstB = o.bi
				}
				lastA, lastB = o.ai, o.bi
			case '-':
				if firstA < 0 {
					firstA = o.ai
				}
				lastA = o.ai
			case '+':
				if firstB < 0 {
					firstB = o.bi
				}
				lastB = o.bi
			}
		}
		if firstA < 0 {
			firstA = 0
		}
		if firstB < 0 {
			firstB = 0
		}
		h.oldStart = firstA
		h.newStart = firstB
		if lastA >= 0 {
			h.oldLen = lastA - firstA + 1
		}
		if lastB >= 0 {
			h.newLen = lastB - firstB + 1
		}
		h.lines = lines
		hunks = append(hunks, h)
		k = end
	}
	return hunks
}
