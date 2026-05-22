package rlm

// Document represents a document in the RLM kernel.
type Document struct {
	ID       string `json:"id"`
	Title    string `json:"title,omitempty"`
	URI      string `json:"uri,omitempty"`
	Text     string `json:"text"`
	Metadata any    `json:"metadata,omitempty"`
}

// WorkspaceOptions controls repository indexing for retrieval.
type WorkspaceOptions struct {
	Root         string   `json:"root,omitempty" jsonschema:"description=Workspace root to index; defaults to current working directory"`
	MaxFiles     int      `json:"max_files,omitempty" jsonschema:"description=Maximum number of files to index,minimum=1"`
	MaxFileBytes int64    `json:"max_file_bytes,omitempty" jsonschema:"description=Maximum bytes per indexed file,minimum=1"`
	IncludeGlobs []string `json:"include_globs,omitempty" jsonschema:"description=Optional file globs to include"`
	ExcludeGlobs []string `json:"exclude_globs,omitempty" jsonschema:"description=Optional file globs or directories to exclude"`
	Refresh      bool     `json:"refresh,omitempty" jsonschema:"description=Force re-indexing of the workspace"`
}

// Input represents input to the RLM tool.
type Input struct {
	Mode             string           `json:"mode" jsonschema:"description=RLM action,enum=index,enum=index_workspace,enum=retrieve,enum=analyze,enum=lambda_analyze"`
	Query            string           `json:"query,omitempty" jsonschema:"description=Search query (required for retrieve/analyze)"`
	Documents        []Document       `json:"documents,omitempty" jsonschema:"description=Documents to index (required for index)"`
	ChunkSize        int              `json:"chunk_size,omitempty" jsonschema:"description=Chunk size for indexing,minimum=1"`
	Replace          bool             `json:"replace,omitempty" jsonschema:"description=Replace existing index entries"`
	TopK             int              `json:"top_k,omitempty" jsonschema:"description=Number of top results to return,minimum=1"`
	IncludeIndexed   *bool            `json:"include_indexed,omitempty" jsonschema:"description=Include indexed documents in search"`
	IncludeWorkspace *bool            `json:"include_workspace,omitempty" jsonschema:"description=Include workspace in search"`
	Workspace        WorkspaceOptions `json:"workspace,omitempty" jsonschema:"description=Workspace indexing options (for index_workspace)"`
}

// QueryResult is a bounded matching chunk from a document.
type QueryResult struct {
	DocumentID string         `json:"document_id"`
	Title      string         `json:"title,omitempty"`
	URI        string         `json:"uri,omitempty"`
	ChunkIndex int            `json:"chunk_index"`
	Start      int            `json:"start"`
	End        int            `json:"end"`
	Score      float64        `json:"score"`
	Text       string         `json:"text"`
	Terms      []string       `json:"terms,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// LambdaPlan describes the bounded recursive decomposition used by lambda_analyze.
type LambdaPlan struct {
	TaskType      string  `json:"task_type"`
	ComposeOp     string  `json:"compose_op"`
	Branching     int     `json:"branching"`
	LeafChunkSize int     `json:"leaf_chunk_size"`
	Depth         int     `json:"depth"`
	CostEstimate  float64 `json:"cost_estimate"`
}

// IndexResult describes indexed corpus state.
type IndexResult struct {
	IndexedIDs       []string `json:"indexed_ids"`
	IndexedDocuments int      `json:"indexed_documents"`
	WorkspaceRoot    string   `json:"workspace_root,omitempty"`
	WorkspaceFiles   int      `json:"workspace_files,omitempty"`
	SkippedFiles     int      `json:"skipped_files,omitempty"`
}

// Result represents the result of an RLM operation.
type Result struct {
	Documents []Document    `json:"documents,omitempty"`
	Results   []QueryResult `json:"results,omitempty"`
	Analysis  string        `json:"analysis,omitempty"`
	Citations []string      `json:"citations,omitempty"`
	Mode      string        `json:"mode,omitempty"`
	Plan      *LambdaPlan   `json:"plan,omitempty"`
	Index     *IndexResult  `json:"index,omitempty"`
}
