package llm

import (
	"context"
	"encoding/json"
)

type ToolContext struct {
	context.Context

	WorkingDirectory string

	Vars        map[string]any
	ModelConfig *ModelConfig
	Permission  ToolPermissionHandler

	Emit func(event any)
	// NestedToolUpdate is called before and after each tool invocation
	// from inside the js_execute sandbox so the harness can surface them
	// to the bridge / client.
	NestedToolUpdate func(ctx context.Context, call ToolUse, status string, result *ToolResult) error
}

type ToolPermissionHandler func(ctx context.Context, req PermissionRequest) (PermissionResponse, error)

type Tool interface {
	Schema() ToolSchema
	Execute(ctx ToolContext, input map[string]any) ToolResult
}

type GuardedTool interface {
	Tool
	GetPermissions(ctx ToolContext, input map[string]any) ([]PermissionGuard, error)
}

type PermissionRequest struct {
	ToolCall ToolUse           `json:"tool_call"`
	Guards   []PermissionGuard `json:"guards"`
}

type PermissionResponse struct {
	Approved bool                 `json:"approved"`
	Results  []PermissionDecision `json:"results,omitempty"`
}

type PermissionDecision struct {
	Guard    PermissionGuard `json:"guard"`
	Approved bool            `json:"approved"`
	OptionID string          `json:"option_id,omitempty"`
}

type PermissionGuard struct {
	Key        string              `json:"key"`
	Specifiers []string            `json:"specifiers"`
	MatchMode  PermissionMatchMode `json:"match_mode,omitempty"`
	Options    []PermissionOption  `json:"options,omitempty"`
}

type PermissionMatchMode string

const (
	PermissionMatchModeExact  PermissionMatchMode = "exact"
	PermissionMatchModePrefix PermissionMatchMode = "prefix"
	PermissionMatchModePath   PermissionMatchMode = "path"
	PermissionMatchModeGlob   PermissionMatchMode = "glob"
)

type ToolSchema struct {
	Name         string     `json:"name,omitempty"`
	Description  string     `json:"description,omitempty"`
	InputSchema  JSONSchema `json:"input_schema,omitempty"`
	OutputSchema JSONSchema `json:"output_schema,omitempty"`
	Strict       bool       `json:"strict,omitempty"`
}

// UnmarshalJSON handles both wrapped format {input_schema: {...}} and direct schema format
func (t *ToolSchema) UnmarshalJSON(data []byte) error {
	// First try standard unmarshaling
	type toolSchemaAlias ToolSchema
	alias := &toolSchemaAlias{}

	if err := json.Unmarshal(data, alias); err != nil {
		return err
	}

	*t = ToolSchema(*alias)

	// If InputSchema is empty and Name/Description are also empty,
	// this might be a direct schema object
	if t.InputSchema == nil && t.Name == "" && t.Description == "" {
		var directSchema map[string]any
		if err := json.Unmarshal(data, &directSchema); err == nil {
			// Check if this looks like a JSON schema (has type field)
			if _, hasType := directSchema["type"]; hasType {
				t.InputSchema = directSchema
			}
		}
	}

	return nil
}

type ToolResult struct {
	Output           []ContentBlock `json:"output"`
	StructuredOutput any            `json:"structured_output,omitempty"`
	Usage            *Usage         `json:"usage,omitempty"`
	Metadata         ToolMetadata   `json:"metadata"`
	Variables        map[string]any `json:"variables,omitempty"`
	Wait             *AgentWait     `json:"wait,omitempty"`
	Final            bool           `json:"final,omitempty"`
	Error            error          `json:"-"`
	ErrorStr         string         `json:"error,omitempty"`
}

type ToolMetadata struct {
	Title     string             `json:"title"`
	Kind      ToolKind           `json:"kind"`
	Locations []ToolCallLocation `json:"locations,omitempty"`
	Content   []ToolCallContent  `json:"content,omitempty"`
}

// JSSummaryContextKey is used to thread the js_execute summary through the
// Go context so nested tool calls (and child-agent wrappers) can include it
// in their display metadata.
type jsSummaryCtxKey string

const JSSummaryContextKey jsSummaryCtxKey = "js_execute_summary"

func Truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "... (truncated)"
	}
	return s
}

type PermissionOptionKind string

const (
	PermissionOptionKindAllowOnce    PermissionOptionKind = "allow_once"
	PermissionOptionKindAllowAlways  PermissionOptionKind = "allow_always"
	PermissionOptionKindRejectOnce   PermissionOptionKind = "reject_once"
	PermissionOptionKindRejectAlways PermissionOptionKind = "reject_always"
)

type PermissionOptionID string

type PermissionScope string

const (
	PermissionScopeOnce    PermissionScope = "once"
	PermissionScopeSession PermissionScope = "session"
	PermissionScopeGlobal  PermissionScope = "global"
)

type PermissionTarget string

const (
	PermissionTargetTool    PermissionTarget = "tool"
	PermissionTargetFile    PermissionTarget = "file"
	PermissionTargetFolder  PermissionTarget = "folder"
	PermissionTargetCommand PermissionTarget = "command"
)

type PermissionOption struct {
	ID       PermissionOptionID   `json:"id"`
	Name     string               `json:"name"`
	Kind     PermissionOptionKind `json:"kind"`
	Scope    PermissionScope      `json:"scope,omitempty"`
	Target   PermissionTarget     `json:"target,omitempty"`
	Grants   []PermissionGrant    `json:"grants,omitempty"`
	Metadata map[string]any       `json:"metadata,omitempty"`
}

type PermissionGrant struct {
	Field     string              `json:"field,omitempty"`
	MatchMode PermissionMatchMode `json:"match_mode,omitempty"`
	Transform string              `json:"transform,omitempty"`
	Segments  int                 `json:"segments,omitempty"`
}

func AllowOnceOption() PermissionOption {
	return PermissionOption{ID: "allow_once", Name: "Allow once", Kind: PermissionOptionKindAllowOnce, Scope: PermissionScopeOnce}
}

func RejectOnceOption() PermissionOption {
	return PermissionOption{ID: "reject_once", Name: "Reject once", Kind: PermissionOptionKindRejectOnce, Scope: PermissionScopeOnce}
}

func AllowAlwaysOption(id, name string, scope PermissionScope, target PermissionTarget, grants ...PermissionGrant) PermissionOption {
	return PermissionOption{ID: PermissionOptionID(id), Name: name, Kind: PermissionOptionKindAllowAlways, Scope: scope, Target: target, Grants: grants}
}

func RejectAlwaysOption(id, name string, scope PermissionScope, target PermissionTarget, grants ...PermissionGrant) PermissionOption {
	return PermissionOption{ID: PermissionOptionID(id), Name: name, Kind: PermissionOptionKindRejectAlways, Scope: scope, Target: target, Grants: grants}
}

type ToolKind string

const (
	ToolKindRead       ToolKind = "read"
	ToolKindEdit       ToolKind = "edit"
	ToolKindDelete     ToolKind = "delete"
	ToolKindMove       ToolKind = "move"
	ToolKindSearch     ToolKind = "search"
	ToolKindExecute    ToolKind = "execute"
	ToolKindThink      ToolKind = "think"
	ToolKindFetch      ToolKind = "fetch"
	ToolKindSwitchMode ToolKind = "switch_mode"
	ToolKindOther      ToolKind = "other"
)

type ToolCallLocation struct {
	Path     string         `json:"path"`
	Line     *int           `json:"line,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type ToolCallDiff struct {
	Path     string         `json:"path"`
	NewText  string         `json:"new_text"`
	OldText  *string        `json:"old_text,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type ToolCallTerminal struct {
	ID string `json:"id"`
}

type ToolCallContentBlock struct {
	Block ContentBlock `json:"block"`
}

type ToolCallContent struct {
	Content  *ToolCallContentBlock `json:"content,omitempty"`
	Diff     *ToolCallDiff         `json:"diff,omitempty"`
	Terminal *ToolCallTerminal     `json:"terminal,omitempty"`
}

func NewToolCallContentBlock(block ContentBlock) ToolCallContent {
	return ToolCallContent{
		Content: &ToolCallContentBlock{Block: block},
	}
}

func NewToolCallDiff(path string, newText string, oldText *string) ToolCallContent {
	return ToolCallContent{
		Diff: &ToolCallDiff{
			Path:    path,
			NewText: newText,
			OldText: oldText,
		},
	}
}

func NewToolCallTerminal(id string) ToolCallContent {
	return ToolCallContent{
		Terminal: &ToolCallTerminal{ID: id},
	}
}
