package tools

import (
	"encoding/json"
	"fmt"
	"sync"

	llm "github.com/proiceremo/ai-sdk"
)

type GenericTool[T any] struct {
	name                string
	description         string
	strict              bool
	executor            func(ctx llm.ToolContext, input T) llm.ToolResult
	permissionExtractor func(ctx llm.ToolContext, input T) ([]llm.PermissionGuard, error)
	outputSchema        llm.JSONSchema

	schemaOnce   sync.Once
	cachedSchema llm.JSONSchema
}

type FinishToolOptions struct {
	Name        string
	Description string
}

func NewGenericTool[T any](
	name, description string,
	executor func(ctx llm.ToolContext, input T) llm.ToolResult,
) *GenericTool[T] {
	return &GenericTool[T]{
		name:        name,
		description: description,
		executor:    executor,
	}
}

func NewFinishTool[T any](opts FinishToolOptions) *GenericTool[T] {
	name := opts.Name
	if name == "" {
		name = "finish"
	}
	description := opts.Description
	if description == "" {
		description = "Finish the current agent turn with the final answer or structured output."
	}
	return NewGenericTool(name, description, func(ctx llm.ToolContext, input T) llm.ToolResult {
		_ = ctx
		output, err := json.Marshal(input)
		if err != nil {
			return ErrorResult(fmt.Errorf("failed to serialize finish output: %w", err))
		}
		return llm.ToolResult{
			Output:           []llm.ContentBlock{llm.NewTextContentBlock(string(output))},
			StructuredOutput: input,
			Final:            true,
			Metadata: llm.ToolMetadata{
				Title: "Finish",
				Kind:  llm.ToolKindOther,
			},
		}
	}).WithStrict(true)
}

func (t *GenericTool[T]) WithPermissionExtractor(extractor func(ctx llm.ToolContext, input T) ([]llm.PermissionGuard, error)) *GenericTool[T] {
	t.permissionExtractor = extractor
	return t
}

func (t *GenericTool[T]) WithPermission(specs ...PermissionExtractor) *GenericTool[T] {
	return t.WithPermissionExtractor(func(ctx llm.ToolContext, input T) ([]llm.PermissionGuard, error) {
		inputJSON, err := json.Marshal(input)
		if err != nil {
			return nil, err
		}
		var inputMap map[string]any
		if err := json.Unmarshal(inputJSON, &inputMap); err != nil {
			return nil, err
		}
		return ExtractPermissions(ctx, inputMap, specs...)
	})
}

func (t *GenericTool[T]) WithStrict(strict bool) *GenericTool[T] {
	t.strict = strict
	return t
}

func (t *GenericTool[T]) WithOutputSchema(schema llm.JSONSchema) *GenericTool[T] {
	t.outputSchema = schema
	return t
}

func (t *GenericTool[T]) GetPermissions(ctx llm.ToolContext, input map[string]any) ([]llm.PermissionGuard, error) {
	if t.permissionExtractor == nil {
		return nil, nil
	}
	params, err := t.ParseParameters(input)
	if err != nil {
		return nil, fmt.Errorf("failed to parse parameters for permission extraction: %w", err)
	}
	return t.permissionExtractor(ctx, *params)
}

func (t *GenericTool[T]) Schema() llm.ToolSchema {
	t.schemaOnce.Do(func() {
		var zero T
		t.cachedSchema = ReflectValue(zero)
	})

	return llm.ToolSchema{
		Name:         t.name,
		Description:  t.description,
		InputSchema:  t.cachedSchema,
		OutputSchema: t.outputSchema,
		Strict:       t.strict,
	}
}

func (t *GenericTool[T]) Execute(ctx llm.ToolContext, input map[string]any) llm.ToolResult {
	params, err := t.ParseParameters(input)
	if err != nil {
		return ErrorResult(fmt.Errorf("invalid input for tool %s: %w", t.name, err))
	}

	return t.executor(ctx, *params)
}

func (t *GenericTool[T]) ParseParameters(input map[string]any) (*T, error) {
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	var params T
	if err := json.Unmarshal(inputJSON, &params); err != nil {
		return nil, fmt.Errorf("invalid input for tool %s: %w", t.name, err)
	}

	return &params, nil
}

func ReflectSchema[T any]() map[string]any {
	var zero T
	return ReflectValue(zero)
}

func ReflectValue(value any) map[string]any {
	return llm.ReflectJSONSchema(value)
}
