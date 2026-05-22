package llm

import (
	"encoding/json"
	"fmt"

	"github.com/invopop/jsonschema"
)

type JSONSchema = map[string]any

func ReflectJSONSchema(value any) JSONSchema {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	return schemaToJSONSchema(reflector.Reflect(value))
}

func DefaultToolInputSchema() JSONSchema {
	return JSONSchema{
		"type":                 "object",
		"properties":           map[string]any{},
		"additionalProperties": false,
	}
}

func DefaultFinishInputSchema() JSONSchema {
	return JSONSchema{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"summary"},
		"properties": map[string]any{
			"summary": map[string]any{
				"type":        "string",
				"description": "Concise final message rendered to the user.",
				"minLength":   1,
			},
			"output": map[string]any{
				"description": "Optional structured final result for callers and automation.",
			},
		},
	}
}

func NormalizeToolInputSchema(schema JSONSchema) (JSONSchema, error) {
	if len(schema) == 0 {
		return cloneJSONSchema(DefaultToolInputSchema()), nil
	}
	normalized := cloneJSONSchema(schema)
	schemaType, _ := normalized["type"].(string)
	if schemaType == "" {
		normalized["type"] = "object"
		schemaType = "object"
	}
	if schemaType != "object" {
		return nil, fmt.Errorf("tool input schema must be an object schema, got %q", schemaType)
	}
	if _, ok := normalized["properties"]; !ok {
		normalized["properties"] = map[string]any{}
	}
	normalizeObjectSchemas(normalized)
	return normalized, nil
}

func cloneJSONSchema(schema JSONSchema) JSONSchema {
	data, err := json.Marshal(schema)
	if err != nil {
		return JSONSchema{}
	}
	var out JSONSchema
	if err := json.Unmarshal(data, &out); err != nil {
		return JSONSchema{}
	}
	return out
}

func normalizeObjectSchemas(value any) {
	switch typed := value.(type) {
	case map[string]any:
		if typed["type"] == "object" {
			if _, ok := typed["properties"]; !ok {
				typed["properties"] = map[string]any{}
			}
			if _, ok := typed["additionalProperties"]; !ok {
				typed["additionalProperties"] = false
			}
		}
		for _, child := range typed {
			normalizeObjectSchemas(child)
		}
	case []any:
		for _, child := range typed {
			normalizeObjectSchemas(child)
		}
	}
}

func schemaToJSONSchema(schema *jsonschema.Schema) JSONSchema {
	if schema == nil {
		return nil
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return nil
	}
	var result JSONSchema
	if err := json.Unmarshal(data, &result); err != nil {
		return nil
	}
	normalized, err := NormalizeToolInputSchema(result)
	if err != nil {
		return result
	}
	return normalized
}
