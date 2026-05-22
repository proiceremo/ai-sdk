package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMessageContentValidateRejectsMissingDocumentPayload(t *testing.T) {
	content := MessageContent{{Type: ContentBlockTypeDocument}}
	if err := content.Validate(); err == nil {
		t.Fatal("expected document payload validation error")
	}
}

func TestMessageContentValidateRejectsInvalidToolUseJSON(t *testing.T) {
	content := MessageContent{{
		Type: ContentBlockTypeToolUse,
		ToolUse: &ToolUse{
			Name:  "fs_edit",
			Input: json.RawMessage("{\"path\":\"/tmp/x\",\"edits\":[{\"old_string\":\"a\tb\",\"new_string\":\"c\"}]}"),
		},
	}}
	if err := content.Validate(); err == nil {
		t.Fatal("expected invalid tool use json validation error")
	}
}

func TestMessageContentTextAndStringIncludeDocumentText(t *testing.T) {
	content := MessageContent{
		NewTextContentBlock("intro"),
		{
			Type: ContentBlockTypeDocument,
			Document: &DocumentSource{
				Type:    DocumentSourceTypeText,
				Title:   "Doc Title",
				Context: "Doc Context",
				Text:    "Doc Body",
			},
		},
		NewToolResultContentBlock(ToolOutput{
			ToolUseID: "tool-1",
			Output: MessageContent{
				NewTextContentBlock("tool output"),
			},
		}),
	}

	text := content.Text()
	for _, want := range []string{"intro", "Doc Title", "Doc Context", "Doc Body", "tool output"} {
		if !strings.Contains(text, want) {
			t.Fatalf("expected %q in text helper output %q", want, text)
		}
	}

	stringValue := content.ToString()
	if !strings.Contains(stringValue, "[DOCUMENT] Doc Title\nDoc Context\nDoc Body") {
		t.Fatalf("expected document text in ToString output, got %q", stringValue)
	}
}

func TestStripEphemeralMessagesRemovesEphemeralBlocks(t *testing.T) {
	messages := []Message{{
		Role: MessageRoleUser,
		Content: MessageContent{
			NewTextContentBlock("keep"),
			NewEphemeralTextContentBlock("drop"),
			NewToolResultContentBlock(ToolOutput{
				ToolUseID: "call_1",
				Output: MessageContent{
					NewTextContentBlock("result"),
					NewEphemeralTextContentBlock("screenshot"),
				},
				Metadata: &ToolMetadata{
					Content: []ToolCallContent{
						NewToolCallContentBlock(NewTextContentBlock("visible")),
						NewToolCallContentBlock(EphemeralContentBlock(NewImageContentBlockFromBase64("abc", "image/png"))),
					},
				},
			}),
		},
	}}

	stripped := StripEphemeralMessages(messages)
	if len(stripped) != 1 {
		t.Fatalf("expected one message after strip, got %d", len(stripped))
	}
	if got := stripped[0].Content.Text(); got != "keepresult" {
		t.Fatalf("unexpected stripped text: %q", got)
	}
	output := stripped[0].Content[1].ToolOutput
	if output == nil || output.Metadata == nil {
		t.Fatalf("expected tool metadata to be preserved")
	}
	if got := len(output.Metadata.Content); got != 1 {
		t.Fatalf("expected ephemeral metadata content to be stripped, got %d entries", got)
	}
}

func TestModelConfigFilterContentKeepsBase64DocumentsForTextModels(t *testing.T) {
	model := ModelConfig{InputModalities: []Modality{ModalityText}}
	content := MessageContent{
		NewDocumentContentBlockFromBase64("ZmFrZQ==", "application/pdf"),
	}

	filtered := model.FilterContent(content)
	if len(filtered) != 1 {
		t.Fatalf("expected base64 document to be preserved, got %#v", filtered)
	}
}

func TestUsageMergeRecomputesLedgerSummaries(t *testing.T) {
	base := NewUsage(UsageOperationCompletion, TokenUsage{
		InputTokens:        10,
		ToolUseInputTokens: 2,
		InputTokensDetails: &UsageTokenDetails{TextTokens: 7},
		ToolUseInputTokensDetails: &UsageTokenDetails{
			DocumentTokens: 2,
		},
	})
	base.Merge(NewUsage(UsageOperationTool, TokenUsage{
		InputTokens:        5,
		ToolUseInputTokens: 3,
		InputTokensDetails: &UsageTokenDetails{
			ImageTokens: 5,
		},
		ToolUseInputTokensDetails: &UsageTokenDetails{
			DocumentTokens: 1,
			TextTokens:     2,
		},
	}))

	if base.Totals.InputTokens != 15 || base.Totals.ToolUseInputTokens != 5 {
		t.Fatalf("unexpected token totals after merge: %#v", base)
	}
	if base.Totals.InputTokensDetails == nil || base.Totals.InputTokensDetails.TextTokens != 7 || base.Totals.InputTokensDetails.ImageTokens != 5 {
		t.Fatalf("unexpected input token details after merge: %#v", base.Totals.InputTokensDetails)
	}
	if base.Totals.ToolUseInputTokensDetails == nil || base.Totals.ToolUseInputTokensDetails.DocumentTokens != 3 || base.Totals.ToolUseInputTokensDetails.TextTokens != 2 {
		t.Fatalf("unexpected tool-use token details after merge: %#v", base.Totals.ToolUseInputTokensDetails)
	}
}
