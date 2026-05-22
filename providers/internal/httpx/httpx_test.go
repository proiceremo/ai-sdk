package httpx

import (
	"strings"
	"testing"
	"time"
)

func TestHTTPStatusErrorErrorOmitsSensitiveRequestDetails(t *testing.T) {
	t.Parallel()

	err := &HTTPStatusError{
		StatusCode: 400,
		URL:        "https://example.com/v1/messages",
		Method:     "POST",
		Body:       `{"error":{"message":"bad request"}}`,
		Request: &DebugRequest{
			Method: "POST",
			URL:    "https://example.com/v1/messages",
			Headers: map[string]string{
				"x-api-key":      "secret-key",
				"Authorization":  "Bearer secret-token",
				"x-goog-api-key": "secret-goog-key",
			},
			Body: `{"api_key":"inline-secret"}`,
			Time: time.Unix(0, 0).UTC(),
		},
	}

	got := err.Error()
	if strings.Contains(got, "secret-key") || strings.Contains(got, "secret-token") || strings.Contains(got, "inline-secret") {
		t.Fatalf("Error leaked sensitive data: %q", got)
	}
	if strings.Contains(got, "Request Details") || strings.Contains(got, "Headers:") || strings.Contains(got, "\nBody:") {
		t.Fatalf("Error unexpectedly included request details: %q", got)
	}
}

func TestHTTPStatusErrorDetailedErrorRedactsSensitiveHeaders(t *testing.T) {
	t.Parallel()

	err := &HTTPStatusError{
		StatusCode: 400,
		URL:        "https://example.com/v1/messages",
		Method:     "POST",
		Body:       `{"error":{"message":"bad request"}}`,
		Request: &DebugRequest{
			Method: "POST",
			URL:    "https://example.com/v1/messages",
			Headers: map[string]string{
				"x-api-key":      "secret-key",
				"Authorization":  "Bearer secret-token",
				"x-goog-api-key": "secret-goog-key",
			},
			Body: `{"api_key":"inline-secret"}`,
			Time: time.Unix(0, 0).UTC(),
		},
	}

	got := err.DetailedError()
	if !strings.Contains(got, "Request Details") {
		t.Fatalf("DetailedError omitted request details: %q", got)
	}
	if strings.Contains(got, "secret-key") || strings.Contains(got, "secret-token") || strings.Contains(got, "inline-secret") {
		t.Fatalf("DetailedError leaked sensitive data: %q", got)
	}
	if !strings.Contains(got, "[REDACTED]") {
		t.Fatalf("DetailedError did not redact sensitive values: %q", got)
	}
}
