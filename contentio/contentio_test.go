package contentio

import (
	"net/http"
	"net/http/httptest"
	"testing"

	llm "ai-sdk"
)

func TestURLToDataURIReturnsErrorOnHTTPFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "missing", http.StatusNotFound)
	}))
	defer server.Close()

	if _, _, err := URLToDataURI(server.URL); err == nil {
		t.Fatal("expected URLToDataURI to fail on non-2xx responses")
	}
}

func TestNewContentBlockFromURIUsesDetectedMIMEType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("\x89PNG\r\n\x1a\npayload"))
	}))
	defer server.Close()

	block, err := NewContentBlockFromURI(server.URL, nil)
	if err != nil {
		t.Fatalf("NewContentBlockFromURI: %v", err)
	}
	if block.Type != llm.ContentBlockTypeImage || block.Image == nil {
		t.Fatalf("expected image block, got %#v", block)
	}
	if block.Image.Type != llm.ImageSourceTypeBase64 {
		t.Fatalf("expected embedded image data, got %#v", block.Image)
	}
	if block.Image.MediaType != "image/png" {
		t.Fatalf("media type = %q, want image/png", block.Image.MediaType)
	}
}

func TestNewContentBlockFromURIWithExplicitMIMEKeepsRemoteURL(t *testing.T) {
	mimeType := "image/png"
	block, err := NewContentBlockFromURI("https://example.com/image.png", &mimeType)
	if err != nil {
		t.Fatalf("NewContentBlockFromURI: %v", err)
	}
	if block.Type != llm.ContentBlockTypeImage || block.Image == nil {
		t.Fatalf("expected image block, got %#v", block)
	}
	if block.Image.Type != llm.ImageSourceTypeURL || block.Image.URL != "https://example.com/image.png" {
		t.Fatalf("expected remote image URL, got %#v", block.Image)
	}
}
