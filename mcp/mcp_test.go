package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"ai-sdk/oauthx"
	"golang.org/x/oauth2"
)

type fakeServerStore struct {
	servers []ServerConfig
}

func (s fakeServerStore) ListServers(ctx context.Context, scopeID string) ([]ServerConfig, error) {
	return s.servers, nil
}

func TestManagerListServersPaginatesAndFilters(t *testing.T) {
	manager := NewManager(fakeServerStore{servers: []ServerConfig{
		{Name: "alpha", URL: "https://alpha.example/mcp"},
		{Name: "beta", URL: "https://beta.example/mcp"},
		{Name: "gamma", URL: "https://gamma.example/mcp"},
	}})
	first, err := manager.Execute("sess", Input{Mode: "servers", Limit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Servers) != 2 || first.NextCursor == "" {
		t.Fatalf("first page = %#v", first)
	}
	second, err := manager.Execute("sess", Input{Mode: "servers", Cursor: first.NextCursor, Limit: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(second.Servers) != 1 || second.Servers[0].Name != "gamma" || second.NextCursor != "" {
		t.Fatalf("second page = %#v", second)
	}
	filtered, err := manager.Execute("sess", Input{Mode: "servers", Query: "bet"})
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered.Servers) != 1 || filtered.Servers[0].Name != "beta" {
		t.Fatalf("filtered = %#v", filtered)
	}
}

func TestServerConfigTransportTypeNormalizesKnownTransports(t *testing.T) {
	cases := map[string]string{
		"":                "streamable_http",
		"http":            "streamable_http",
		"streamable-http": "streamable_http",
		"streamable_http": "streamable_http",
		"sse":             "sse",
		"stdio":           "stdio",
	}
	for input, want := range cases {
		cfg := ServerConfig{Type: input, URL: "https://example.com/mcp"}
		if input == "stdio" {
			cfg.URL = ""
			cfg.Command = "/bin/mcp"
		}
		if got := cfg.transportType(); got != want {
			t.Fatalf("transportType(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestServerConfigOAuthAutoMode(t *testing.T) {
	if !((ServerConfig{URL: "https://example.com/mcp"}).ShouldUseOAuth()) {
		t.Fatal("streamable HTTP without Authorization header should allow OAuth discovery")
	}
	if (ServerConfig{
		URL:     "https://example.com/mcp",
		Headers: []Header{{Name: "Authorization", Value: "Bearer token"}},
	}).ShouldUseOAuth() {
		t.Fatal("Authorization header should disable implicit OAuth discovery")
	}
	if !((ServerConfig{
		URL:     "https://example.com/mcp",
		Headers: []Header{{Name: "Authorization", Value: "Bearer token"}},
		Auth:    AuthConfig{Type: "oauth"},
	}).ShouldUseOAuth()) {
		t.Fatal("explicit OAuth config should override static Authorization header")
	}
	if (ServerConfig{URL: "https://example.com/mcp", Auth: AuthConfig{Type: "disabled"}}).ShouldUseOAuth() {
		t.Fatal("disabled auth should not use OAuth")
	}
}

func TestPersistentOAuthHandler_TokenSource_Refresh(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcp-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		if r.Method != http.MethodPost {
			t.Errorf("expected POST request, got %s", r.Method)
		}
		if err := r.ParseForm(); err != nil {
			t.Fatal(err)
		}
		if r.Form.Get("grant_type") != "refresh_token" {
			t.Errorf("expected grant_type=refresh_token, got %s", r.Form.Get("grant_type"))
		}
		if r.Form.Get("refresh_token") != "old-refresh-token" {
			t.Errorf("expected refresh_token=old-refresh-token, got %s", r.Form.Get("refresh_token"))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "new-access-token",
			"refresh_token": "new-refresh-token",
			"expires_in":    3600,
			"token_type":    "Bearer",
		})
	}))
	defer server.Close()

	// 1. Create an expired token in the temporary directory under the
	// unified credentials root (PRO_HOME/oauth/mcp/<name>.json — the
	// MCPStore path. Filename is sanitised by oauthx, but "test-server"
	// is already safe.
	tokenFile := filepath.Join(tmpDir, "oauth", "mcp", "test-server.json")
	if err := os.MkdirAll(filepath.Dir(tokenFile), 0o700); err != nil {
		t.Fatal(err)
	}
	oldToken := &oauth2.Token{
		AccessToken:  "old-access-token",
		RefreshToken: "old-refresh-token",
		Expiry:       time.Now().Add(-1 * time.Hour), // Expired 1 hour ago
	}
	tokenData, err := json.Marshal(oldToken)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tokenFile, tokenData, 0o600); err != nil {
		t.Fatal(err)
	}

	// 2. Set up the handler with the temporary token path and OAuth config.
	h := &persistentOAuthHandler{
		cfg: ServerConfig{
			Name: "test-server",
			Auth: AuthConfig{
				TokenURL:     server.URL,
				ClientID:     "test-client",
				ClientSecret: "test-secret",
			},
		},
		inner:  &mockOAuthHandler{},
		client: http.DefaultClient,
	}

	t.Setenv("PRO_HOME", tmpDir)

	// Call TokenSource(context.Background())
	ts, err := h.TokenSource(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if ts == nil {
		t.Fatal("expected non-nil TokenSource")
	}

	// Fetch token, which triggers the refresh.
	tok, err := ts.Token()
	if err != nil {
		t.Fatal(err)
	}

	if !called {
		t.Fatal("expected mock refresh token endpoint to be called")
	}

	if tok.AccessToken != "new-access-token" {
		t.Errorf("expected access token 'new-access-token', got '%s'", tok.AccessToken)
	}

	if tok.RefreshToken != "new-refresh-token" {
		t.Errorf("expected refresh token 'new-refresh-token', got '%s'", tok.RefreshToken)
	}

	// Verify the new token was persisted to disk via oauthx.MCPStore.
	persisted, err := oauthx.MCPStore().Load(h.storeKey())
	if err != nil {
		t.Fatal(err)
	}
	if persisted.AccessToken == "" {
		t.Fatal("expected persisted token to exist")
	}
	if persisted.AccessToken != "new-access-token" {
		t.Errorf("expected persisted access token 'new-access-token', got '%s'", persisted.AccessToken)
	}
}

type mockOAuthHandler struct{}

func (m *mockOAuthHandler) TokenSource(context.Context) (oauth2.TokenSource, error) {
	return nil, nil
}

func (m *mockOAuthHandler) Authorize(context.Context, *http.Request, *http.Response) error {
	return nil
}
