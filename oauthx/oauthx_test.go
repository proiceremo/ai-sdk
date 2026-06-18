package oauthx

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestCodexConfigUsesRegisteredRedirectAndAuthParams(t *testing.T) {
	cfg := CodexConfig("openai-codex")
	if cfg.RedirectURL != "http://localhost:1455/auth/callback" {
		t.Fatalf("redirect = %q", cfg.RedirectURL)
	}
	if cfg.AuthParams["codex_cli_simplified_flow"] != "true" || cfg.AuthParams["originator"] != "proagent" {
		t.Fatalf("auth params = %#v", cfg.AuthParams)
	}
}

func TestAnthropicConfigUsesClaudeSubscriptionOAuth(t *testing.T) {
	cfg := AnthropicConfig("anthropic-oauth")
	if cfg.Type != "anthropic" {
		t.Fatalf("type = %q", cfg.Type)
	}
	if cfg.RedirectURL != "http://localhost:53692/callback" {
		t.Fatalf("redirect = %q", cfg.RedirectURL)
	}
	if cfg.ClientID != "9d1c250a-e61b-44d9-88ed-5944d1962f5e" {
		t.Fatalf("client id = %q", cfg.ClientID)
	}
	if cfg.CacheKey != "anthropic-oauth" {
		t.Fatalf("cache key = %q", cfg.CacheKey)
	}
}

func TestLoadFreshProviderCredentialsRefreshesAnthropicAndPersistsRotation(t *testing.T) {
	store := FileStore{Dir: t.TempDir()}
	if err := store.Save("anthropic-oauth", Credentials{
		AccessToken:  "old-access",
		RefreshToken: "old-refresh",
		TokenType:    "Bearer",
		Expiry:       time.Now().Add(-time.Hour),
	}); err != nil {
		t.Fatal(err)
	}
	var requestBody map[string]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("content-type") != "application/json" {
			t.Errorf("content-type = %q", r.Header.Get("content-type"))
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Error(err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "new-access",
			"refresh_token": "new-refresh",
			"token_type":    "Bearer",
			"expires_in":    3600,
		})
	}))
	defer server.Close()

	cfg := AnthropicConfig("anthropic-oauth")
	cfg.TokenURL = server.URL
	creds, err := loadFreshProviderCredentials(context.Background(), store, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if requestBody["grant_type"] != "refresh_token" || requestBody["refresh_token"] != "old-refresh" || requestBody["scope"] != "" {
		t.Fatalf("unexpected refresh body: %#v", requestBody)
	}
	if creds.AccessToken != "new-access" || creds.RefreshToken != "new-refresh" {
		t.Fatalf("unexpected refreshed creds: %+v", creds)
	}
	persisted, err := store.Load("anthropic-oauth")
	if err != nil {
		t.Fatal(err)
	}
	if persisted.AccessToken != "new-access" || persisted.RefreshToken != "new-refresh" {
		t.Fatalf("rotated creds not persisted: %+v", persisted)
	}
}

func TestLoadFreshProviderCredentialsSkipsFreshToken(t *testing.T) {
	store := FileStore{Dir: t.TempDir()}
	if err := store.Save("anthropic-oauth", Credentials{
		AccessToken:  "fresh-access",
		RefreshToken: "fresh-refresh",
		TokenType:    "Bearer",
		Expiry:       time.Now().Add(time.Hour),
	}); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("fresh token should not be refreshed")
	}))
	defer server.Close()
	cfg := AnthropicConfig("anthropic-oauth")
	cfg.TokenURL = server.URL
	creds, err := loadFreshProviderCredentials(context.Background(), store, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if creds.AccessToken != "fresh-access" {
		t.Fatalf("unexpected creds: %+v", creds)
	}
}

func TestClaudeCodeOAuthTokenUsesAnthropicOAuthCache(t *testing.T) {
	store := FileStore{Dir: t.TempDir()}
	if err := store.Save("anthropic-oauth", Credentials{
		AccessToken:  "anthropic-access",
		RefreshToken: "anthropic-refresh",
		TokenType:    "Bearer",
		Expiry:       time.Now().Add(time.Hour),
	}); err != nil {
		t.Fatal(err)
	}
	token, err := ClaudeCodeOAuthToken(context.Background(), store)
	if err != nil {
		t.Fatal(err)
	}
	if token != "anthropic-access" {
		t.Fatalf("token = %q", token)
	}
}

func TestRefreshCodexCredentialsUsesFormEncoding(t *testing.T) {
	var form url.Values
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("content-type"); !strings.Contains(got, "application/x-www-form-urlencoded") {
			t.Errorf("content-type = %q", got)
		}
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}
		form = r.Form
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "new-codex-access",
			"refresh_token": "new-codex-refresh",
			"expires_in":    3600,
		})
	}))
	defer server.Close()
	cfg := CodexConfig("openai-codex")
	cfg.TokenURL = server.URL
	creds, err := refreshCodexCredentials(context.Background(), cfg, "old-codex-refresh")
	if err != nil {
		t.Fatal(err)
	}
	if form.Get("grant_type") != "refresh_token" || form.Get("refresh_token") != "old-codex-refresh" || form.Get("client_id") != cfg.ClientID {
		t.Fatalf("unexpected form: %v", form)
	}
	if creds.AccessToken != "new-codex-access" || creds.RefreshToken != "new-codex-refresh" {
		t.Fatalf("unexpected codex creds: %+v", creds)
	}
}
