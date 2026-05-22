package oauthx

import "testing"

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
