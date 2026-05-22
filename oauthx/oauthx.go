package oauthx

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/clientcredentials"
)

type Config struct {
	Type         string            `json:"type,omitempty"`
	ProviderID   string            `json:"provider_id,omitempty"`
	TokenURL     string            `json:"token_url,omitempty"`
	AuthURL      string            `json:"auth_url,omitempty"`
	ClientID     string            `json:"client_id,omitempty"`
	ClientSecret string            `json:"client_secret,omitempty"`
	Scopes       []string          `json:"scopes,omitempty"`
	RedirectURL  string            `json:"redirect_url,omitempty"`
	CacheKey     string            `json:"cache_key,omitempty"`
	AuthParams   map[string]string `json:"auth_params,omitempty"`
}

type Credentials struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	TokenType    string    `json:"token_type,omitempty"`
	Expiry       time.Time `json:"expiry,omitempty"`
	AccountID    string    `json:"account_id,omitempty"`
}

func (c Credentials) Token() *oauth2.Token {
	return &oauth2.Token{
		AccessToken:  c.AccessToken,
		RefreshToken: c.RefreshToken,
		TokenType:    firstNonEmpty(c.TokenType, "Bearer"),
		Expiry:       c.Expiry,
	}
}

func CredentialsFromToken(token *oauth2.Token) Credentials {
	if token == nil {
		return Credentials{}
	}
	return Credentials{
		AccessToken:  token.AccessToken,
		RefreshToken: token.RefreshToken,
		TokenType:    token.TokenType,
		Expiry:       token.Expiry,
	}
}

type FileStore struct {
	Dir string
}

// Root resolves the canonical proagent credentials root. PRO_HOME wins when
// set (containerised runs, bench-style sandboxes); otherwise we fall back to
// ~/.pro. Empty string means we have no place to write — callers treat that
// as "store disabled".
func Root() string {
	if base := strings.TrimSpace(os.Getenv("PRO_HOME")); base != "" {
		return base
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ""
	}
	return filepath.Join(home, ".pro")
}

// ProvidersStore holds OAuth credentials for ai-sdk model providers
// (openai-codex, anthropic-oauth, claude-code, …). Keyed by provider id.
func ProvidersStore() FileStore {
	root := Root()
	if root == "" {
		return FileStore{}
	}
	return FileStore{Dir: filepath.Join(root, "oauth", "providers")}
}

// MCPStore holds OAuth credentials for MCP servers. Keyed by the server's
// safe name. Lives next to ProvidersStore under a single oauth/ root so
// there's one directory to seed, back up, and reason about — and so server
// names can't collide with provider ids.
func MCPStore() FileStore {
	root := Root()
	if root == "" {
		return FileStore{}
	}
	return FileStore{Dir: filepath.Join(root, "oauth", "mcp")}
}

func (s FileStore) Load(key string) (Credentials, error) {
	path := s.path(key)
	if path == "" {
		return Credentials{}, os.ErrNotExist
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Credentials{}, err
	}
	var creds Credentials
	if err := json.Unmarshal(data, &creds); err != nil {
		return Credentials{}, err
	}
	return creds, nil
}

func (s FileStore) Save(key string, creds Credentials) error {
	path := s.path(key)
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(creds, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func (s FileStore) path(key string) string {
	if s.Dir == "" || strings.TrimSpace(key) == "" {
		return ""
	}
	return filepath.Join(s.Dir, safeFilename(key)+".json")
}

// Delete removes the persisted credentials for key. Returns nil if the
// entry didn't exist — callers shouldn't have to special-case "logout
// when never logged in".
func (s FileStore) Delete(key string) error {
	path := s.path(key)
	if path == "" {
		return nil
	}
	err := os.Remove(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Has reports whether a credential entry exists on disk for key. Cheap
// status check that doesn't parse the file.
func (s FileStore) Has(key string) bool {
	path := s.path(key)
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

// LoginOptions tunes the interactive login flow. OnAuthURL, when set,
// receives the authorization URL — the caller is then responsible for
// presenting it to the user (e.g. an ACP slash command streams it to the
// client). When nil we fall back to printing the URL to stderr and
// trying to open the user's default browser, which is the right default
// for CLI usage.
type LoginOptions struct {
	OnAuthURL func(url string)
}

func (o LoginOptions) present(authURL string) {
	if o.OnAuthURL != nil {
		o.OnAuthURL(authURL)
		return
	}
	fmt.Fprintf(os.Stderr, "Open this URL to authenticate:\n%s\n", authURL)
	_ = openBrowser(authURL)
}

func TokenSource(ctx context.Context, cfg Config) (oauth2.TokenSource, error) {
	cfg.Type = normalizeType(cfg.Type)
	switch cfg.Type {
	case "", "none", "disabled":
		return nil, nil
	case "client_credentials":
		if cfg.TokenURL == "" || cfg.ClientID == "" {
			return nil, fmt.Errorf("oauth client_credentials requires token_url and client_id")
		}
		cc := clientcredentials.Config{
			ClientID:     cfg.ClientID,
			ClientSecret: cfg.ClientSecret,
			TokenURL:     cfg.TokenURL,
			Scopes:       cfg.Scopes,
		}
		return cc.TokenSource(ctx), nil
	case "authorization_code", "oauth", "oauth2", "auto", "anthropic":
		creds, err := ProvidersStore().Load(cfg.Key())
		if err != nil || creds.AccessToken == "" {
			return nil, nil
		}
		if strings.EqualFold(cfg.ProviderID, "anthropic") || strings.EqualFold(cfg.Type, "anthropic") {
			return &persistingTokenSource{key: cfg.Key(), source: &anthropicTokenSource{ctx: ctx, cfg: cfg, creds: creds.Token()}}, nil
		}
		source := oauth2.StaticTokenSource(creds.Token())
		if creds.RefreshToken != "" && cfg.TokenURL != "" && cfg.ClientID != "" {
			oc := oauth2.Config{
				ClientID:     cfg.ClientID,
				ClientSecret: cfg.ClientSecret,
				Endpoint: oauth2.Endpoint{
					AuthURL:  cfg.AuthURL,
					TokenURL: cfg.TokenURL,
				},
				Scopes:      cfg.Scopes,
				RedirectURL: cfg.RedirectURL,
			}
			source = &persistingTokenSource{key: cfg.Key(), source: oc.TokenSource(ctx, creds.Token())}
		}
		return source, nil
	default:
		return nil, fmt.Errorf("unsupported oauth type %q", cfg.Type)
	}
}

func LoginAuthorizationCode(ctx context.Context, cfg Config, opts LoginOptions) (Credentials, error) {
	if cfg.AuthURL == "" || cfg.TokenURL == "" || cfg.ClientID == "" {
		return Credentials{}, fmt.Errorf("authorization_code login requires auth_url, token_url, and client_id")
	}
	callback, err := newCallbackServer(cfg.RedirectURL)
	if err != nil {
		return Credentials{}, err
	}
	defer callback.Close()
	cfg.RedirectURL = callback.RedirectURL()
	verifier, challenge, err := pkce()
	if err != nil {
		return Credentials{}, err
	}
	state, err := randomString(16)
	if err != nil {
		return Credentials{}, err
	}
	oc := oauth2.Config{
		ClientID:     cfg.ClientID,
		ClientSecret: cfg.ClientSecret,
		Endpoint: oauth2.Endpoint{
			AuthURL:  cfg.AuthURL,
			TokenURL: cfg.TokenURL,
		},
		Scopes:      cfg.Scopes,
		RedirectURL: cfg.RedirectURL,
	}
	options := []oauth2.AuthCodeOption{
		oauth2.SetAuthURLParam("code_challenge", challenge),
		oauth2.SetAuthURLParam("code_challenge_method", "S256"),
	}
	for key, value := range cfg.AuthParams {
		options = append(options, oauth2.SetAuthURLParam(key, value))
	}
	authURL := oc.AuthCodeURL(state, options...)
	opts.present(authURL)
	result, err := callback.Wait(ctx)
	if err != nil {
		return Credentials{}, err
	}
	if result.State != state {
		return Credentials{}, fmt.Errorf("oauth state mismatch")
	}
	token, err := oc.Exchange(ctx, result.Code, oauth2.SetAuthURLParam("code_verifier", verifier))
	if err != nil {
		return Credentials{}, err
	}
	creds := CredentialsFromToken(token)
	if err := ProvidersStore().Save(cfg.Key(), creds); err != nil {
		return Credentials{}, err
	}
	return creds, nil
}

func LoginAnthropic(ctx context.Context, providerID string, opts LoginOptions) (Credentials, error) {
	cfg := AnthropicConfig(providerID)
	callback, err := newCallbackServer(cfg.RedirectURL)
	if err != nil {
		return Credentials{}, err
	}
	defer callback.Close()
	cfg.RedirectURL = callback.RedirectURL()
	verifier, challenge, err := pkce()
	if err != nil {
		return Credentials{}, err
	}
	authURL, err := url.Parse(cfg.AuthURL)
	if err != nil {
		return Credentials{}, err
	}
	q := authURL.Query()
	q.Set("code", "true")
	q.Set("client_id", cfg.ClientID)
	q.Set("response_type", "code")
	q.Set("redirect_uri", cfg.RedirectURL)
	q.Set("scope", strings.Join(cfg.Scopes, " "))
	q.Set("code_challenge", challenge)
	q.Set("code_challenge_method", "S256")
	q.Set("state", verifier)
	authURL.RawQuery = q.Encode()
	opts.present(authURL.String())
	result, err := callback.Wait(ctx)
	if err != nil {
		return Credentials{}, err
	}
	if result.State != verifier {
		return Credentials{}, fmt.Errorf("oauth state mismatch")
	}
	creds, err := anthropicExchange(ctx, cfg, map[string]any{
		"grant_type":    "authorization_code",
		"client_id":     cfg.ClientID,
		"code":          result.Code,
		"state":         result.State,
		"redirect_uri":  cfg.RedirectURL,
		"code_verifier": verifier,
	})
	if err != nil {
		return Credentials{}, err
	}
	if err := ProvidersStore().Save(cfg.Key(), creds); err != nil {
		return Credentials{}, err
	}
	return creds, nil
}

func (c Config) Key() string {
	return firstNonEmpty(c.CacheKey, c.ProviderID, c.ClientID, "default")
}

type persistingTokenSource struct {
	key    string
	source oauth2.TokenSource
}

func (s *persistingTokenSource) Token() (*oauth2.Token, error) {
	token, err := s.source.Token()
	if err == nil && token != nil {
		_ = ProvidersStore().Save(s.key, CredentialsFromToken(token))
	}
	return token, err
}

type callbackServer struct {
	server       *http.Server
	ln           net.Listener
	redirectURL  string
	callbackPath string
	ch           chan callbackResult
}

type callbackResult struct {
	Code  string
	State string
	Err   error
}

func newCallbackServer(redirectURL string) (*callbackServer, error) {
	addr := "127.0.0.1:0"
	callbackPath := "/oauth/callback"
	scheme := "http"
	hostForRedirect := ""
	if redirectURL != "" {
		if parsed, err := url.Parse(redirectURL); err == nil && parsed.Host != "" {
			addr = parsed.Host
			scheme = parsed.Scheme
			callbackPath = parsed.Path
			hostForRedirect = parsed.Host
		}
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	if hostForRedirect == "" || strings.HasSuffix(hostForRedirect, ":0") {
		hostForRedirect = ln.Addr().String()
		if strings.HasPrefix(hostForRedirect, "127.0.0.1:") && strings.Contains(addr, "localhost") {
			hostForRedirect = "localhost:" + strings.TrimPrefix(hostForRedirect, "127.0.0.1:")
		}
	}
	if scheme == "" {
		scheme = "http"
	}
	cb := &callbackServer{ln: ln, redirectURL: scheme + "://" + hostForRedirect + callbackPath, callbackPath: callbackPath, ch: make(chan callbackResult, 1)}
	mux := http.NewServeMux()
	mux.HandleFunc("/", cb.handle)
	cb.server = &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = cb.server.Serve(ln) }()
	return cb, nil
}

func (c *callbackServer) RedirectURL() string {
	return c.redirectURL
}

func (c *callbackServer) Wait(ctx context.Context) (callbackResult, error) {
	select {
	case result := <-c.ch:
		return result, result.Err
	case <-ctx.Done():
		return callbackResult{}, ctx.Err()
	}
}

func (c *callbackServer) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return c.server.Shutdown(ctx)
}

func (c *callbackServer) handle(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	if c.callbackPath != "" && r.URL.Path != c.callbackPath {
		http.NotFound(w, r)
		return
	}
	result := callbackResult{Code: query.Get("code"), State: query.Get("state")}
	if e := query.Get("error"); e != "" {
		result.Err = fmt.Errorf("oauth error: %s", e)
	} else if result.Code == "" {
		result.Err = fmt.Errorf("oauth callback missing code")
	}
	select {
	case c.ch <- result:
	default:
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if result.Err != nil {
		http.Error(w, result.Err.Error(), http.StatusBadRequest)
		return
	}
	_, _ = io.WriteString(w, "<html><body><h1>Authorization complete</h1><p>You can close this tab.</p></body></html>")
}

func pkce() (verifier, challenge string, err error) {
	verifier, err = randomString(32)
	if err != nil {
		return "", "", err
	}
	sum := sha256.Sum256([]byte(verifier))
	challenge = base64.RawURLEncoding.EncodeToString(sum[:])
	return verifier, challenge, nil
}

func randomString(n int) (string, error) {
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

func openBrowser(u string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", u)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", u)
	default:
		cmd = exec.Command("xdg-open", u)
	}
	return cmd.Start()
}

func normalizeType(t string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(t)), "-", "_")
}

func safeFilename(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "default"
	}
	return b.String()
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func CodexConfig(providerID string) Config {
	return Config{
		Type:        "authorization_code",
		ProviderID:  providerID,
		AuthURL:     "https://auth.openai.com/oauth/authorize",
		TokenURL:    "https://auth.openai.com/oauth/token",
		ClientID:    "app_EMoamEEZ73f0CkXaXp7hrann",
		Scopes:      []string{"openid", "profile", "email", "offline_access"},
		RedirectURL: "http://localhost:1455/auth/callback",
		CacheKey:    providerID,
		AuthParams: map[string]string{
			"id_token_add_organizations": "true",
			"codex_cli_simplified_flow":  "true",
			"originator":                 "proagent",
		},
	}
}

func AnthropicConfig(providerID string) Config {
	if providerID == "" {
		providerID = "anthropic"
	}
	return Config{
		Type:        "anthropic",
		ProviderID:  "anthropic",
		AuthURL:     "https://claude.ai/oauth/authorize",
		TokenURL:    "https://platform.claude.com/v1/oauth/token",
		ClientID:    "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
		Scopes:      []string{"org:create_api_key", "user:profile", "user:inference", "user:sessions:claude_code", "user:mcp_servers", "user:file_upload"},
		RedirectURL: "http://localhost:53692/callback",
		CacheKey:    providerID,
	}
}

type anthropicTokenSource struct {
	ctx   context.Context
	cfg   Config
	creds *oauth2.Token
}

func (s *anthropicTokenSource) Token() (*oauth2.Token, error) {
	if s.creds != nil && s.creds.Valid() {
		return s.creds, nil
	}
	if s.creds == nil || s.creds.RefreshToken == "" {
		return nil, fmt.Errorf("anthropic oauth credentials are missing a refresh token; run oauth-login --provider anthropic")
	}
	creds, err := anthropicExchange(s.ctx, s.cfg, map[string]any{
		"grant_type":    "refresh_token",
		"client_id":     s.cfg.ClientID,
		"refresh_token": s.creds.RefreshToken,
	})
	if err != nil {
		return nil, err
	}
	if creds.RefreshToken == "" {
		creds.RefreshToken = s.creds.RefreshToken
	}
	s.creds = creds.Token()
	return s.creds, nil
}

func anthropicExchange(ctx context.Context, cfg Config, payload map[string]any) (Credentials, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return Credentials{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.TokenURL, bytes.NewReader(data))
	if err != nil {
		return Credentials{}, err
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("accept", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Credentials{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Credentials{}, fmt.Errorf("anthropic token exchange failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var out struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		TokenType    string `json:"token_type"`
		ExpiresIn    int    `json:"expires_in"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return Credentials{}, err
	}
	if out.AccessToken == "" {
		return Credentials{}, fmt.Errorf("anthropic token response missing access_token")
	}
	expiry := time.Now().Add(time.Duration(out.ExpiresIn)*time.Second - 5*time.Minute)
	if out.ExpiresIn <= 0 {
		expiry = time.Time{}
	}
	return Credentials{AccessToken: out.AccessToken, RefreshToken: out.RefreshToken, TokenType: firstNonEmpty(out.TokenType, "Bearer"), Expiry: expiry}, nil
}

func AuthorizationHeader(ctx context.Context, cfg Config) (string, error) {
	source, err := TokenSource(ctx, cfg)
	if err != nil || source == nil {
		return "", err
	}
	token, err := source.Token()
	if err != nil || token == nil || token.AccessToken == "" {
		return "", err
	}
	return "Bearer " + token.AccessToken, nil
}

func AccountIDFromJWT(token string, claimPath string) string {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return ""
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		payload, err = base64.StdEncoding.DecodeString(parts[1])
	}
	if err != nil {
		return ""
	}
	var claims map[string]any
	if json.Unmarshal(payload, &claims) != nil {
		return ""
	}
	raw, _ := claims[claimPath].(map[string]any)
	accountID, _ := raw["chatgpt_account_id"].(string)
	return accountID
}

func WithExtraAuthParams(authURL string, params map[string]string) string {
	u, err := url.Parse(authURL)
	if err != nil {
		return authURL
	}
	q := u.Query()
	for k, v := range params {
		q.Set(k, v)
	}
	u.RawQuery = q.Encode()
	return u.String()
}
