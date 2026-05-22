package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/proiceremo/ai-sdk/oauthx"
	mcpauth "github.com/modelcontextprotocol/go-sdk/auth"
	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/clientcredentials"
)

const (
	defaultConnectTimeout = 30 * time.Second
	defaultCallTimeout    = 120 * time.Second
	defaultKeepAlive      = 30 * time.Second
)

type Connection struct {
	cfg ServerConfig

	mu       sync.Mutex
	client   *sdkmcp.Client
	session  *sdkmcp.ClientSession
	callback *oauthCallbackServer
}

func NewConnection(cfg ServerConfig) *Connection {
	return &Connection{cfg: cfg}
}

func (c *Connection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closeLocked()
}

func (c *Connection) closeLocked() error {
	var err error
	if c.session != nil {
		err = c.session.Close()
		c.session = nil
	}
	if c.callback != nil {
		_ = c.callback.Close()
		c.callback = nil
	}
	return err
}

func (c *Connection) Ping(ctx context.Context) error {
	session, err := c.ensureSession(ctx)
	if err != nil {
		return err
	}
	return session.Ping(ctx, &sdkmcp.PingParams{})
}

func (c *Connection) ListTools(ctx context.Context, cursor string) (*sdkmcp.ListToolsResult, error) {
	var out *sdkmcp.ListToolsResult
	err := c.withSession(ctx, func(session *sdkmcp.ClientSession) error {
		result, err := session.ListTools(ctx, &sdkmcp.ListToolsParams{Cursor: cursor})
		out = result
		return err
	})
	return out, err
}

func (c *Connection) CallTool(ctx context.Context, name string, args map[string]any) (*sdkmcp.CallToolResult, error) {
	var out *sdkmcp.CallToolResult
	err := c.withSession(ctx, func(session *sdkmcp.ClientSession) error {
		result, err := session.CallTool(ctx, &sdkmcp.CallToolParams{Name: name, Arguments: args})
		out = result
		return err
	})
	return out, err
}

func (c *Connection) ListResources(ctx context.Context, cursor string) (*sdkmcp.ListResourcesResult, error) {
	var out *sdkmcp.ListResourcesResult
	err := c.withSession(ctx, func(session *sdkmcp.ClientSession) error {
		result, err := session.ListResources(ctx, &sdkmcp.ListResourcesParams{Cursor: cursor})
		out = result
		return err
	})
	return out, err
}

func (c *Connection) ReadResource(ctx context.Context, uri string) (*sdkmcp.ReadResourceResult, error) {
	var out *sdkmcp.ReadResourceResult
	err := c.withSession(ctx, func(session *sdkmcp.ClientSession) error {
		result, err := session.ReadResource(ctx, &sdkmcp.ReadResourceParams{URI: uri})
		out = result
		return err
	})
	return out, err
}

func (c *Connection) Reconnect(ctx context.Context) error {
	c.mu.Lock()
	_ = c.closeLocked()
	c.mu.Unlock()
	_, err := c.ensureSession(ctx)
	return err
}

func (c *Connection) withSession(ctx context.Context, fn func(*sdkmcp.ClientSession) error) error {
	session, err := c.ensureSession(ctx)
	if err != nil {
		return err
	}
	err = fn(session)
	if !isReconnectableMCPError(err) {
		return err
	}
	c.mu.Lock()
	_ = c.closeLocked()
	c.mu.Unlock()
	session, connErr := c.ensureSession(ctx)
	if connErr != nil {
		return fmt.Errorf("%w; reconnect failed: %v", err, connErr)
	}
	return fn(session)
}

func (c *Connection) ensureSession(ctx context.Context) (*sdkmcp.ClientSession, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session != nil {
		return c.session, nil
	}
	connectTimeout := c.cfg.ConnectTimeout()
	connectCtx, cancel := context.WithTimeout(ctx, connectTimeout)
	defer cancel()

	transport, callback, err := c.transport(connectCtx)
	if err != nil {
		return nil, err
	}
	c.callback = callback
	c.client = sdkmcp.NewClient(&sdkmcp.Implementation{
		Name:    "proagent",
		Title:   "ProAgent",
		Version: "0.1.0",
	}, &sdkmcp.ClientOptions{KeepAlive: c.cfg.KeepAlive()})
	session, err := c.client.Connect(connectCtx, transport, nil)
	if err != nil {
		if callback != nil {
			_ = callback.Close()
			c.callback = nil
		}
		return nil, err
	}
	c.session = session
	go c.watch(session)
	return session, nil
}

func (c *Connection) watch(session *sdkmcp.ClientSession) {
	_ = session.Wait()
	c.mu.Lock()
	if c.session == session {
		c.session = nil
	}
	c.mu.Unlock()
}

func (c *Connection) transport(ctx context.Context) (sdkmcp.Transport, *oauthCallbackServer, error) {
	if c.cfg.URL != "" {
		httpClient := &http.Client{
			Transport: staticHeaderTransport{headers: c.cfg.HeaderMap(), base: http.DefaultTransport},
		}
		if c.cfg.transportType() == "sse" {
			if c.cfg.Auth.Enabled() {
				return nil, nil, fmt.Errorf("mcp server %q uses deprecated SSE transport; OAuth auto-flow is only supported for streamable HTTP", c.cfg.Name)
			}
			return &sdkmcp.SSEClientTransport{
				Endpoint:   c.cfg.URL,
				HTTPClient: httpClient,
			}, nil, nil
		}
		var callback *oauthCallbackServer
		var oauthHandler mcpauth.OAuthHandler
		if strings.EqualFold(strings.ReplaceAll(c.cfg.Auth.Type, "-", "_"), "client_credentials") {
			oauthHandler = clientCredentialsOAuthHandler{
				cfg:    c.cfg.Auth,
				client: httpClient,
			}
		} else if c.cfg.ShouldUseOAuth() {
			var err error
			callback, oauthHandler, err = newOAuthHandler(ctx, c.cfg, httpClient)
			if err != nil {
				return nil, nil, err
			}
		}
		return &sdkmcp.StreamableClientTransport{
			Endpoint:     c.cfg.URL,
			HTTPClient:   httpClient,
			MaxRetries:   c.cfg.MaxRetries,
			OAuthHandler: oauthHandler,
		}, callback, nil
	}
	if c.cfg.Command == "" {
		return nil, nil, fmt.Errorf("mcp server %q has neither url nor command", c.cfg.Name)
	}
	cmd := exec.Command(c.cfg.Command, c.cfg.Args...)
	cmd.Env = append(os.Environ(), c.cfg.EnvList()...)
	cmd.Stderr = mcpStderr()
	return &sdkmcp.CommandTransport{Command: cmd, TerminateDuration: 5 * time.Second}, nil, nil
}

type clientCredentialsOAuthHandler struct {
	cfg    AuthConfig
	client *http.Client
}

func (h clientCredentialsOAuthHandler) TokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	clientCtx := context.WithValue(ctx, oauth2.HTTPClient, h.client)
	cc := clientcredentials.Config{
		ClientID:     h.cfg.ClientID,
		ClientSecret: h.cfg.ClientSecret,
		TokenURL:     h.cfg.TokenURL,
		Scopes:       h.cfg.Scopes,
	}
	return cc.TokenSource(clientCtx), nil
}

func (h clientCredentialsOAuthHandler) Authorize(context.Context, *http.Request, *http.Response) error {
	return nil
}

func newOAuthHandler(ctx context.Context, cfg ServerConfig, httpClient *http.Client) (*oauthCallbackServer, mcpauth.OAuthHandler, error) {
	callback, err := newOAuthCallbackServer()
	if err != nil {
		return nil, nil, err
	}
	authCfg := &mcpauth.AuthorizationCodeHandlerConfig{
		RedirectURL:              callback.RedirectURL(),
		AuthorizationCodeFetcher: callback.Fetch,
		Client:                   httpClient,
	}
	if cfg.Auth.ClientID != "" {
		authCfg.PreregisteredClient = &oauthex.ClientCredentials{ClientID: cfg.Auth.ClientID}
		if cfg.Auth.ClientSecret != "" {
			authCfg.PreregisteredClient.ClientSecretAuth = &oauthex.ClientSecretAuth{ClientSecret: cfg.Auth.ClientSecret}
		}
	} else {
		authCfg.DynamicClientRegistrationConfig = &mcpauth.DynamicClientRegistrationConfig{
			Metadata: &oauthex.ClientRegistrationMetadata{
				RedirectURIs:            []string{callback.RedirectURL()},
				ClientName:              "ProAgent MCP Client",
				GrantTypes:              []string{"authorization_code", "refresh_token"},
				ResponseTypes:           []string{"code"},
				TokenEndpointAuthMethod: "none",
				Scope:                   strings.Join(cfg.Auth.Scopes, " "),
			},
		}
	}
	handler, err := mcpauth.NewAuthorizationCodeHandler(authCfg)
	if err != nil {
		_ = callback.Close()
		return nil, nil, err
	}
	_ = ctx
	return callback, newPersistentOAuthHandler(cfg, handler, httpClient), nil
}

type persistentOAuthHandler struct {
	cfg    ServerConfig
	inner  mcpauth.OAuthHandler
	client *http.Client
}

func newPersistentOAuthHandler(cfg ServerConfig, inner mcpauth.OAuthHandler, client *http.Client) mcpauth.OAuthHandler {
	return &persistentOAuthHandler{cfg: cfg, inner: inner, client: client}
}

func (h *persistentOAuthHandler) TokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	source, err := h.inner.TokenSource(ctx)
	if err != nil {
		return nil, err
	}
	key := h.storeKey()
	if source != nil {
		return &persistingTokenSource{source: source, key: key}, nil
	}
	creds, err := oauthx.MCPStore().Load(key)
	if err != nil || creds.AccessToken == "" {
		return nil, nil
	}
	token := creds.Token()
	if h.cfg.Auth.TokenURL != "" && h.cfg.Auth.ClientID != "" {
		oc := oauth2.Config{
			ClientID:     h.cfg.Auth.ClientID,
			ClientSecret: h.cfg.Auth.ClientSecret,
			Endpoint: oauth2.Endpoint{
				TokenURL: h.cfg.Auth.TokenURL,
			},
			Scopes: h.cfg.Auth.Scopes,
		}
		clientCtx := context.WithValue(ctx, oauth2.HTTPClient, h.client)
		ts := oc.TokenSource(clientCtx, token)
		return &persistingTokenSource{source: ts, key: key}, nil
	}
	if !token.Valid() {
		return nil, nil
	}
	return oauth2.StaticTokenSource(token), nil
}

func (h *persistentOAuthHandler) Authorize(ctx context.Context, req *http.Request, resp *http.Response) error {
	if err := h.inner.Authorize(ctx, req, resp); err != nil {
		return err
	}
	source, err := h.inner.TokenSource(ctx)
	if err != nil || source == nil {
		return err
	}
	token, err := source.Token()
	if err == nil && token != nil {
		_ = oauthx.MCPStore().Save(h.storeKey(), oauthx.CredentialsFromToken(token))
	}
	return err
}

// storeKey resolves the MCP server's identifier under oauthx.MCPStore().
// FileStore handles filename sanitisation internally, so we just hand it
// the human-readable name (preferred) or URL (fallback).
func (h *persistentOAuthHandler) storeKey() string {
	if h.cfg.Name != "" {
		return h.cfg.Name
	}
	return h.cfg.URL
}

// persistingTokenSource writes refreshed tokens back to oauthx.MCPStore so
// the next process inherits a valid access token without re-doing the
// OAuth dance.
type persistingTokenSource struct {
	source oauth2.TokenSource
	key    string
}

func (s *persistingTokenSource) Token() (*oauth2.Token, error) {
	token, err := s.source.Token()
	if err == nil && token != nil {
		_ = oauthx.MCPStore().Save(s.key, oauthx.CredentialsFromToken(token))
	}
	return token, err
}

type staticHeaderTransport struct {
	headers map[string]string
	base    http.RoundTripper
}

func (t staticHeaderTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	for name, value := range t.headers {
		if clone.Header.Get(name) == "" {
			clone.Header.Set(name, value)
		}
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(clone)
}

func isReconnectableMCPError(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, sdkmcp.ErrConnectionClosed) ||
		errors.Is(err, sdkmcp.ErrSessionMissing) ||
		strings.Contains(strings.ToLower(err.Error()), "connection closed")
}

func mcpStderr() io.Writer {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return io.Discard
	}
	dir := home + string(os.PathSeparator) + ".pro" + string(os.PathSeparator) + "logs"
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return io.Discard
	}
	f, err := os.OpenFile(dir+string(os.PathSeparator)+"mcp-stderr.log", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return io.Discard
	}
	return f
}

type oauthCallbackServer struct {
	server *http.Server
	ln     net.Listener

	mu      sync.Mutex
	waiters []chan oauthResult
}

type oauthResult struct {
	code  string
	state string
	err   error
}

func newOAuthCallbackServer() (*oauthCallbackServer, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	cb := &oauthCallbackServer{ln: ln}
	mux := http.NewServeMux()
	mux.HandleFunc("/", cb.handle)
	cb.server = &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		_ = cb.server.Serve(ln)
	}()
	return cb, nil
}

func (cb *oauthCallbackServer) RedirectURL() string {
	return "http://" + cb.ln.Addr().String() + "/oauth/callback"
}

func (cb *oauthCallbackServer) Fetch(ctx context.Context, args *mcpauth.AuthorizationArgs) (*mcpauth.AuthorizationResult, error) {
	ch := make(chan oauthResult, 1)
	cb.mu.Lock()
	cb.waiters = append(cb.waiters, ch)
	cb.mu.Unlock()
	fmt.Printf("[MCP AUTH URL] %s\n", args.URL)
	_ = openBrowser(args.URL)
	select {
	case res := <-ch:
		if res.err != nil {
			return nil, res.err
		}
		return &mcpauth.AuthorizationResult{Code: res.code, State: res.state}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (cb *oauthCallbackServer) handle(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	res := oauthResult{code: query.Get("code"), state: query.Get("state")}
	if e := query.Get("error"); e != "" {
		res.err = fmt.Errorf("oauth error: %s", e)
	} else if res.code == "" {
		res.err = fmt.Errorf("oauth callback missing code")
	}
	cb.mu.Lock()
	if len(cb.waiters) == 0 {
		cb.mu.Unlock()
		http.Error(w, "No authorization flow is waiting for this callback.", http.StatusBadRequest)
		return
	}
	ch := cb.waiters[0]
	cb.waiters = cb.waiters[1:]
	cb.mu.Unlock()
	ch <- res
	if res.err != nil {
		http.Error(w, res.err.Error(), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = io.WriteString(w, "<html><body><h1>Authorization complete</h1><p>You can close this tab.</p></body></html>")
}

func (cb *oauthCallbackServer) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return cb.server.Shutdown(ctx)
}

func openBrowser(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	return cmd.Start()
}
