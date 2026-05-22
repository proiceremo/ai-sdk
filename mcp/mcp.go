package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	llm "github.com/proiceremo/ai-sdk"
	"github.com/proiceremo/ai-sdk/tools"
)

type ServerConfig struct {
	Name             string         `json:"name"`
	Command          string         `json:"command,omitempty"`
	Args             []string       `json:"args,omitempty"`
	Env              []EnvVar       `json:"env,omitempty"`
	Headers          []Header       `json:"headers,omitempty"`
	Type             string         `json:"type,omitempty"`
	URL              string         `json:"url,omitempty"`
	Auth             AuthConfig     `json:"auth,omitempty"`
	TimeoutMS        int            `json:"timeout_ms,omitempty"`
	ConnectTimeoutMS int            `json:"connect_timeout_ms,omitempty"`
	KeepAliveMS      int            `json:"keep_alive_ms,omitempty"`
	MaxRetries       int            `json:"max_retries,omitempty"`
	Meta             map[string]any `json:"_meta,omitempty"`
}

type AuthConfig struct {
	Type         string   `json:"type,omitempty"`
	TokenURL     string   `json:"token_url,omitempty"`
	ClientID     string   `json:"client_id,omitempty"`
	ClientSecret string   `json:"client_secret,omitempty"`
	Scopes       []string `json:"scopes,omitempty"`
}

type EnvVar struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type Header struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type ToolInfo struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema,omitempty"`
}

type ServerInfo struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Type   string `json:"type,omitempty"`
	URL    string `json:"url,omitempty"`
	Error  string `json:"error,omitempty"`
}

type Input struct {
	Mode      string         `json:"mode" jsonschema:"description=Action to perform.,enum=info,enum=servers,enum=list_servers,enum=tools,enum=list_tools,enum=search_tools,enum=execute,enum=call_tool,enum=resources,enum=list_resources,enum=read_resource,enum=read,enum=ping,enum=reconnect"`
	Server    string         `json:"server,omitempty" jsonschema:"description=Server name. Optional for server listing/searching; required for execution when more than one server is configured."`
	Tool      string         `json:"tool,omitempty" jsonschema:"description=MCP tool name for execute/call_tool."`
	Arguments map[string]any `json:"arguments,omitempty" jsonschema:"description=Arguments for execute/call_tool."`
	URI       string         `json:"uri,omitempty" jsonschema:"description=Resource URI for read_resource/read."`
	Query     string         `json:"query,omitempty" jsonschema:"description=Case-insensitive search/filter text for server names, tool names, descriptions, and resource metadata."`
	Limit     int            `json:"limit,omitempty" jsonschema:"description=Maximum items to return. Defaults to 50; capped at 200."`
	Cursor    string         `json:"cursor,omitempty" jsonschema:"description=Opaque cursor returned by a previous list/search call."`
	TimeoutMS int            `json:"timeout_ms,omitempty" jsonschema:"description=Call timeout in milliseconds."`
}

type Output struct {
	Servers    []ServerInfo   `json:"servers,omitempty"`
	Tools      []ToolInfo     `json:"tools,omitempty"`
	Resources  []ResourceInfo `json:"resources,omitempty"`
	Result     map[string]any `json:"result,omitempty"`
	Error      string         `json:"error,omitempty"`
	NextCursor string         `json:"next_cursor,omitempty"`
}

type ResourceInfo struct {
	Name        string `json:"name,omitempty"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	URI         string `json:"uri"`
	MIMEType    string `json:"mime_type,omitempty"`
	Size        int64  `json:"size,omitempty"`
}

type ServerStore interface {
	ListServers(ctx context.Context, scopeID string) ([]ServerConfig, error)
}

type Manager struct {
	store       ServerStore
	mu          sync.RWMutex
	connections map[string]map[string]*Connection
}

func NewManager(store ServerStore) *Manager {
	return &Manager{store: store, connections: map[string]map[string]*Connection{}}
}

func (m *Manager) ListServers(ctx context.Context, scopeID string) []ServerInfo {
	configs, _ := m.store.ListServers(ctx, scopeID)
	m.mu.RLock()
	defer m.mu.RUnlock()
	seen := map[string]bool{}
	var servers []ServerInfo
	for _, cfg := range configs {
		status := "configured"
		if conns, ok := m.connections[scopeID]; ok {
			if conns[cfg.Name] != nil {
				status = "connected"
			}
		}
		servers = append(servers, ServerInfo{Name: cfg.Name, Status: status, Type: cfg.transportType(), URL: cfg.URL})
		seen[cfg.Name] = true
	}
	if conns, ok := m.connections[scopeID]; ok {
		for name := range conns {
			if !seen[name] {
				servers = append(servers, ServerInfo{Name: name, Status: "connected"})
			}
		}
	}
	return servers
}

func (m *Manager) Execute(scopeID string, input Input) (Output, error) {
	ctx := context.Background()
	timeout := time.Duration(input.TimeoutMS) * time.Millisecond
	if timeout <= 0 {
		timeout = defaultCallTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	mode := input.Mode
	if mode == "" {
		mode = "info"
	}
	switch mode {
	case "info", "servers", "list_servers":
		return m.listServers(ctx, scopeID, input)
	case "tools", "list_tools", "search_tools":
		return m.listTools(ctx, scopeID, input)
	case "execute", "call_tool":
		return m.callTool(ctx, scopeID, input)
	case "resources", "list_resources":
		return m.listResources(ctx, scopeID, input)
	case "read_resource", "read":
		return m.readResource(ctx, scopeID, input)
	case "ping":
		conn, err := m.connection(ctx, scopeID, input.Server)
		if err != nil {
			return Output{}, err
		}
		return Output{Result: map[string]any{"ok": conn.Ping(ctx) == nil}}, nil
	case "reconnect":
		conn, err := m.connection(ctx, scopeID, input.Server)
		if err != nil {
			return Output{}, err
		}
		return Output{Result: map[string]any{"ok": true}}, conn.Reconnect(ctx)
	default:
		return Output{}, fmt.Errorf("unsupported mcp mode: %s", mode)
	}
}

func (m *Manager) CloseSession(scopeID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, conn := range m.connections[scopeID] {
		_ = conn.Close()
	}
	delete(m.connections, scopeID)
	return nil
}

func (m *Manager) listServers(ctx context.Context, scopeID string, input Input) (Output, error) {
	servers := m.ListServers(ctx, scopeID)
	filtered := make([]ServerInfo, 0, len(servers))
	query := strings.ToLower(strings.TrimSpace(input.Query))
	for _, server := range servers {
		if query == "" || strings.Contains(strings.ToLower(server.Name+" "+server.Type+" "+server.URL+" "+server.Status), query) {
			filtered = append(filtered, server)
		}
	}
	start := cursorOffset(input.Cursor)
	start, end, next := pageBounds(start, input.Limit, len(filtered))
	return Output{Servers: filtered[start:end], NextCursor: next}, nil
}

func (m *Manager) listTools(ctx context.Context, scopeID string, input Input) (Output, error) {
	configs, err := m.store.ListServers(ctx, scopeID)
	if err != nil {
		return Output{}, err
	}
	var servers []string
	if input.Server != "" {
		servers = []string{input.Server}
	} else {
		for _, cfg := range configs {
			if strings.TrimSpace(cfg.Name) != "" {
				servers = append(servers, cfg.Name)
			}
		}
		sort.Strings(servers)
	}
	query := strings.ToLower(strings.TrimSpace(input.Query))
	var all []ToolInfo
	for _, server := range servers {
		conn, err := m.connection(ctx, scopeID, server)
		if err != nil {
			if input.Server != "" {
				return Output{}, err
			}
			continue
		}
		cursor := ""
		for {
			result, err := conn.ListTools(ctx, cursor)
			if err != nil {
				if input.Server != "" {
					return Output{}, err
				}
				break
			}
			for _, tool := range result.Tools {
				info := ToolInfo{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema}
				if server != "" {
					info.Name = server + "/" + info.Name
				}
				if query == "" || strings.Contains(strings.ToLower(info.Name+" "+info.Description), query) {
					all = append(all, info)
				}
			}
			if result.NextCursor == "" {
				break
			}
			cursor = result.NextCursor
		}
	}
	start := cursorOffset(input.Cursor)
	start, end, next := pageBounds(start, input.Limit, len(all))
	return Output{Tools: all[start:end], NextCursor: next}, nil
}

func (m *Manager) callTool(ctx context.Context, scopeID string, input Input) (Output, error) {
	if input.Tool == "" {
		return Output{}, fmt.Errorf("mcp execute requires tool")
	}
	if input.Server == "" && strings.Contains(input.Tool, "/") {
		parts := strings.SplitN(input.Tool, "/", 2)
		input.Server = parts[0]
		input.Tool = parts[1]
	}
	conn, err := m.connection(ctx, scopeID, input.Server)
	if err != nil {
		return Output{}, err
	}
	result, err := conn.CallTool(ctx, input.Tool, input.Arguments)
	if err != nil {
		return Output{}, err
	}
	return Output{Result: map[string]any{
		"content":            result.Content,
		"structured_content": result.StructuredContent,
		"is_error":           result.IsError,
	}}, nil
}

func (m *Manager) listResources(ctx context.Context, scopeID string, input Input) (Output, error) {
	conn, err := m.connection(ctx, scopeID, input.Server)
	if err != nil {
		return Output{}, err
	}
	result, err := conn.ListResources(ctx, input.Cursor)
	if err != nil {
		return Output{}, err
	}
	out := Output{NextCursor: result.NextCursor}
	for _, resource := range result.Resources {
		out.Resources = append(out.Resources, ResourceInfo{
			Name: resource.Name, Title: resource.Title, Description: resource.Description,
			URI: resource.URI, MIMEType: resource.MIMEType, Size: resource.Size,
		})
	}
	return out, nil
}

func (m *Manager) readResource(ctx context.Context, scopeID string, input Input) (Output, error) {
	if input.URI == "" {
		return Output{}, fmt.Errorf("mcp read_resource requires uri")
	}
	conn, err := m.connection(ctx, scopeID, input.Server)
	if err != nil {
		return Output{}, err
	}
	result, err := conn.ReadResource(ctx, input.URI)
	if err != nil {
		return Output{}, err
	}
	return Output{Result: map[string]any{"contents": result.Contents}}, nil
}

func (m *Manager) connection(ctx context.Context, scopeID, serverName string) (*Connection, error) {
	configs, err := m.store.ListServers(ctx, scopeID)
	if err != nil {
		return nil, err
	}
	var cfg *ServerConfig
	for i := range configs {
		if serverName == "" || configs[i].Name == serverName {
			if cfg != nil && serverName == "" {
				return nil, fmt.Errorf("multiple MCP servers configured; specify server")
			}
			cfg = &configs[i]
		}
	}
	if cfg == nil {
		return nil, fmt.Errorf("MCP server %q not configured", serverName)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.connections[scopeID] == nil {
		m.connections[scopeID] = map[string]*Connection{}
	}
	if conn := m.connections[scopeID][cfg.Name]; conn != nil {
		return conn, nil
	}
	conn := NewConnection(*cfg)
	m.connections[scopeID][cfg.Name] = conn
	return conn, nil
}

func (cfg ServerConfig) transportType() string {
	if cfg.Type != "" {
		switch strings.ToLower(strings.TrimSpace(cfg.Type)) {
		case "http", "streamable_http", "streamable-http":
			return "streamable_http"
		case "sse":
			return "sse"
		case "stdio":
			return "stdio"
		default:
			return strings.ToLower(strings.TrimSpace(cfg.Type))
		}
	}
	if cfg.URL != "" {
		return "streamable_http"
	}
	return "stdio"
}

func (cfg ServerConfig) HeaderMap() map[string]string {
	headers := map[string]string{}
	for _, h := range cfg.Headers {
		if h.Name != "" {
			headers[h.Name] = h.Value
		}
	}
	return headers
}

func (cfg ServerConfig) EnvList() []string {
	var env []string
	for _, e := range cfg.Env {
		if e.Name != "" {
			env = append(env, e.Name+"="+e.Value)
		}
	}
	return env
}

func (cfg ServerConfig) ConnectTimeout() time.Duration {
	if cfg.ConnectTimeoutMS > 0 {
		return time.Duration(cfg.ConnectTimeoutMS) * time.Millisecond
	}
	return defaultConnectTimeout
}

func (cfg ServerConfig) CallTimeout() time.Duration {
	if cfg.TimeoutMS > 0 {
		return time.Duration(cfg.TimeoutMS) * time.Millisecond
	}
	return defaultCallTimeout
}

func (cfg ServerConfig) KeepAlive() time.Duration {
	if cfg.KeepAliveMS > 0 {
		return time.Duration(cfg.KeepAliveMS) * time.Millisecond
	}
	return defaultKeepAlive
}

func (a AuthConfig) Enabled() bool {
	switch strings.ToLower(strings.TrimSpace(a.Type)) {
	case "oauth", "oauth2", "authorization_code", "authorization-code", "client_credentials", "client-credentials", "auto":
		return true
	default:
		return false
	}
}

func (cfg ServerConfig) ShouldUseOAuth() bool {
	authType := strings.ToLower(strings.TrimSpace(cfg.Auth.Type))
	switch authType {
	case "none", "disabled", "off", "false":
		return false
	case "oauth", "oauth2", "authorization_code", "authorization-code", "client_credentials", "client-credentials", "auto":
		return true
	case "":
		return !cfg.hasAuthorizationHeader()
	default:
		return false
	}
}

func (cfg ServerConfig) hasAuthorizationHeader() bool {
	for _, header := range cfg.Headers {
		if strings.EqualFold(strings.TrimSpace(header.Name), "authorization") && strings.TrimSpace(header.Value) != "" {
			return true
		}
	}
	return false
}

func NewTool(manager *Manager) llm.Tool {
	return tools.NewGenericTool("mcp", "Inspect and execute tools from Model Context Protocol servers", func(ctx llm.ToolContext, input Input) llm.ToolResult {
		scopeID := ""
		if v, ok := ctx.Vars["session_id"].(string); ok {
			scopeID = v
		}
		mode := input.Mode
		if mode == "" {
			mode = "info"
		}
		switch mode {
		case "info", "servers", "list_servers", "tools", "list_tools", "search_tools", "execute", "call_tool", "resources", "list_resources", "read_resource", "read", "ping", "reconnect":
			output, err := manager.Execute(scopeID, input)
			if err != nil {
				return tools.ErrorResult(err)
			}
			return tools.JSONResult("MCP execute", llm.ToolKindOther, output, "")
		default:
			return tools.ErrorResult(fmt.Errorf("unsupported mcp mode: %s", mode))
		}
	})
}

func cursorOffset(cursor string) int {
	if strings.TrimSpace(cursor) == "" {
		return 0
	}
	data, err := base64.RawURLEncoding.DecodeString(cursor)
	if err != nil {
		return 0
	}
	var payload struct {
		Offset int `json:"offset"`
	}
	if json.Unmarshal(data, &payload) != nil || payload.Offset < 0 {
		return 0
	}
	return payload.Offset
}

func makeCursor(offset int) string {
	if offset <= 0 {
		return ""
	}
	data, _ := json.Marshal(map[string]any{"offset": offset})
	return base64.RawURLEncoding.EncodeToString(data)
}

func pageBounds(start, limit, total int) (int, int, string) {
	if start > total {
		start = total
	}
	if limit <= 0 {
		limit = 50
	}
	if limit > 200 {
		limit = 200
	}
	end := start + limit
	if end > total {
		end = total
	}
	next := ""
	if end < total {
		next = makeCursor(end)
	}
	return start, end, next
}
