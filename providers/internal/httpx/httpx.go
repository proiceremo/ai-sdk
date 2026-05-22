package httpx

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"os"
	"strings"
	"sync"
	"time"
)

// DebugRequest stores request info for error reporting
type DebugRequest struct {
	Method  string
	URL     string
	Headers map[string]string
	Body    string
	Time    time.Time
}

var (
	requestLogMu sync.Mutex
	requestLog   = make(map[string]*DebugRequest) // key = request ID or URL
)

// HTTPStatusError represents an HTTP error with full context
type HTTPStatusError struct {
	StatusCode int
	Body       string
	URL        string
	Method     string
	Request    *DebugRequest
	Response   *http.Response
}

func (e *HTTPStatusError) Error() string {
	return e.format(false)
}

func (e *HTTPStatusError) format(includeRequestDetails bool) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("HTTP %d error from %s %s", e.StatusCode, e.Method, e.URL))
	if body := clipDebugText(strings.TrimSpace(e.Body), 1200); body != "" {
		sb.WriteString(fmt.Sprintf("\nResponse Body: %s", body))
	}

	if includeRequestDetails && e.Request != nil {
		sb.WriteString("\n\n--- Request Details ---\n")
		sb.WriteString(fmt.Sprintf("Method: %s\n", e.Request.Method))
		sb.WriteString(fmt.Sprintf("URL: %s\n", e.Request.URL))
		sb.WriteString(fmt.Sprintf("Time: %s\n", e.Request.Time.Format(time.RFC3339)))
		if len(e.Request.Headers) > 0 {
			sb.WriteString("Headers:\n")
			for k, v := range e.Request.Headers {
				if isSensitiveHeader(k) {
					sb.WriteString(fmt.Sprintf("  %s: [REDACTED]\n", k))
				} else {
					sb.WriteString(fmt.Sprintf("  %s: %s\n", k, v))
				}
			}
		}
		if e.Request.Body != "" {
			sb.WriteString(fmt.Sprintf("Body: %s\n", clipDebugText(maskSensitiveData(e.Request.Body), 4000)))
		}
	}

	return sb.String()
}

// DetailedError returns full error details for logging
func (e *HTTPStatusError) DetailedError() string {
	return e.format(true)
}

func DoJSON(ctx context.Context, client *http.Client, method, url string, headers map[string]string, body any, dest any) error {
	reqDebug := &DebugRequest{
		Method:  method,
		URL:     url,
		Headers: headers,
		Time:    time.Now(),
	}

	if body != nil {
		bodyJSON, _ := json.MarshalIndent(body, "", "  ")
		reqDebug.Body = string(bodyJSON)
	}

	// Store for potential error reporting
	requestLogMu.Lock()
	requestLog[url] = reqDebug
	requestLogMu.Unlock()

	resp, err := doWithDebug(ctx, client, method, url, headers, body, false, reqDebug)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := checkResponse(resp, reqDebug); err != nil {
		return err
	}
	if dest == nil {
		io.Copy(io.Discard, resp.Body)
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(dest)
}

func DoSSE(ctx context.Context, client *http.Client, method, url string, headers map[string]string, body any) (*SSEStream, error) {
	reqDebug := &DebugRequest{
		Method:  method,
		URL:     url,
		Headers: headers,
		Time:    time.Now(),
	}

	if body != nil {
		bodyJSON, _ := json.MarshalIndent(body, "", "  ")
		reqDebug.Body = string(bodyJSON)
	}

	// Log the request
	if os.Getenv("BENCH_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[httpx SSE REQUEST] %s %s\n", method, url)
		if reqDebug.Body != "" {
			fmt.Fprintf(os.Stderr, "[httpx SSE REQUEST BODY]\n%s\n", reqDebug.Body)
		}
	}

	resp, err := doWithDebug(ctx, client, method, url, headers, body, true, reqDebug)
	if err != nil {
		return nil, err
	}
	if err := checkResponse(resp, reqDebug); err != nil {
		resp.Body.Close()
		return nil, err
	}
	return newSSEStream(resp.Body), nil
}

func doWithDebug(ctx context.Context, client *http.Client, method, url string, headers map[string]string, body any, sse bool, reqDebug *DebugRequest) (*http.Response, error) {
	if client == nil {
		client = http.DefaultClient
	}

	var payload io.Reader
	if body != nil {
		buf := &bytes.Buffer{}
		if err := json.NewEncoder(buf).Encode(body); err != nil {
			return nil, err
		}
		payload = bytes.NewReader(buf.Bytes())
	}

	req, err := http.NewRequestWithContext(ctx, method, url, payload)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if sse {
		req.Header.Set("Accept", "text/event-stream")
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	// Dump full request for debugging
	if os.Getenv("BENCH_DEBUG") != "" {
		if dump, err := httputil.DumpRequest(req, true); err == nil {
			// Mask API keys in the dump
			dumpStr := string(dump)
			dumpStr = maskSensitiveData(dumpStr)
			fmt.Fprintf(os.Stderr, "[httpx REQUEST DUMP]\n%s\n", dumpStr)
		}
	}

	return client.Do(req)
}

func checkResponse(resp *http.Response, reqDebug *DebugRequest) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	bodyStr := string(body)

	// Log to stderr immediately
	fmt.Fprintf(os.Stderr, "\n[httpx ERROR RESPONSE]\n")
	fmt.Fprintf(os.Stderr, "Status: %d %s\n", resp.StatusCode, resp.Status)
	fmt.Fprintf(os.Stderr, "URL: %s\n", reqDebug.URL)
	fmt.Fprintf(os.Stderr, "Method: %s\n", reqDebug.Method)
	fmt.Fprintf(os.Stderr, "Response Headers:\n")
	for k, v := range resp.Header {
		fmt.Fprintf(os.Stderr, "  %s: %v\n", k, v)
	}
	fmt.Fprintf(os.Stderr, "Response Body:\n%s\n", bodyStr)
	fmt.Fprintf(os.Stderr, "[httpx END ERROR RESPONSE]\n\n")

	httpErr := &HTTPStatusError{
		StatusCode: resp.StatusCode,
		Body:       bodyStr,
		URL:        reqDebug.URL,
		Method:     reqDebug.Method,
		Request:    reqDebug,
		Response:   resp,
	}

	return httpErr
}

// maskSensitiveData masks API keys and tokens in debug output
func maskSensitiveData(s string) string {
	// Mask API keys in various formats
	s = maskPattern(s, "x-api-key: ", "\n")
	s = maskPattern(s, "X-Api-Key: ", "\n")
	s = maskPattern(s, "x-goog-api-key: ", "\n")
	s = maskPattern(s, "Authorization: Bearer ", "\n")
	s = maskPattern(s, "\"api_key\": \"", "\"")
	s = maskPattern(s, "\"api_key\":\"", "\"")
	return s
}

func isSensitiveHeader(key string) bool {
	lower := strings.ToLower(strings.TrimSpace(key))
	if lower == "" {
		return false
	}
	return strings.Contains(lower, "authorization") || strings.Contains(lower, "api-key")
}

func clipDebugText(text string, limit int) string {
	if limit <= 0 || len(text) <= limit {
		return text
	}
	if limit <= 32 {
		return text[:limit]
	}
	return text[:limit] + "\n...[truncated]"
}

func maskPattern(s, prefix, suffix string) string {
	start := 0
	for {
		idx := strings.Index(s[start:], prefix)
		if idx == -1 {
			break
		}
		idx += start
		endIdx := strings.Index(s[idx+len(prefix):], suffix)
		if endIdx == -1 {
			break
		}
		endIdx += idx + len(prefix)

		// Replace the sensitive part with [REDACTED]
		s = s[:idx+len(prefix)] + "[REDACTED]" + s[endIdx:]
		start = idx + len(prefix) + len("[REDACTED]")
	}
	return s
}

type SSEEvent struct {
	Event string
	Data  []byte
}

type SSEStream struct {
	body    io.ReadCloser
	scanner *bufio.Scanner
	current SSEEvent
	err     error
	ctx     context.Context
	cancel  context.CancelFunc
}

func newSSEStream(body io.ReadCloser) *SSEStream {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024), 8<<20)
	ctx, cancel := context.WithCancel(context.Background())
	return &SSEStream{
		body:    body,
		scanner: scanner,
		ctx:     ctx,
		cancel:  cancel,
	}
}

func (s *SSEStream) Next() bool {
	if s.err != nil {
		return false
	}

	var (
		event string
		data  []string
	)

	emit := func() bool {
		if len(data) == 0 && event == "" {
			return false
		}
		s.current = SSEEvent{
			Event: event,
			Data:  []byte(strings.Join(data, "\n")),
		}
		return true
	}

	for s.scanner.Scan() {
		// Check if context is cancelled
		select {
		case <-s.ctx.Done():
			s.err = s.ctx.Err()
			return false
		default:
		}

		line := strings.TrimSuffix(s.scanner.Text(), "\r")
		if line == "" {
			if emit() {
				return true
			}
			event = ""
			data = data[:0]
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, found := strings.Cut(line, ":")
		if !found {
			field = line
			value = ""
		}
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			event = value
		case "data":
			data = append(data, value)
		}
	}

	if err := s.scanner.Err(); err != nil {
		s.err = err
		return false
	}

	if emit() {
		return true
	}

	return false
}

func (s *SSEStream) Current() SSEEvent {
	return s.current
}

func (s *SSEStream) Err() error {
	return s.err
}

func (s *SSEStream) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	if s.body == nil {
		return nil
	}
	return s.body.Close()
}
