package contentio

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	llm "github.com/proiceremo/ai-sdk"
)

func NewContentBlockFromURI(uri string, mimeType *string) (llm.ContentBlock, error) {
	mt := ""
	if mimeType != nil {
		mt = *mimeType
	}

	if after, found := strings.CutPrefix(uri, "file://"); found {
		dataURI, inferredMT, err := FileToDataURI(after)
		if err != nil {
			return llm.ContentBlock{}, err
		}
		if mt == "" {
			mt = inferredMT
		}
		return ContentBlockFromMimeTypeBase64(mt, dataURI), nil
	}

	if mt == "" {
		dataURI, inferredMT, err := URLToDataURI(uri)
		if err != nil {
			return llm.ContentBlock{}, err
		}
		mt = inferredMT
		return ContentBlockFromMimeTypeBase64(mt, dataURI), nil
	}

	return ContentBlockFromMimeTypeURL(mt, uri), nil
}

func ContentBlockFromMimeTypeBase64(mimeType, data string) llm.ContentBlock {
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return llm.NewImageContentBlockFromBase64(data, mimeType)
	case strings.HasPrefix(mimeType, "audio/"):
		return llm.NewAudioContentBlockFromBase64(data, mimeType)
	case strings.HasPrefix(mimeType, "video/"):
		return llm.NewVideoContentBlockFromBase64(data, mimeType)
	default:
		return llm.NewDocumentContentBlockFromBase64(data, mimeType)
	}
}

func ContentBlockFromMimeTypeURL(mimeType, url string) llm.ContentBlock {
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return llm.NewImageContentBlockFromURL(url, mimeType)
	case strings.HasPrefix(mimeType, "audio/"):
		return llm.NewAudioContentBlockFromURL(url, mimeType)
	case strings.HasPrefix(mimeType, "video/"):
		return llm.NewVideoContentBlockFromURL(url, mimeType)
	default:
		return llm.NewDocumentContentBlockFromURL(url, mimeType)
	}
}

func URLToDataURI(url string) (string, string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", "", fmt.Errorf("failed to open file: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return "", "", fmt.Errorf("failed to fetch url: status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	sniffBuf := make([]byte, 512)
	n, err := resp.Body.Read(sniffBuf)
	if err != nil && err != io.EOF {
		return "", "", fmt.Errorf("failed to read for sniffing: %w", err)
	}

	mimeType := http.DetectContentType(sniffBuf[:n])

	var sb strings.Builder
	sb.WriteString("data:")
	sb.WriteString(mimeType)
	sb.WriteString(";base64,")

	fullStream := io.MultiReader(bytes.NewReader(sniffBuf[:n]), resp.Body)

	encoder := base64.NewEncoder(base64.StdEncoding, &sb)
	if _, err := io.Copy(encoder, fullStream); err != nil {
		return "", "", fmt.Errorf("failed to encode stream: %w", err)
	}
	if err := encoder.Close(); err != nil {
		return "", "", fmt.Errorf("failed to finish encoding stream: %w", err)
	}

	return sb.String(), mimeType, nil
}

func FileToDataURI(filePath string) (string, string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	sniffBuf := make([]byte, 512)
	n, err := file.Read(sniffBuf)
	if err != nil && err != io.EOF {
		return "", "", fmt.Errorf("failed to read for sniffing: %w", err)
	}

	mimeType := http.DetectContentType(sniffBuf[:n])

	var sb strings.Builder
	sb.WriteString("data:")
	sb.WriteString(mimeType)
	sb.WriteString(";base64,")

	fullStream := io.MultiReader(bytes.NewReader(sniffBuf[:n]), file)

	encoder := base64.NewEncoder(base64.StdEncoding, &sb)
	if _, err := io.Copy(encoder, fullStream); err != nil {
		return "", "", fmt.Errorf("failed to encode stream: %w", err)
	}
	if err := encoder.Close(); err != nil {
		return "", "", fmt.Errorf("failed to finish encoding stream: %w", err)
	}

	return sb.String(), mimeType, nil
}

func FileToContentBlock(filePath string) (llm.ContentBlock, error) {
	dataURI, mimeType, err := FileToDataURI(filePath)
	if err != nil {
		return llm.ContentBlock{}, fmt.Errorf("failed to convert file to data URI: %w", err)
	}
	return ContentBlockFromMimeTypeBase64(mimeType, dataURI), nil
}
