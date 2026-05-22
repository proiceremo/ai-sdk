package llm

import (
	"encoding/base64"
	"strings"
)

// SupportedImageBytes reports whether data looks like an image format accepted
// by common chat-completions vision endpoints.
func SupportedImageBytes(data []byte) bool {
	if len(data) >= 8 && string(data[:8]) == "\x89PNG\r\n\x1a\n" {
		return true
	}
	if len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff {
		return true
	}
	if len(data) >= 6 && (string(data[:6]) == "GIF87a" || string(data[:6]) == "GIF89a") {
		return true
	}
	if len(data) >= 4 && (string(data[:4]) == "II*\x00" || string(data[:4]) == "MM\x00*") {
		return true
	}
	if len(data) >= 2 && data[0] == 'B' && data[1] == 'M' {
		return true
	}
	return false
}

// SupportedImageBase64 decodes data URI or bare base64 image data and checks
// the decoded file signature.
func SupportedImageBase64(data string) bool {
	data = strings.TrimSpace(data)
	if data == "" {
		return false
	}
	if strings.HasPrefix(data, "data:") {
		_, payload, ok := strings.Cut(data, ",")
		if !ok {
			return false
		}
		data = payload
	}
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return false
	}
	return SupportedImageBytes(decoded)
}
