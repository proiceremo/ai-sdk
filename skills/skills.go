package skills

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"go.yaml.in/yaml/v4"
)

type Options struct {
	Roots       []string
	ManagedRoot string
}

type Loader struct {
	roots       []string
	managedRoot string
}

func NewLoader(opts Options) *Loader {
	roots := opts.Roots
	if len(roots) == 0 {
		roots = defaultRoots()
	}
	return &Loader{
		roots:       roots,
		managedRoot: opts.ManagedRoot,
	}
}

func defaultRoots() []string {
	var roots []string
	if home, err := os.UserHomeDir(); err == nil {
		roots = append(roots, filepath.Join(home, ".agents", "skills"))
	}
	return roots
}

type Input struct {
	Mode     string `json:"mode" jsonschema:"description=Skills action.,enum=list,enum=search,enum=read,enum=create,enum=patch,enum=archive,enum=restore,default=list"`
	Skill    string `json:"skill,omitempty" jsonschema:"description=Skill name (required for read/create/patch/archive/restore)"`
	Query    string `json:"query,omitempty" jsonschema:"description=Search query used for list/search"`
	Resource string `json:"resource,omitempty" jsonschema:"description=Optional resource discriminator"`
	Limit    int    `json:"limit,omitempty" jsonschema:"description=Maximum results to return,minimum=1"`
	Cursor   string `json:"cursor,omitempty" jsonschema:"description=Pagination cursor reserved for future use"`
	Content  string `json:"content,omitempty" jsonschema:"description=Full SKILL.md content for create/patch"`
	Pinned   *bool  `json:"pinned,omitempty" jsonschema:"description=Whether to pin an agent-created skill"`
}

type SkillInfo struct {
	Name          string            `json:"name"`
	Description   string            `json:"description,omitempty"`
	License       string            `json:"license,omitempty"`
	Compatibility string            `json:"compatibility,omitempty"`
	AllowedTools  string            `json:"allowed_tools,omitempty"`
	Metadata      map[string]string `json:"metadata,omitempty"`
	Path          string            `json:"path,omitempty"`
	Location      string            `json:"location,omitempty"`
	Scope         string            `json:"scope,omitempty"`
	Diagnostics   []string          `json:"diagnostics,omitempty"`
	Resources     []string          `json:"resources,omitempty"`
}

type ListResult struct {
	Skills []SkillInfo `json:"skills"`
	Cursor string      `json:"cursor,omitempty"`
}

type ReadResult struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Content     string `json:"content,omitempty"`
	Path        string `json:"path"`
	Location    string `json:"location,omitempty"`
	Scope       string `json:"scope,omitempty"`
}

type skillFile struct {
	info    SkillInfo
	content string
}

func (l *Loader) List(input Input, workingDir string) (ListResult, error) {
	skills, err := l.load(workingDir)
	if err != nil {
		return ListResult{}, err
	}
	query := strings.ToLower(strings.TrimSpace(input.Query))
	limit := input.Limit
	if limit < 0 {
		limit = 0
	}
	results := make([]SkillInfo, 0, len(skills))
	for _, skill := range skills {
		if query != "" && !strings.Contains(strings.ToLower(skill.info.Name), query) && !strings.Contains(strings.ToLower(skill.info.Description), query) {
			continue
		}
		results = append(results, skill.info)
		if limit > 0 && len(results) >= limit {
			break
		}
	}
	return ListResult{Skills: results}, nil
}

func (l *Loader) Read(input Input, workingDir ...string) (ReadResult, error) {
	name := strings.TrimSpace(input.Skill)
	if name == "" {
		return ReadResult{}, fmt.Errorf("skill name required")
	}
	wd := ""
	if len(workingDir) > 0 {
		wd = workingDir[0]
	}
	skills, err := l.load(wd)
	if err != nil {
		return ReadResult{}, err
	}
	for _, skill := range skills {
		if name != "" && skill.info.Name != name {
			continue
		}
		return ReadResult{
			Name:        skill.info.Name,
			Description: skill.info.Description,
			Content:     skill.content,
			Path:        skill.info.Path,
			Location:    skill.info.Location,
			Scope:       skill.info.Scope,
		}, nil
	}
	return ReadResult{}, fmt.Errorf("skill not found: %s", name)
}

func (l *Loader) load(workingDir string) ([]skillFile, error) {
	roots := l.rootsFor(workingDir)
	skills := make([]skillFile, 0)
	seen := map[string]bool{}
	for _, root := range roots {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			dir := filepath.Join(root, entry.Name())
			path := filepath.Join(dir, "SKILL.md")
			if seen[entry.Name()] {
				continue
			}
			info, content, err := readSkillFile(path, entry.Name(), root, skillScope(root, workingDir))
			if err != nil {
				continue
			}
			seen[info.Name] = true
			skills = append(skills, skillFile{info: info, content: content})
		}
	}
	slices.SortFunc(skills, func(a, b skillFile) int { return strings.Compare(a.info.Name, b.info.Name) })
	return skills, nil
}

func (l *Loader) rootsFor(workingDir string) []string {
	roots := make([]string, 0, len(l.roots)+2)
	if strings.TrimSpace(workingDir) != "" {
		roots = append(roots,
			filepath.Join(workingDir, ".agents", "skills"),
		)
	}
	roots = append(roots, l.roots...)
	out := make([]string, 0, len(roots))
	seen := map[string]bool{}
	for _, root := range roots {
		root = filepath.Clean(root)
		if root == "." || seen[root] {
			continue
		}
		seen[root] = true
		out = append(out, root)
	}
	return out
}

func readSkillFile(path, fallbackName, root, scope string) (SkillInfo, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SkillInfo{}, "", err
	}
	content := string(data)
	meta, diagnostics := parseFrontMatter(content)
	if meta.Name == "" {
		meta.Name = fallbackName
	}
	return SkillInfo{
		Name:          meta.Name,
		Description:   meta.Description,
		License:       meta.License,
		Compatibility: meta.Compatibility,
		AllowedTools:  meta.AllowedTools,
		Metadata:      meta.Metadata,
		Path:          path,
		Location:      root,
		Scope:         scope,
		Diagnostics:   diagnostics,
		Resources:     skillResources(filepath.Dir(path)),
	}, content, nil
}

type metadata struct {
	Name          string            `yaml:"name"`
	Description   string            `yaml:"description"`
	License       string            `yaml:"license"`
	Compatibility string            `yaml:"compatibility"`
	AllowedTools  string            `yaml:"allowed-tools"`
	Metadata      map[string]string `yaml:"metadata"`
}

func parseFrontMatter(content string) (metadata, []string) {
	trimmed := strings.TrimPrefix(content, "\ufeff")
	if !strings.HasPrefix(trimmed, "---\n") {
		return metadata{}, nil
	}
	rest := strings.TrimPrefix(trimmed, "---\n")
	header, _, found := strings.Cut(rest, "\n---")
	if !found {
		return metadata{}, []string{"unterminated YAML frontmatter"}
	}
	var meta metadata
	if err := yaml.Unmarshal([]byte(header), &meta); err != nil {
		return metadata{}, []string{"invalid YAML frontmatter: " + err.Error()}
	}
	return meta, nil
}

func skillScope(root, workingDir string) string {
	if workingDir != "" {
		cleanRoot := filepath.Clean(root)
		if cleanRoot == filepath.Clean(filepath.Join(workingDir, ".agents", "skills")) {
			return "workspace"
		}
	}
	if home, err := os.UserHomeDir(); err == nil && strings.HasPrefix(filepath.Clean(root), filepath.Clean(home)) {
		return "global"
	}
	return "local"
}

func skillResources(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	resources := make([]string, 0)
	for _, entry := range entries {
		if entry.Name() == "SKILL.md" || strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		resources = append(resources, entry.Name())
	}
	slices.Sort(resources)
	return resources
}

type UsageMetadata struct {
	Provenance    string    `json:"provenance"`
	State         string    `json:"state"`
	Pinned        bool      `json:"pinned"`
	UseCount      int       `json:"use_count"`
	ViewCount     int       `json:"view_count"`
	PatchCount    int       `json:"patch_count"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
	LastUsedAt    time.Time `json:"last_used_at,omitempty"`
	LastViewedAt  time.Time `json:"last_viewed_at,omitempty"`
	LastPatchedAt time.Time `json:"last_patched_at,omitempty"`
}

func ReadUsage(skillDir string) (UsageMetadata, error) {
	data, err := os.ReadFile(filepath.Join(skillDir, ".usage.json"))
	if err != nil {
		return UsageMetadata{}, err
	}
	var usage UsageMetadata
	err = json.Unmarshal(data, &usage)
	return usage, err
}

func WriteUsage(skillDir string, usage UsageMetadata) error {
	if usage.CreatedAt.IsZero() {
		usage.CreatedAt = time.Now().UTC()
	}
	usage.UpdatedAt = time.Now().UTC()
	data, err := json.MarshalIndent(usage, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(skillDir, ".usage.json"), append(data, '\n'), 0o600)
}

func TouchUsage(skillDir string, action string) error {
	usage, err := ReadUsage(skillDir)
	if err != nil {
		usage = UsageMetadata{Provenance: "manual", State: "active", CreatedAt: time.Now().UTC()}
	}
	now := time.Now().UTC()
	switch action {
	case "use":
		usage.UseCount++
		usage.LastUsedAt = now
	case "view":
		usage.ViewCount++
		usage.LastViewedAt = now
	case "patch":
		usage.PatchCount++
		usage.LastPatchedAt = now
	}
	return WriteUsage(skillDir, usage)
}
