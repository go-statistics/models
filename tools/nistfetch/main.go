// nistfetch downloads the 49 NIST Statistical Reference Datasets the models
// repository certifies against (11 linear, 11 one-way ANOVA, 27 nonlinear)
// into testdata/nist and records each file's SHA-256 and size in
// testdata/nist/SOURCES; its verify subcommand re-hashes the committed files
// against SOURCES, offline.
//
// Usage (from tools/nistfetch):
//
//	GOWORK=off go run . fetch --dir ../../testdata/nist
//	GOWORK=off go run . verify --dir ../../testdata/nist
//
// Only https://www.itl.nist.gov is contacted, redirects are refused, and
// every response is bounded (maxFileBytes) before it is written.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

const (
	base         = "https://www.itl.nist.gov/div898/strd/"
	maxFileBytes = 8 << 20
	sourcesName  = "SOURCES"
)

// File is one dataset: the group directory under testdata/nist, the dataset
// name, and where NIST serves it.
type File struct {
	Group, Name, URL string
}

// Path returns the file's path relative to the data directory.
func (f File) Path() string { return f.Group + "/" + f.Name + ".dat" }

func linkURL(group, name string) string {
	switch group {
	case "linear":
		return base + "lls/data/LINKS/DATA/" + name + ".dat"
	case "nls":
		return base + "nls/data/LINKS/DATA/" + name + ".dat"
	}
	return base + "anova/" + name + ".dat"
}

var names = map[string][]string{
	"linear": {"Norris", "Pontius", "NoInt1", "NoInt2", "Filip", "Longley",
		"Wampler1", "Wampler2", "Wampler3", "Wampler4", "Wampler5"},
	"anova": {"SiRstv", "SmLs01", "SmLs02", "SmLs03", "SmLs04", "SmLs05",
		"SmLs06", "SmLs07", "SmLs08", "SmLs09", "AtmWtAg"},
	"nls": {"Misra1a", "Chwirut2", "Chwirut1", "Lanczos3", "Gauss1", "Gauss2",
		"DanWood", "Misra1b", "Kirby2", "Hahn1", "Nelson", "MGH17", "Lanczos1",
		"Lanczos2", "Gauss3", "Misra1c", "Misra1d", "Roszman1", "ENSO", "MGH09",
		"Thurber", "BoxBOD", "Rat42", "MGH10", "Eckerle4", "Rat43", "Bennett5"},
}

// Manifest returns the 49 datasets in a fixed order.
func Manifest() []File {
	var out []File
	for _, g := range []string{"linear", "anova", "nls"} {
		for _, n := range names[g] {
			out = append(out, File{g, n, linkURL(g, n)})
		}
	}
	return out
}

// osExit is replaced by tests.
var osExit = os.Exit

// client refuses redirects: a moved file is a finding, not something to
// follow to another host.
var client = &http.Client{
	Timeout: 60 * time.Second,
	CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	},
}

// get downloads a URL; tests replace it.
var get = func(url string) ([]byte, error) {
	if !strings.HasPrefix(url, base) {
		return nil, fmt.Errorf("refusing %s: not under %s", url, base)
	}
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: HTTP %d", url, resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxFileBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxFileBytes {
		return nil, fmt.Errorf("%s: larger than %d bytes", url, maxFileBytes)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("%s: empty body", url)
	}
	return data, nil
}

// Entry is one SOURCES line.
type Entry struct {
	SHA256 string
	Bytes  int
	Path   string
	URL    string
}

func (e Entry) String() string {
	return fmt.Sprintf("%s  %d  %s  %s", e.SHA256, e.Bytes, e.Path, e.URL)
}

// ParseSources reads a SOURCES file: one entry per line, four fields.
func ParseSources(r io.Reader) ([]Entry, error) {
	var out []Entry
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var e Entry
		if _, err := fmt.Sscanf(line, "%64s %d %s %s", &e.SHA256, &e.Bytes, &e.Path, &e.URL); err != nil || len(e.SHA256) != 64 {
			return nil, fmt.Errorf("SOURCES line %d: malformed", n)
		}
		out = append(out, e)
	}
	return out, sc.Err()
}

func sum(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func fetch(dir string, files []File, w io.Writer) error {
	var entries []Entry
	for _, f := range files {
		data, err := get(f.URL)
		if err != nil {
			return err
		}
		p := filepath.Join(dir, filepath.FromSlash(f.Path()))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(p, data, 0o644); err != nil {
			return err
		}
		entries = append(entries, Entry{sum(data), len(data), f.Path(), f.URL})
		fmt.Fprintf(w, "%s  %d bytes\n", f.Path(), len(data))
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	var b strings.Builder
	b.WriteString("# sha256  bytes  path  url (NIST StRD, public domain; see https://www.itl.nist.gov/div898/strd/)\n")
	for _, e := range entries {
		b.WriteString(e.String() + "\n")
	}
	return os.WriteFile(filepath.Join(dir, sourcesName), []byte(b.String()), 0o644)
}

// ErrMismatch is returned by verify when a file differs from SOURCES or the
// directory does not hold exactly the manifest.
var ErrMismatch = errors.New("data does not match SOURCES")

func verify(dir string, files []File, w io.Writer) error {
	sf, err := os.Open(filepath.Join(dir, sourcesName))
	if err != nil {
		return err
	}
	defer sf.Close()
	entries, err := ParseSources(sf)
	if err != nil {
		return err
	}
	byPath := map[string]Entry{}
	for _, e := range entries {
		byPath[e.Path] = e
	}
	bad := 0
	for _, f := range files {
		e, ok := byPath[f.Path()]
		if !ok {
			fmt.Fprintf(w, "%s: not in SOURCES\n", f.Path())
			bad++
			continue
		}
		delete(byPath, f.Path())
		data, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(f.Path())))
		if err != nil {
			fmt.Fprintf(w, "%s: %v\n", f.Path(), err)
			bad++
			continue
		}
		if e.URL != f.URL || e.Bytes != len(data) || e.SHA256 != sum(data) {
			fmt.Fprintf(w, "%s: differs from SOURCES\n", f.Path())
			bad++
		}
	}
	for p := range byPath {
		fmt.Fprintf(w, "%s: in SOURCES but not in the manifest\n", p)
		bad++
	}
	fmt.Fprintf(w, "%d files checked, %d problems\n", len(files), bad)
	if bad > 0 {
		return ErrMismatch
	}
	return nil
}

// RootCmd returns the command line of the tool.
func RootCmd() *cobra.Command {
	var dir string
	cmd := &cobra.Command{
		Use:          "nistfetch",
		Short:        "Download and verify the NIST StRD datasets of go-statistics/models",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE:         func(cmd *cobra.Command, _ []string) error { return cmd.Help() },
	}
	cmd.PersistentFlags().StringVar(&dir, "dir", "testdata/nist", "directory holding the data files and SOURCES")
	cmd.AddCommand(
		&cobra.Command{
			Use: "fetch", Short: "Download the 49 files and write SOURCES", SilenceUsage: true, Args: cobra.NoArgs,
			RunE: func(cmd *cobra.Command, _ []string) error { return fetch(dir, Manifest(), cmd.OutOrStdout()) },
		},
		&cobra.Command{
			Use: "verify", Short: "Check the files against SOURCES (offline; exit 1 on any difference)", SilenceUsage: true, Args: cobra.NoArgs,
			RunE: func(cmd *cobra.Command, _ []string) error { return verify(dir, Manifest(), cmd.OutOrStdout()) },
		},
	)
	return cmd
}

func main() {
	cmd := RootCmd()
	cmd.SetArgs(os.Args[1:])
	if err := cmd.Execute(); err != nil {
		osExit(1)
	}
}
