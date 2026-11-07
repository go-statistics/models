package main

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestManifest(t *testing.T) {
	m := Manifest()
	if len(m) != 49 {
		t.Fatalf("%d files, want 49 (11 linear, 11 anova, 27 nls)", len(m))
	}
	count := map[string]int{}
	seen := map[string]bool{}
	for _, f := range m {
		count[f.Group]++
		if seen[f.Path()] || seen[f.URL] {
			t.Errorf("duplicate %s", f.Path())
		}
		seen[f.Path()], seen[f.URL] = true, true
		if !strings.HasPrefix(f.URL, base) || !strings.HasSuffix(f.URL, f.Name+".dat") {
			t.Errorf("bad URL %s", f.URL)
		}
	}
	if count["linear"] != 11 || count["anova"] != 11 || count["nls"] != 27 {
		t.Errorf("counts %v", count)
	}
	if got := linkURL("linear", "Norris"); got != base+"lls/data/LINKS/DATA/Norris.dat" {
		t.Errorf("linear URL %s", got)
	}
	if got := linkURL("anova", "SmLs01"); got != base+"anova/SmLs01.dat" {
		t.Errorf("anova URL %s", got)
	}
}

// fake serves canned responses to client, for any URL.
type fake func(*http.Request) (*http.Response, error)

func (f fake) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func respond(code int, body string) fake {
	return func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: code, Body: io.NopCloser(strings.NewReader(body)), Header: http.Header{}}, nil
	}
}

func withClient(t *testing.T, rt http.RoundTripper) {
	t.Helper()
	old := client
	client = &http.Client{Transport: rt, CheckRedirect: old.CheckRedirect}
	t.Cleanup(func() { client = old })
}

func TestGet(t *testing.T) {
	url := base + "x.dat"
	withClient(t, respond(200, "data"))
	if b, err := get(url); err != nil || string(b) != "data" {
		t.Fatalf("get: %q %v", b, err)
	}
	if _, err := get("https://example.com/x.dat"); err == nil {
		t.Error("foreign host accepted")
	}
	withClient(t, respond(404, "no"))
	if _, err := get(url); err == nil || !strings.Contains(err.Error(), "404") {
		t.Errorf("404: %v", err)
	}
	withClient(t, respond(200, ""))
	if _, err := get(url); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Errorf("empty: %v", err)
	}
	withClient(t, respond(200, strings.Repeat("x", maxFileBytes+1)))
	if _, err := get(url); err == nil || !strings.Contains(err.Error(), "larger") {
		t.Errorf("oversize: %v", err)
	}
	withClient(t, fake(func(*http.Request) (*http.Response, error) { return nil, errors.New("down") }))
	if _, err := get(url); err == nil {
		t.Error("transport error")
	}
	withClient(t, fake(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(errBody{}), Header: http.Header{}}, nil
	}))
	if _, err := get(url); err == nil {
		t.Error("body read error")
	}
}

type errBody struct{}

func (errBody) Read([]byte) (int, error) { return 0, errors.New("reset") }

func TestRedirectRefused(t *testing.T) {
	r := client.CheckRedirect(nil, nil)
	if !errors.Is(r, http.ErrUseLastResponse) {
		t.Errorf("CheckRedirect = %v", r)
	}
}

func withGet(t *testing.T, fn func(string) ([]byte, error)) {
	t.Helper()
	old := get
	get = fn
	t.Cleanup(func() { get = old })
}

func run(t *testing.T, args ...string) (string, error) {
	t.Helper()
	cmd := RootCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs(args)
	err := cmd.Execute()
	return out.String(), err
}

func TestFetchThenVerify(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nist")
	withGet(t, func(url string) ([]byte, error) { return []byte("content of " + url), nil })
	if _, err := run(t, "fetch", "--dir", dir); err != nil {
		t.Fatal(err)
	}
	out, err := run(t, "verify", "--dir", dir)
	if err != nil || !strings.Contains(out, "49 files checked, 0 problems") {
		t.Fatalf("verify: %v %q", err, out)
	}
	// Tamper with one file, drop another, add an unknown SOURCES line.
	if err := os.WriteFile(filepath.Join(dir, "linear", "Norris.dat"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "anova", "SmLs01.dat")); err != nil {
		t.Fatal(err)
	}
	sp := filepath.Join(dir, sourcesName)
	src, _ := os.ReadFile(sp)
	src = append(src, []byte(strings.Repeat("a", 64)+"  1  linear/Extra.dat  https://x\n")...)
	src = bytes.Replace(src, []byte("linear/Pontius.dat"), []byte("linear/Pontiux.dat"), 1)
	if err := os.WriteFile(sp, src, 0o644); err != nil {
		t.Fatal(err)
	}
	out, err = run(t, "verify", "--dir", dir)
	if !errors.Is(err, ErrMismatch) {
		t.Fatalf("verify err = %v", err)
	}
	for _, want := range []string{
		"linear/Norris.dat: differs from SOURCES",
		"anova/SmLs01.dat: open",
		"linear/Pontius.dat: not in SOURCES",
		"linear/Extra.dat: in SOURCES but not in the manifest",
		"linear/Pontiux.dat: in SOURCES but not in the manifest",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in\n%s", want, out)
		}
	}
}

func TestFetchFailures(t *testing.T) {
	dir := t.TempDir()
	withGet(t, func(string) ([]byte, error) { return nil, errors.New("offline") })
	if _, err := run(t, "fetch", "--dir", dir); err == nil {
		t.Error("download error not reported")
	}
	withGet(t, func(string) ([]byte, error) { return []byte("x"), nil })
	blocker := filepath.Join(dir, "file")
	if err := os.WriteFile(blocker, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := run(t, "fetch", "--dir", filepath.Join(blocker, "sub")); err == nil {
		t.Error("mkdir under a file not reported")
	}
	ro := filepath.Join(dir, "ro")
	if err := os.MkdirAll(filepath.Join(ro, "linear", "Norris.dat"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := run(t, "fetch", "--dir", ro); err == nil {
		t.Error("writing over a directory not reported")
	}
	// SOURCES itself cannot be written: it is a directory.
	d2 := filepath.Join(dir, "d2")
	if err := os.MkdirAll(filepath.Join(d2, sourcesName), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := run(t, "fetch", "--dir", d2); err == nil {
		t.Error("unwritable SOURCES not reported")
	}
}

func TestVerifyFailures(t *testing.T) {
	dir := t.TempDir()
	if _, err := run(t, "verify", "--dir", dir); err == nil {
		t.Error("missing SOURCES not reported")
	}
	if err := os.WriteFile(filepath.Join(dir, sourcesName), []byte("garbage\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := run(t, "verify", "--dir", dir); err == nil || !strings.Contains(err.Error(), "malformed") {
		t.Errorf("malformed SOURCES: %v", err)
	}
}

func TestParseSources(t *testing.T) {
	good := strings.Repeat("0", 64) + "  12  a/b.dat  https://u\n# note\n\n"
	es, err := ParseSources(strings.NewReader(good))
	if err != nil || len(es) != 1 || es[0].Bytes != 12 || es[0].Path != "a/b.dat" {
		t.Fatalf("%v %v", es, err)
	}
	if es[0].String() != strings.TrimSpace(good[:strings.Index(good, "\n")]) {
		t.Errorf("String() = %q", es[0].String())
	}
	if _, err := ParseSources(strings.NewReader("abc 1 p u\n")); err == nil {
		t.Error("short hash accepted")
	}
	if _, err := ParseSources(errLines{}); err == nil {
		t.Error("read error not reported")
	}
}

type errLines struct{}

func (errLines) Read([]byte) (int, error) { return 0, errors.New("io") }

func TestRootAndMain(t *testing.T) {
	if _, err := run(t); err != nil {
		t.Errorf("bare root: %v", err)
	}
	if _, err := run(t, "bogus"); err == nil {
		t.Error("unknown subcommand accepted")
	}
	var code int
	oldExit, oldArgs := osExit, os.Args
	osExit = func(c int) { code = c }
	defer func() { osExit, os.Args = oldExit, oldArgs }()
	os.Args = []string{"nistfetch", "bogus"}
	main()
	if code != 1 {
		t.Errorf("exit %d, want 1", code)
	}
	os.Args = []string{"nistfetch"}
	code = 0
	main()
	if code != 0 {
		t.Errorf("exit %d, want 0", code)
	}
}
