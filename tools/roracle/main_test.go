package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const good = `{"tool":"roracle","r":"4.5.2","packages":{"stats":"4.5.2"},"machine":"x86_64","cases":[
{"name":"b","seed":1,"ints":{"df":3},"floats":{"p":[1,2]}},
{"name":"a","seed":2,"ints":{"df":4},"floats":{"p":[0,1e-3]}}]}`

func decode(t *testing.T, s string) *Fixture {
	t.Helper()
	f, err := Decode(strings.NewReader(s))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestDecodeRefusals(t *testing.T) {
	for name, in := range map[string]string{
		"syntax":    `{`,
		"unknown":   `{"nope":1}`,
		"trailing":  good + ` {}`,
		"empty":     `{"cases":[]}`,
		"noname":    `{"cases":[{"name":""}]}`,
		"duplicate": `{"cases":[{"name":"a"},{"name":"a"}]}`,
		"nan":       `{"cases":[{"name":"a","floats":{"p":[1e999]}}]}`,
	} {
		if _, err := Decode(strings.NewReader(in)); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}

func TestEncodeRefusesNaN(t *testing.T) {
	var z float64
	f := &Fixture{Cases: []Case{{Name: "a", Floats: map[string][]float64{"p": {z / z}}}}}
	if err := Encode(io.Discard, f); err == nil {
		t.Error("Encode of NaN should fail")
	}
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("boom") }

func TestDecodeReadErrorAndSize(t *testing.T) {
	if _, err := Decode(errReader{}); err == nil {
		t.Error("read error not reported")
	}
	if _, err := Decode(io.LimitReader(zeros{}, maxFixtureBytes+10)); err == nil ||
		!strings.Contains(err.Error(), "too large") {
		t.Errorf("size bound: %v", err)
	}
}

type zeros struct{}

func (zeros) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = ' '
	}
	return len(p), nil
}

func TestEncodeSortedAndStable(t *testing.T) {
	f := decode(t, good)
	var a, b bytes.Buffer
	if err := Encode(&a, f); err != nil {
		t.Fatal(err)
	}
	if err := Encode(&b, decode(t, a.String())); err != nil {
		t.Fatal(err)
	}
	if a.String() != b.String() {
		t.Error("encode is not stable")
	}
	if strings.Index(a.String(), `"a"`) > strings.Index(a.String(), `"b"`) {
		t.Error("cases are not sorted by name")
	}
}

func TestDiff(t *testing.T) {
	a := decode(t, good)
	if d := Diff(a, decode(t, good), 0); len(d) != 0 {
		t.Fatalf("identical fixtures disagree: %v", d)
	}
	b := decode(t, `{"cases":[
{"name":"b","ints":{"df":4},"floats":{"p":[1,2.5,3]}},
{"name":"a","ints":{"other":1},"floats":{"q":[1],"p":[1e-3,1e-3]}},
{"name":"c","ints":{}}]}`)
	got := Diff(a, b, 1e-12)
	var all []string
	for _, d := range got {
		all = append(all, d.String())
	}
	text := strings.Join(all, "\n")
	for _, want := range []string{
		"b df: 3 vs 4", "b p: shape differs", "a df: 4 vs absent", "a other: absent vs 1",
		"a q: shape differs", "a p[0]", "c case: missing from first",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in\n%s", want, text)
		}
	}
	if d := Diff(b, a, 1e-12); !strings.Contains(d[len(d)-1].String()+d[0].String(), "case") {
		t.Errorf("reverse diff: %v", d)
	}
	x := decode(t, `{"cases":[{"name":"z","floats":{"p":[0,100]}}]}`)
	y := decode(t, `{"cases":[{"name":"z","floats":{"p":[1e-13,100.00000000001]}}]}`)
	if d := Diff(x, y, 1e-12); len(d) != 0 {
		t.Errorf("within tolerance but reported: %v", d)
	}
	if d := Diff(x, y, 1e-15); len(d) != 2 {
		t.Errorf("tolerance 1e-15 should report 2, got %v", d)
	}
	if d := Diff(a, &Fixture{}, 0); len(d) != 2 || !strings.Contains(d[0].String(), "missing from second") {
		t.Errorf("empty second: %v", d)
	}
}

func TestCommand(t *testing.T) {
	def := command("")
	if def[0] != "docker" || !contains(def, "linux/amd64") || !contains(def, rImage) || !contains(def, "--vanilla") {
		t.Errorf("default command: %v", def)
	}
	c := command("linux/arm64")
	if !contains(c, "linux/arm64") || contains(c, "linux/amd64") {
		t.Errorf("platform not honoured: %v", c)
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func TestDefaultRunner(t *testing.T) {
	var out bytes.Buffer
	if err := runner([]string{"go", "version"}, strings.NewReader(""), &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out.String(), "go version") {
		t.Errorf("output %q", out.String())
	}
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

func withRunner(t *testing.T, fn func(argv []string, stdin io.Reader, stdout, stderr io.Writer) error) {
	t.Helper()
	old := runner
	runner = fn
	t.Cleanup(func() { runner = old })
}

func TestGen(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "f.json")
	var gotScript []byte
	var gotArgv []string
	withRunner(t, func(argv []string, stdin io.Reader, stdout, _ io.Writer) error {
		gotArgv = argv
		gotScript, _ = io.ReadAll(stdin)
		_, err := io.WriteString(stdout, good)
		return err
	})
	msg, err := run(t, "gen", "-o", out, "--platform", "linux/amd64")
	if err != nil {
		t.Fatal(err, msg)
	}
	if !bytes.Equal(gotScript, script) || gotArgv[0] != "docker" {
		t.Errorf("runner got argv %v, script %d bytes", gotArgv, len(gotScript))
	}
	if !strings.Contains(msg, "wrote 2 cases") {
		t.Errorf("message %q", msg)
	}
	if _, err := readFixture(out); err != nil {
		t.Error(err)
	}
	if _, err := run(t, "diff", out, out); err != nil {
		t.Error(err)
	}
}

func TestGenFailures(t *testing.T) {
	dir := t.TempDir()
	withRunner(t, func([]string, io.Reader, io.Writer, io.Writer) error { return errors.New("no uv") })
	if _, err := run(t, "gen", "-o", filepath.Join(dir, "f.json")); err == nil {
		t.Error("runner error not reported")
	}
	withRunner(t, func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error {
		_, err := io.WriteString(w, "not json")
		return err
	})
	if _, err := run(t, "gen", "-o", filepath.Join(dir, "f.json")); err == nil {
		t.Error("bad oracle output not reported")
	}
	withRunner(t, func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error {
		_, err := io.WriteString(w, good)
		return err
	})
	if _, err := run(t, "gen", "-o", filepath.Join(dir, "missing", "f.json")); err == nil {
		t.Error("unwritable output not reported")
	}
}

func TestDiffCommand(t *testing.T) {
	dir := t.TempDir()
	a, b, bad := filepath.Join(dir, "a.json"), filepath.Join(dir, "b.json"), filepath.Join(dir, "bad.json")
	other := strings.Replace(good, `"df":3`, `"df":9`, 1)
	for p, s := range map[string]string{a: good, b: other, bad: "{"} {
		if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	msg, err := run(t, "diff", a, b)
	if !errors.Is(err, ErrDisagree) || !strings.Contains(msg, "1 disagree") {
		t.Errorf("diff a b: %v %q", err, msg)
	}
	if _, err := run(t, "diff", filepath.Join(dir, "none.json"), b); err == nil {
		t.Error("missing first file")
	}
	if _, err := run(t, "diff", a, filepath.Join(dir, "none.json")); err == nil {
		t.Error("missing second file")
	}
	if _, err := run(t, "diff", a, bad); err == nil {
		t.Error("undecodable file")
	}
	if _, err := run(t, "diff", a); err == nil {
		t.Error("wrong arity")
	}
}

func TestMain_(t *testing.T) {
	var code int
	oldExit, oldArgs := osExit, os.Args
	osExit = func(c int) { code = c }
	defer func() { osExit, os.Args = oldExit, oldArgs }()
	os.Args = []string{"roracle", "bogus"}
	main()
	if code != 1 {
		t.Errorf("exit code %d, want 1", code)
	}
	os.Args = []string{"roracle"}
	code = 0
	main()
	if code != 0 {
		t.Errorf("exit code %d, want 0", code)
	}
}
