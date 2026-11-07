package nist

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const dataDir = "../../testdata/nist"

// sources reads testdata/nist/SOURCES: path -> sha256.
func sources(t *testing.T) map[string]string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dataDir, "SOURCES"))
	if err != nil {
		t.Fatal(err)
	}
	m := map[string]string{}
	for _, l := range strings.Split(string(b), "\n") {
		f := strings.Fields(l)
		if len(f) == 4 && !strings.HasPrefix(l, "#") {
			m[f[2]] = f[0]
		}
	}
	return m
}

// TestAll49 parses every committed file, checks it against SOURCES, and
// checks the facts the header states: the group counts, the observation
// count, the certified block, the second start vector of the nonlinear files.
func TestAll49(t *testing.T) {
	src := sources(t)
	if len(src) != 49 {
		t.Fatalf("SOURCES lists %d files, want 49", len(src))
	}
	groups := map[string]int{}
	for path, want := range src {
		raw, err := os.ReadFile(filepath.Join(dataDir, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256(raw)
		if hex.EncodeToString(h[:]) != want {
			t.Errorf("%s: SHA-256 differs from SOURCES", path)
		}
		d, err := Parse(strings.NewReader(string(raw)))
		if err != nil {
			t.Errorf("%s: %v", path, err)
			continue
		}
		group := strings.SplitN(path, "/", 2)[0]
		groups[group]++
		base := strings.TrimSuffix(filepath.Base(path), ".dat")
		if d.Name != base {
			t.Errorf("%s: dataset name %q", path, d.Name)
		}
		if d.NObs == 0 || len(d.Data) == 0 || len(d.Data[0]) != d.NObs {
			t.Errorf("%s: %d observations stated, %d columns", path, d.NObs, len(d.Data))
		}
		if len(d.Certified) == 0 {
			t.Errorf("%s: empty certified block", path)
		}
		if (group == "nls") != (d.Starting != nil) {
			t.Errorf("%s: starting values present = %v in group %s", path, d.Starting != nil, group)
		}
		wantProc := map[string]string{
			"linear": "Linear Least Squares Regression",
			"anova":  "Analysis of Variance",
			"nls":    "Nonlinear Least Squares Regression",
		}[group]
		if d.Procedure != wantProc {
			t.Errorf("%s: procedure %q, want %q", path, d.Procedure, wantProc)
		}
	}
	if groups["linear"] != 11 || groups["anova"] != 11 || groups["nls"] != 27 {
		t.Errorf("parsed per group: %v", groups)
	}
}

// TestKnownValues checks parsed numbers against the files read by eye.
func TestKnownValues(t *testing.T) {
	load := func(p string) *Dataset {
		f, err := os.Open(filepath.Join(dataDir, p))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		d, err := Parse(f)
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	ni := load("linear/NoInt1.dat")
	if fmt.Sprint(ni.Columns) != "[y x]" || ni.Data[0][0] != 130 || ni.Data[1][10] != 70 {
		t.Errorf("NoInt1: %v %v", ni.Columns, ni.Data)
	}
	// B1 line: "B1  2.07438016528926  0.165289256198347E-01"
	var b1 []float64
	for _, l := range ni.Certified {
		if strings.Contains(l, "B1  ") {
			b1, _ = Numbers(l)
		}
	}
	if len(b1) != 3 || b1[1] != 2.07438016528926 || b1[2] != 0.0165289256198347 {
		t.Errorf("NoInt1 B1: %v", b1)
	}
	sm := load("anova/SmLs07.dat")
	if len(sm.Data[1]) != 189 || sm.Data[1][0] != 1000000000000.4 || sm.Data[0][188] != 9 {
		t.Errorf("SmLs07: %d rows", len(sm.Data[1]))
	}
	mg := load("nls/MGH09.dat")
	if len(mg.Starting) != 4 || len(mg.Certified) != 9 {
		t.Errorf("MGH09 blocks: %d starting, %d certified", len(mg.Starting), len(mg.Certified))
	}
	n, err := Numbers(mg.Starting[0])
	if err != nil || len(n) != 5 || n[1] != 25 || n[2] != 0.25 || n[3] != 1.9280693458e-01 {
		t.Errorf("MGH09 b1: %v %v", n, err)
	}
}

// minimal: lines 1-6 header, 7-8 certified (c1, c2), 9-11 blank, 12 data
// header, 13-14 data.
const minimal = `NIST/ITL StRD
Dataset Name:  Tiny (Tiny.dat)
File Format:   ASCII
               Certified Values (lines 7 to  8)
               Data             (lines 13 to 14)
Procedure:     Linear Least Squares Regression
c1
c2
               2 Observations


Data:  y  x
 1 2
 3 4
`

func TestParseMinimal(t *testing.T) {
	d, err := Parse(strings.NewReader(minimal))
	if err != nil {
		t.Fatal(err)
	}
	if d.Name != "Tiny" || d.NObs != 2 || fmt.Sprint(d.Data) != "[[1 3] [2 4]]" || len(d.Certified) != 2 || d.Starting != nil {
		t.Errorf("%+v", d)
	}
}

func TestParseRefusals(t *testing.T) {
	rep := func(old, new string) string { return strings.Replace(minimal, old, new, 1) }
	for name, in := range map[string]string{
		"empty":            "",
		"no name":          rep("Dataset Name:  Tiny (Tiny.dat)", "x"),
		"no procedure":     rep("Procedure:", "Other:"),
		"no certified":     rep("Certified Values", "Certified Valves"),
		"no data":          rep("Data             (lines", "Datum            (lines"),
		"range reversed":   rep("(lines 7 to  8)", "(lines 8 to 7)"),
		"range zero":       rep("(lines 7 to  8)", "(lines 0 to 8)"),
		"range overflow":   rep("(lines 7 to  8)", "(lines 7 to 99999999999999999999)"),
		"certified past":   rep("(lines 7 to  8)", "(lines 7 to 99)"),
		"data past":        rep("(lines 13 to 14)", "(lines 13 to 15)"),
		"starting past":    rep("Certified Values (lines 7 to  8)", "Certified Values (lines 7 to  8)\n               Starting Values (lines 7 to 99)"),
		"data at line one": rep("(lines 13 to 14)", "(lines 1 to 14)"),
		"no header names":  rep("Data:  y  x", "Data:"),
		"short row":        rep(" 3 4", " 3"),
		"not a number":     rep(" 3 4", " 3 four"),
		"nan":              rep(" 3 4", " 3 nan"),
		"inf":              rep(" 3 4", " 3 +Inf"),
		"nobs mismatch":    rep("2 Observations", "5 Observations"),
		"nobs overflow":    rep("2 Observations", "99999999999999999999 Observations"),
		"long line":        minimal + strings.Repeat("x", 1<<17) + "\n",
	} {
		if _, err := Parse(strings.NewReader(in)); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("io") }

// endless yields ' ' and newlines forever.
type endless struct{ nl bool }

func (e endless) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = '\n'
		if e.nl == false {
			p[i] = ' '
		}
	}
	return len(p), nil
}

func TestParseBounds(t *testing.T) {
	if _, err := Parse(errReader{}); err == nil {
		t.Error("read error")
	}
	// Short lines forever: more than MaxLines lines is refused. MaxBytes
	// would trip first on 1 byte lines, so check both limits separately.
	if _, err := Parse(io.LimitReader(endless{nl: true}, MaxLines+10)); err == nil ||
		!strings.Contains(err.Error(), "too many lines") {
		t.Errorf("many short lines: %v", err)
	}
	// 4 KiB lines: the byte bound trips before the line bound.
	big := strings.NewReader(strings.Repeat(strings.Repeat("a", 4095)+"\n", MaxBytes/4096+2))
	if _, err := Parse(big); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Errorf("big input: %v", err)
	}
}

func TestNumbers(t *testing.T) {
	n, err := Numbers("x = -1.5E+02 .5 7. 3e2 name2")
	want := []float64{-150, 0.5, 7, 300, 2}
	if err != nil || len(n) != len(want) {
		t.Fatalf("%v %v", n, err)
	}
	for i := range want {
		if n[i] != want[i] {
			t.Errorf("n[%d] = %v, want %v", i, n[i], want[i])
		}
	}
	if _, err := Numbers("1e999"); err == nil {
		t.Error("overflowing literal accepted")
	}
	if n, err := Numbers("no numbers here"); err != nil || len(n) != 0 {
		t.Errorf("%v %v", n, err)
	}
	if math.IsNaN(0) {
		t.Fatal("unreachable")
	}
}

func FuzzParse(f *testing.F) {
	f.Add(minimal)
	f.Add("")
	f.Add("Dataset Name: a\nProcedure: b\nCertified Values (lines 1 to 1)\nData (lines 2 to 2)\n")
	f.Fuzz(func(t *testing.T, s string) {
		d, err := Parse(strings.NewReader(s))
		if err != nil {
			return
		}
		// Invariants of an accepted file: rectangular finite data.
		for _, col := range d.Data {
			if len(col) != len(d.Data[0]) {
				t.Fatal("ragged data")
			}
			for _, v := range col {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatal("non-finite value accepted")
				}
			}
		}
		if len(d.Columns) != len(d.Data) {
			t.Fatal("columns and data differ")
		}
	})
}
