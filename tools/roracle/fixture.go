package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
)

// maxFixtureBytes bounds what Decode reads: a fixture is a small committed
// file, so anything larger is refused before it is parsed.
const maxFixtureBytes = 64 << 20

// Fixture is the file the oracle writes and the models tests replay.
type Fixture struct {
	Tool     string            `json:"tool"`
	R        string            `json:"r"`
	Packages map[string]string `json:"packages"`
	Machine  string            `json:"machine"`
	Cases    []Case            `json:"cases"`
}

// Case is one model fit. Integers and decisions (rank, df, flags, iteration
// counts) are compared exactly; floats within a relative tolerance.
type Case struct {
	Name   string               `json:"name"`
	Seed   int64                `json:"seed"`
	Ints   map[string]int64     `json:"ints"`
	Floats map[string][]float64 `json:"floats"`
}

// Decode parses a fixture, refusing unknown fields, trailing data, an empty
// case list and duplicate case names. Non-finite floats cannot occur: JSON
// has no NaN or Inf, and an overflowing literal such as 1e999 is a syntax
// error to encoding/json.
func Decode(r io.Reader) (*Fixture, error) {
	data, err := io.ReadAll(io.LimitReader(r, maxFixtureBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxFixtureBytes {
		return nil, errors.New("fixture too large")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var f Fixture
	if err := dec.Decode(&f); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, errors.New("trailing data after fixture")
	}
	if len(f.Cases) == 0 {
		return nil, errors.New("fixture has no cases")
	}
	seen := map[string]bool{}
	for _, c := range f.Cases {
		if c.Name == "" || seen[c.Name] {
			return nil, fmt.Errorf("case name %q is empty or duplicated", c.Name)
		}
		seen[c.Name] = true
	}
	return &f, nil
}

// Encode writes a fixture with cases sorted by name, so regenerating gives a
// byte-identical file.
func Encode(w io.Writer, f *Fixture) error {
	g := *f
	g.Cases = append([]Case(nil), f.Cases...)
	sort.Slice(g.Cases, func(i, j int) bool { return g.Cases[i].Name < g.Cases[j].Name })
	enc := json.NewEncoder(w)
	enc.SetIndent("", " ")
	return enc.Encode(&g)
}

// Disagreement is one difference between two fixtures.
type Disagreement struct {
	Case, Field, Detail string
}

func (d Disagreement) String() string { return d.Case + " " + d.Field + ": " + d.Detail }

// Diff lists every difference between a and b. Floats use the relative
// tolerance rtol (absolute when the reference is 0).
func Diff(a, b *Fixture, rtol float64) []Disagreement {
	var out []Disagreement
	bi := map[string]Case{}
	for _, c := range b.Cases {
		bi[c.Name] = c
	}
	ai := map[string]bool{}
	for _, ca := range a.Cases {
		ai[ca.Name] = true
		cb, ok := bi[ca.Name]
		if !ok {
			out = append(out, Disagreement{ca.Name, "case", "missing from second"})
			continue
		}
		for _, k := range sortedKeys(ca.Ints, cb.Ints) {
			x, okx := ca.Ints[k]
			y, oky := cb.Ints[k]
			if !okx || !oky || x != y {
				out = append(out, Disagreement{ca.Name, k, fmt.Sprintf("%v vs %v", ifPresent(okx, x), ifPresent(oky, y))})
			}
		}
		for _, k := range sortedKeys(ca.Floats, cb.Floats) {
			x, okx := ca.Floats[k]
			y, oky := cb.Floats[k]
			if !okx || !oky || len(x) != len(y) {
				out = append(out, Disagreement{ca.Name, k, "shape differs"})
				continue
			}
			for i := range x {
				if !close(x[i], y[i], rtol) {
					out = append(out, Disagreement{ca.Name, fmt.Sprintf("%s[%d]", k, i), fmt.Sprintf("%.17g vs %.17g", x[i], y[i])})
				}
			}
		}
	}
	for _, cb := range b.Cases {
		if !ai[cb.Name] {
			out = append(out, Disagreement{cb.Name, "case", "missing from first"})
		}
	}
	return out
}

func ifPresent(ok bool, v int64) any {
	if !ok {
		return "absent"
	}
	return v
}

func close(x, y, rtol float64) bool {
	if x == y {
		return true
	}
	d := math.Abs(x - y)
	if x == 0 {
		return d <= rtol
	}
	return d <= rtol*math.Abs(x)
}

func sortedKeys[V any](a, b map[string]V) []string {
	set := map[string]bool{}
	for k := range a {
		set[k] = true
	}
	for k := range b {
		set[k] = true
	}
	keys := make([]string, 0, len(set))
	for k := range set {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
