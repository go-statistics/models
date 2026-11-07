// Package nist reads the NIST Statistical Reference Datasets (StRD) files
// that testdata/nist holds: 11 linear least squares, 11 one-way ANOVA and 27
// nonlinear least squares datasets. Each file has a free-text header that
// states where its blocks are ("Certified Values (lines 31 to 44)", "Data
// (lines 61 to 71)"), the certified block, and the data block.
//
// Parse returns the data columns as float64 and the certified (and starting
// values) blocks as the raw lines the file prints, so that a test can parse
// the certified numbers it needs from the exact text, with Numbers. The
// parser refuses what it does not understand instead of guessing: a block
// whose lines are out of range, a data row with the wrong number of fields,
// a value that is not a finite number, a header that disagrees with the data.
package nist

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

const (
	// MaxBytes bounds the input: the largest StRD file is under 1 MiB.
	MaxBytes = 8 << 20
	// MaxLines bounds the number of lines read.
	MaxLines = 1 << 20
	// headerLines is how many leading lines may hold the block locations.
	headerLines = 20
)

// Dataset is one parsed StRD file.
type Dataset struct {
	Name      string
	Procedure string
	// NObs is the number of observations the header states.
	NObs int
	// Columns names the data columns; Data[j][i] is observation i of column j.
	Columns []string
	Data    [][]float64
	// Certified and Starting hold the raw lines of those blocks (Starting is
	// nil for linear and ANOVA files).
	Certified []string
	Starting  []string
}

var (
	reName      = regexp.MustCompile(`^Dataset Name:\s+(\S+)`)
	reProc      = regexp.MustCompile(`^Procedure:\s+(.+?)\s*$`)
	reCertified = regexp.MustCompile(`Certified Values\s*\(lines\s+(\d+)\s+to\s+(\d+)\)`)
	reStarting  = regexp.MustCompile(`Starting Values\s*\(lines\s+(\d+)\s+to\s+(\d+)\)`)
	reData      = regexp.MustCompile(`^\s*Data\s+\(lines\s+(\d+)\s+to\s+(\d+)\)`)
	reObs       = regexp.MustCompile(`^\s*(\d+) Observations\s*$`)
	reNumber    = regexp.MustCompile(`[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?`)
)

type span struct{ from, to int }

func (s span) ok() bool { return s.from > 0 }

func parseSpan(m []string) (span, error) {
	a, err1 := strconv.Atoi(m[1])
	b, err2 := strconv.Atoi(m[2])
	if err1 != nil || err2 != nil || a < 1 || b < a {
		return span{}, fmt.Errorf("bad line range %q to %q", m[1], m[2])
	}
	return span{a, b}, nil
}

// Parse reads one StRD file.
func Parse(r io.Reader) (*Dataset, error) {
	lines, err := readLines(r)
	if err != nil {
		return nil, err
	}
	d := &Dataset{}
	var cert, start, data span
	for i, l := range lines {
		if i >= headerLines {
			break
		}
		if m := reName.FindStringSubmatch(l); m != nil {
			d.Name = m[1]
		}
		if m := reProc.FindStringSubmatch(l); m != nil {
			d.Procedure = m[1]
		}
		for _, p := range []struct {
			re  *regexp.Regexp
			dst *span
		}{{reCertified, &cert}, {reStarting, &start}, {reData, &data}} {
			if m := p.re.FindStringSubmatch(l); m != nil {
				if *p.dst, err = parseSpan(m); err != nil {
					return nil, err
				}
			}
		}
	}
	if d.Name == "" || d.Procedure == "" || !cert.ok() || !data.ok() {
		return nil, errors.New("nist: header lacks the name, procedure, certified or data block")
	}
	for _, l := range lines {
		if m := reObs.FindStringSubmatch(l); m != nil {
			if d.NObs, err = strconv.Atoi(m[1]); err != nil {
				return nil, err
			}
			break
		}
	}
	if d.Certified, err = block(lines, cert, "certified"); err != nil {
		return nil, err
	}
	if start.ok() {
		if d.Starting, err = block(lines, start, "starting"); err != nil {
			return nil, err
		}
	}
	if err := d.readData(lines, data); err != nil {
		return nil, err
	}
	return d, nil
}

func readLines(r io.Reader) ([]string, error) {
	var lines []string
	sc := bufio.NewScanner(io.LimitReader(r, MaxBytes+1))
	sc.Buffer(nil, 1<<16)
	total := 0
	for sc.Scan() {
		total += len(sc.Text()) + 1
		if total > MaxBytes {
			return nil, errors.New("nist: input too large")
		}
		if len(lines) >= MaxLines {
			return nil, errors.New("nist: too many lines")
		}
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return lines, nil
}

func block(lines []string, s span, what string) ([]string, error) {
	if s.to > len(lines) {
		return nil, fmt.Errorf("nist: %s block lines %d to %d, file has %d", what, s.from, s.to, len(lines))
	}
	return lines[s.from-1 : s.to], nil
}

// readData parses the data block: the line before it names the columns
// ("Data:  y  x"), every row has one value per column.
func (d *Dataset) readData(lines []string, s span) error {
	if s.from < 2 {
		return errors.New("nist: data block has no header line before it")
	}
	rows, err := block(lines, s, "data")
	if err != nil {
		return err
	}
	head := strings.Fields(strings.TrimPrefix(strings.TrimSpace(lines[s.from-2]), "Data:"))
	if len(head) == 0 {
		return errors.New("nist: data header names no column")
	}
	d.Columns = head
	d.Data = make([][]float64, len(head))
	for i, l := range rows {
		f := strings.Fields(l)
		if len(f) != len(head) {
			return fmt.Errorf("nist: data line %d has %d fields, want %d", s.from+i, len(f), len(head))
		}
		for j, tok := range f {
			v, err := strconv.ParseFloat(tok, 64)
			if err != nil || v-v != 0 { // v-v is NaN for NaN and Inf
				return fmt.Errorf("nist: data line %d: %q is not a finite number", s.from+i, tok)
			}
			d.Data[j] = append(d.Data[j], v)
		}
	}
	if d.NObs != 0 && d.NObs != len(rows) {
		return fmt.Errorf("nist: header says %d observations, data has %d", d.NObs, len(rows))
	}
	return nil
}

// Numbers returns the numeric tokens of a certified-block line, in order
// ("b1 = 25  0.25  1.928E-01" gives 1, 25, 0.25, 1.928E-01: the index in the
// name counts as a number, so tests skip it by position). A token that does
// not parse as a finite float64 is an error.
func Numbers(line string) ([]float64, error) {
	var out []float64
	for _, tok := range reNumber.FindAllString(line, -1) {
		v, err := strconv.ParseFloat(tok, 64)
		if err != nil || v-v != 0 {
			return nil, fmt.Errorf("nist: %q is not a finite number", tok)
		}
		out = append(out, v)
	}
	return out, nil
}
