// smoracle runs statsmodels on the cases of smoke.py (W1 and later add the
// NIST and seeded cases) and writes the fixture file the models tests replay;
// its diff subcommand compares two fixtures (arm64 against x86_64 wheels, or a
// regenerated fixture against the committed one).
//
// Usage (from tools/smoracle):
//
//	GOWORK=off go run . gen --platform linux/amd64 -o ../../testdata/smoracle.json
//	GOWORK=off go run . diff a.json b.json
//
// Generating needs uv on PATH (and Docker with --platform); statsmodels,
// NumPy and SciPy are pinned below. The Python script is fed on standard
// input to an isolated interpreter (python -I), so nothing in the working
// directory is imported. This module is separate from the library and is
// never imported by it. Committed fixtures must be generated with
// --platform linux/amd64: macOS arm64 wheels use FMA.
//
// Not yet implemented (design-models.md, section 15): the bench subcommand and
// --mp (50-digit mpmath re-solve).
package main

import (
	"bytes"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/spf13/cobra"
)

//go:embed smoke.py
var script []byte

const (
	statsmodelsPin = "statsmodels==0.15.0"
	scipyPin       = "scipy==1.18.1"
	numpyPin       = "numpy==2.5.4"
)

// osExit is replaced by tests.
var osExit = os.Exit

// uvImage is the container image used with --platform.
const uvImage = "ghcr.io/astral-sh/uv:0.9.5-python3.12-bookworm-slim"

// command returns the oracle command line: uv on the host, or uv inside a
// container of the given platform (e.g. linux/amd64).
func command(platform string) []string {
	uv := []string{"uv", "run", "-q", "--no-project", "--with", statsmodelsPin, "--with", scipyPin, "--with", numpyPin, "python", "-I", "-"}
	if platform == "" {
		return uv
	}
	return append([]string{"docker", "run", "--rm", "-i", "--platform", platform, uvImage}, uv...)
}

// runner runs the oracle command; tests replace it.
var runner = func(argv []string, stdin io.Reader, stdout, stderr io.Writer) error {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = stdin, stdout, stderr
	return cmd.Run()
}

// ErrDisagree is returned by diff when the fixtures differ.
var ErrDisagree = errors.New("fixtures disagree")

func readFixture(path string) (*Fixture, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fx, err := Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return fx, nil
}

func genCmd() *cobra.Command {
	var out, platform string
	cmd := &cobra.Command{
		Use:          "gen",
		Short:        "Run statsmodels and write the fixture file",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			var buf bytes.Buffer
			if err := runner(command(platform), bytes.NewReader(script), &buf, cmd.ErrOrStderr()); err != nil {
				return fmt.Errorf("oracle: %w", err)
			}
			f, err := Decode(&buf)
			if err != nil {
				return fmt.Errorf("oracle output: %w", err)
			}
			var enc bytes.Buffer
			// Cannot fail: Decode returned only finite floats, and Encode's
			// sole error is a NaN or Inf (TestEncodeRefusesNaN).
			_ = Encode(&enc, f)
			if err := os.WriteFile(out, enc.Bytes(), 0o644); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "wrote %d cases (statsmodels %s, %s) to %s\n", len(f.Cases), f.Statsmodels, f.Machine, out)
			return nil
		},
	}
	cmd.Flags().StringVarP(&out, "output", "o", "testdata/smoracle.json", "fixture file to write")
	cmd.Flags().StringVar(&platform, "platform", "",
		"run in a container of this platform (e.g. linux/amd64) instead of on the host")
	return cmd
}

func diffCmd() *cobra.Command {
	var rtol float64
	cmd := &cobra.Command{
		Use:          "diff first.json second.json",
		Short:        "Report the cases on which two fixtures disagree (exit 1 if any)",
		SilenceUsage: true,
		Args:         cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			a, err := readFixture(args[0])
			if err != nil {
				return err
			}
			b, err := readFixture(args[1])
			if err != nil {
				return err
			}
			w := cmd.OutOrStdout()
			d := Diff(a, b, rtol)
			names := map[string]bool{}
			for _, x := range d {
				fmt.Fprintln(w, x)
				names[x.Case] = true
			}
			fmt.Fprintf(w, "%d cases (%s, %s) against %d (%s, %s): %d disagree\n",
				len(a.Cases), a.Statsmodels, a.Machine, len(b.Cases), b.Statsmodels, b.Machine, len(names))
			if len(d) > 0 {
				return ErrDisagree
			}
			return nil
		},
	}
	cmd.Flags().Float64Var(&rtol, "rtol", 1e-12, "relative tolerance for floating outputs (integers are exact)")
	return cmd
}

// RootCmd returns the command line of the harness.
func RootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "smoracle",
		Short:        "Generate and compare the statsmodels fixtures of go-statistics/models",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		// A runnable root makes cobra validate Args, so an unknown
		// subcommand is an error instead of a silent help page.
		RunE: func(cmd *cobra.Command, _ []string) error { return cmd.Help() },
	}
	cmd.AddCommand(genCmd(), diffCmd())
	return cmd
}

func main() {
	cmd := RootCmd()
	// Explicit arguments: without SetArgs, cobra ignores os.Args when run
	// from a test binary.
	cmd.SetArgs(os.Args[1:])
	if err := cmd.Execute(); err != nil {
		osExit(1)
	}
}
