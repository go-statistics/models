// roracle runs R on the cases of smoke.R (W1 and later add the NIST, aliasing
// and sandwich cases) and writes the fixture file the models tests replay;
// its diff subcommand compares two fixtures. Outputs only: R's numbers are
// stored, R itself is never a dependency of the library.
//
// Usage (from tools/roracle):
//
//	GOWORK=off go run . gen -o ../../testdata/roracle.json
//	GOWORK=off go run . diff a.json b.json
//
// Generating needs Docker: the pinned image rocker/r-ver carries R and the
// CRAN snapshot of its date, so R and package versions are fixed together.
// The script is fed on standard input to `Rscript --vanilla -`, so no
// profile, site library or working-directory file is read. This module is
// separate from the library and is never imported by it. Committed fixtures
// are generated with --platform linux/amd64.
//
// Not yet implemented (design-models.md, section 15): the bench subcommand and
// --mp (50-digit mpmath re-solve, which runs in smoracle's Python).
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

//go:embed smoke.R
var script []byte

// rImage pins R and, through rocker's dated CRAN snapshot, the package
// versions. Its tag has not been pulled yet: the first gen run on a machine
// with Docker confirms it.
const rImage = "rocker/r-ver:4.5.2"

// osExit is replaced by tests.
var osExit = os.Exit

// command returns the oracle command line: Rscript in a container of the
// given platform (default linux/amd64, the platform of committed fixtures).
func command(platform string) []string {
	if platform == "" {
		platform = "linux/amd64"
	}
	return []string{"docker", "run", "--rm", "-i", "--platform", platform, rImage, "Rscript", "--vanilla", "-"}
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
		Short:        "Run R and write the fixture file",
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
			fmt.Fprintf(cmd.OutOrStdout(), "wrote %d cases (R %s, %s) to %s\n", len(f.Cases), f.R, f.Machine, out)
			return nil
		},
	}
	cmd.Flags().StringVarP(&out, "output", "o", "testdata/roracle.json", "fixture file to write")
	cmd.Flags().StringVar(&platform, "platform", "linux/amd64", "container platform")
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
			fmt.Fprintf(w, "%d cases (R %s, %s) against %d (R %s, %s): %d disagree\n",
				len(a.Cases), a.R, a.Machine, len(b.Cases), b.R, b.Machine, len(names))
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
		Use:          "roracle",
		Short:        "Generate and compare the R fixtures of go-statistics/models",
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
