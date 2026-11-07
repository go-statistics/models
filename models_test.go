package models

import "testing"

func TestName(t *testing.T) {
	if got, want := Name(), "github.com/go-statistics/models"; got != want {
		t.Fatalf("Name() = %q, want %q", got, want)
	}
}

func FuzzName(f *testing.F) {
	f.Add(0)
	f.Fuzz(func(t *testing.T, _ int) {
		if Name() != ModulePath {
			t.Fatal("Name() != ModulePath")
		}
	})
}
