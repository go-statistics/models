// Package models will hold the linear models of go-statistics/models (OLS,
// WLS, GLS, results, ANOVA, prediction). The design is in docs/design-models.md.
//
// At W0 the package is a skeleton: it exists so that CI has something to
// build, test and gate on.
package models

// ModulePath is the import path of the module.
const ModulePath = "github.com/go-statistics/models"

// Name returns the module path; it is the single exported symbol of the W0
// skeleton and is replaced by the real API in W1.
func Name() string { return ModulePath }
