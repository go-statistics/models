# Design: statistical models — linear, generalized, robust, discrete, mixed, nonlinear (`go-statistics/models`)

Item 6 of the go-scikit parity plan (wave 2) asks for "linear models and GLM
with standard errors and ANOVA, certified on NIST StRD". On the user's
instruction this design covers the **whole regression surface** of statsmodels
0.15.0, plus what R (`lm`, `glm`, `anova`, `nls`, `step`, `confint`) and MATLAB
(`fitlm`, `fitglm`, `fitnlm`, `fitrm`, `fitlme`, `fitglme`, `stepwiselm`,
`robustfit`) add. Everything is ordered in waves: the numerical core first,
then families of models. Each wave has its own tasks and validation reference.
§2.3 lists everything left out, with the reason.

This document was written in go-statistics/stats before the target
repository existed and moved here at W0; the repository `go-statistics/models`
(module `github.com/go-statistics/models`) is the proposal it argues for.
Reasons:

- PLAN.md "Homes" lists `go-statistics (models, timeseries)`. The bib table
  (`bib/optimization-learning-autodiff.md`, row "Multiple linear regression +
  GLM") says "go-statistics (new repo `models`)".
- Dependency direction: models imports stats (distributions, descriptive
  statistics), and stats must never import models. A sibling module keeps the
  graph acyclic.
- Its own semver: stats is at v0.1.0 and mirrors `scipy.stats`. Models mirrors
  statsmodels/R/MATLAB, is about ten times larger, and will break its API
  several times below v1.
- Different oracles (statsmodels and R, not SciPy and mpmath), different
  fixtures, and some optional heavy dependencies (formula layer, dataframe)
  that stats users should not pull.

Whoever holds the permission must create the repository with `ghnew -public`.
Nothing in this design creates it.

## 1. Sources

### Opened

| Source | What was taken |
|---|---|
| go-scikit/roadmap PLAN.md (wave 2 item 6, "Homes") | scope, home |
| roadmap `bib/optimization-learning-autodiff.md` l. 15, 35–39, 77, 106–111, 130, 149, 162–163, 174, 186 | parity rows; method references named there (McCullagh & Nelder, Golub & Van Loan); statsmodels BSD-3 = "port", R GPL = "oracle only"; NIST row; LRE criterion; "linear algebra is the bottleneck" |
| roadmap `bib/r.md` l. 199 | R `lm.fit` uses LINPACK-derived `dqrdc2` with "limited column pivoting" |
| NIST StRD `general/main.html`, `lls/lls.shtml`, `lls/lls_info.shtml`, `anova/anova.html`, `nls/nls_main.shtml` | the 11 linear, 11 one-way ANOVA and 27 nonlinear datasets with difficulty levels; certified quantities |
| NIST files `Filip.dat`, `Longley.dat`, `Wampler5.dat`, `NoInt1.dat`, `anova/SmLs09.dat`, `nls/.../MGH09.dat` | layouts (certified block, data block), digits (15 for linear and ANOVA, 11 for nonlinear), uncentered R² for NoInt, two start vectors for NLS |
| statsmodels **v0.15.0** (BSD-3), sparse clone of the tag at `~/Documents/VCS/GIT/localhost/statsmodels-statsmodels-v0150`: `regression/{linear_model,_tools,quantile_regression,recursive_ls,rolling,mixed_linear_model,feasible_gls,dimred,process_regression}.py`, `robust/{robust_linear_model,norms,scale,resistant_linear_model}.py`, `discrete/{discrete_model,count_model,truncated_model,conditional_models}.py`, `miscmodels/{ordinal_model,nonlinls,tmodel}.py`, `genmod/{generalized_linear_model,generalized_estimating_equations,cov_struct,qif,bayes_mixed_glm}.py`, `genmod/families/*.py`, `stats/{anova,sandwich_covariance,outliers_influence,diagnostic,_diagnostic_other,stattools}.py`, `base/{model,optimizer,data,elastic_net,_penalties,l1_slsqp}.py`, `tools/tools.py`, `formula/_manager.py`, `gam/`, `multivariate/manova.py`, `duration/hazard_regression.py` (class list only) | every algorithm and default below; line numbers refer to that tag |
| R documentation (not source): `lm`, `glm`, `glm.control`, `anova.glm`, `nls`, `step`, `confint` (R-devel HTML manual) | QR-only `lm`, `singular.ok`; IWLS and `glm.control` rule abs(dev − dev_old)/(abs(dev) + 0.1) < 1e-8, `maxit = 25`; `anova.glm` tests Chisq/LRT/Rao/F/Cp; `nls` Gauss–Newton, `plinear` (Golub–Pereyra), `port` (nl2sol, bounds), relative-offset criterion, `scaleOffset`; `step` directions, `k`, scope, `steps = 1000`, `extractAIC`; `confint` profiles for glm/nls, t formula for lm |
| MATLAB documentation: `fitlm`, `fitglm`, `linearmodel.anova`, `robustfit`, `repeatedmeasuresmodel.ranova`, `fitlme`, `stepwiselm` | QR fitting; `RobustOpts`; sstype 1/2/3/'h'; robustfit weight functions and constants, `r = resid/(tune·s·√(1−h))`, `s = MAD/0.6745`; ranova GG/HF/LB-corrected p-values, `mauchly`, `epsilon`; fitlme ML (default)/REML, five covariance patterns, quasi-Newton; stepwise criteria and PEnter/PRemove defaults |
| go-ndarray v0.31.0 `linalg/{linalg,qr,lstsq}.go` | blocked Householder QR (`ModeRaw`), SVD (Golub–Kahan, `bdc.go`), `Lstsq`, `Pinv`, `Cholesky`, `Solve`, `Inv`, eig/eigh; **no pivoted QR, no public triangular solve** |
| go-statistics/stats main @ 07b6ccb: README, `go.mod`, `correlation.go`, `anova.go`, `studentt.go`, `fdist.go`, `chi2.go`, `normal.go`, `internal/dd` | what exists (`Linregress`, `FOneway`, distributions, descriptive statistics); `dd` is internal and duplicated in `go-numerics/special/internal/dd`; **no multiple-testing corrections** |
| go-numerics/optimize v0.3.1 exported API (module cache), main `tools/slsqporacle/main.go`, `docs/design-qp.md` | `Minimize` (NM, Powell, CG, BFGS, Newton-CG, L-BFGS-B), `LeastSquares` (TRF, LM), `Leastsq`, `CurveFit`, `Linprog`, `ApproxFprime`, SLSQP; **no numerical Hessian**; the oracle-harness pattern |
| go-simd/floats v0.7.0 API | `SumPairwise`, `Dot`, unfused `AxpyNoFMA`/`XpbyNoFMA`/`Axpby3NoFMA` |

### Not obtained (not opened; no number below comes from them)

- McCullagh & Nelder, *Generalized Linear Models* (2nd ed.); Golub & Van Loan,
  *Matrix Computations*; Björck, *Numerical Methods for Least Squares
  Problems*; Huber, *Robust Statistics*; Koenker, *Quantile Regression*;
  Liang & Zeger (1986, GEE); Pinheiro & Bates (mixed models); Bates & Watts
  (nonlinear regression, the `nls` relative-offset criterion); McCullough
  (1998/1999, LRE and StRD). All are books or paywalled papers. Algorithms
  here come from statsmodels code and R/MATLAB documentation. Error bounds in
  §4.4 are standard results stated without page references and must be
  checked against Golub & Van Loan before W1-3 merges.
- Formulas behind MATLAB's Greenhouse–Geisser/Huynh–Feldt epsilons, the
  `mauchly` statistic, `fitglme`'s MPL/REMPL, Satterthwaite DF for `fitlme`,
  and `dwtest`'s exact p-value (Pan's algorithm). The pages opened name them
  but give no formulas. Each task that needs one starts by opening the
  primary source (bibliography first).
- How NIST computed the certified values (`lls_info` says only "16 significant
  digits … accurate up to the last digit"). The other 44 `.dat` files were not
  opened; the loader must verify their layout.
- R and lme4 source code: not read (GPL), by rule. R is used only as an
  output oracle.

## 2. Coverage inventory

Legend: **W*n*** = the wave that delivers it (§3). "SM" = statsmodels 0.15.0.

### 2.1 statsmodels, module by module

| SM component | Content | Wave | Validation reference |
|---|---|---|---|
| `regression.linear_model` OLS, WLS, GLS | fit (`pinv`, `qr`), all results statistics, `get_prediction` | W1 | NIST StRD linear (certified), SM |
| OLS/WLS/GLS `cov_type` | nonrobust, fixed scale, HC0–HC3, HAC (Bartlett, uniform), hac-panel, hac-groupsum (Driscoll–Kraay), cluster 1-way/2-way, cluster-crv3, cluster-jk | W2 | SM |
| Results tests | `t_test`, `f_test`, `wald_test`, `wald_test_terms`, `compare_f_test`, `compare_lr_test`, `compare_lm_test`, `conf_int` | W2 | SM |
| `stats.anova` | `anova_lm` types I/II/III (+robust), nested models, `AnovaRM` | W3 | NIST StRD ANOVA (one-way, certified), SM, R |
| `multivariate.manova` (for repeated measures) | Wilks, Pillai, Hotelling–Lawley, Roy | W3 | SM |
| `stats.outliers_influence` | `OLSInfluence` (all attributes), `GLMInfluence`, `MLEInfluence`, `variance_inflation_factor`, `outlier_test`, `reset_ramsey`, `summary_table` | W4 | SM, R `influence.measures` |
| `stats.diagnostic`, `stats.stattools` | het_breuschpagan, het_white, het_goldfeldquandt, het_arch, acorr_breusch_godfrey, acorr_ljungbox, acorr_lm, linear_reset, linear_rainbow, linear_harvey_collier, linear_lm, spec_white, recursive_olsresiduals, breaks_cusumolsresid, breaks_hansen, compare_cox, compare_j, compare_encompassing, durbin_watson, omni_normtest, robust_skewness, robust_kurtosis, medcouple | W4 | SM |
| `stats._diagnostic_other` | lm_test_glm, cm_test_robust, lm_robust (+subset, reparameterized), conditional moment tests | W4 | SM |
| `genmod.GLM` | IRLS, families Gaussian, Binomial, Poisson, Gamma, InverseGaussian, NegativeBinomial, Tweedie; 13 links; offset, exposure, var/freq weights; scale x2/dev/fixed; `method` newton/bfgs/…; `fit_constrained`; `score_test`; `get_prediction` | W5 | SM, R `glm` |
| GLM `fit_regularized` | elastic net (coordinate descent) | W9 | SM |
| `discrete.discrete_model` | Logit, Probit, MNLogit, Poisson, NegativeBinomial (nb1, nb2, geometric), NegativeBinomialP, GeneralizedPoisson; `get_margeff`; `fit_regularized` L1 | W6 | SM, R |
| `discrete.count_model` | ZeroInflatedPoisson, ZeroInflatedGeneralizedPoisson, ZeroInflatedNegativeBinomialP | W6 | SM |
| `discrete.truncated_model` | TruncatedLFPoisson, TruncatedLFNegativeBinomialP, TruncatedLFGeneralizedPoisson, HurdleCountModel | W6 | SM |
| `discrete.conditional_models` | ConditionalLogit, ConditionalPoisson, ConditionalMNLogit | W6 | SM |
| `discrete._diagnostics_count`, `discrete.diagnostic` | dispersion tests, zero-inflation tests, chi-square probability tests, PoissonDiagnostic | W6 | SM |
| `miscmodels.ordinal_model` | OrderedModel (probit, logit, cloglog, custom distribution) | W6 | SM, R |
| `miscmodels.tmodel` | TLinearModel (t-distributed errors, MLE) | W6 | SM |
| `base.model.GenericLikelihoodModel` | user log-likelihood with numerical score/Hessian | W6 | SM |
| `robust.RLM` | IRLS, 10 norms (LeastSquares, HuberT, RamsayE, AndrewWave, TrimmedMean, Hampel, TukeyBiweight, TukeyQuartic, StudentT, MQuantileNorm), scale `mad`/HuberScale/callable, cov H1/H2/H3, conv dev/coefs/weights/sresid | W7 | SM, MATLAB doc examples |
| `robust.scale` | mad, iqr, qn_scale, Huber (proposal 2), HuberScale, MScale, scale_tau, scale_trimmed | W7 | SM |
| `robust.resistant_linear_model` | RLMDetS (deterministic S), RLMDetSMM (MM) | W7 | SM |
| `regression.quantile_regression` | QuantReg (IRLS, 5 kernels, 3 bandwidths, robust/iid vcov) | W7 | SM; exact LP solution as a second reference |
| `regression.linear_model` GLSAR, `yule_walker`, `burg`; `feasible_gls` GLSHet, GLSHet2 | iterated feasible GLS | W8 | SM |
| `regression.recursive_ls` RecursiveLS | recursive coefficients, CUSUM/CUSUM-of-squares | W8 | SM |
| `regression.rolling` RollingOLS, RollingWLS | | W8 | SM |
| `genmod.GEE`, `OrdinalGEE`, `NominalGEE`, `cov_struct` (Independence, Exchangeable, Autoregressive, Nested, Stationary, Unstructured, GlobalOddsRatio, OrdinalIndependence, NominalIndependence, Equivalence) | robust/naive/bias-reduced cov | W8 | SM |
| `genmod.qif` QIF (Independence, Exchangeable, Autoregressive) | | W8 | SM |
| `regression.mixed_linear_model` MixedLM | REML/ML, random intercepts/slopes, variance components, BLUPs, profile likelihood, penalized fixed effects | W8 | SM, R lme4 outputs, MATLAB doc examples |
| `genmod.bayes_mixed_glm` Binomial/PoissonBayesMixedGLM | MAP (Laplace) and variational Bayes | W8 | SM |
| OLS/WLS `fit_regularized` | elastic net, L1 (coordinate descent) | W9 | SM |
| `base._penalties`, `PenalizedMixin` | L2, SCAD, SCADSmoothed, PseudoHuber, L2ConstraintsPenalty | W9 | SM |
| `gam` GLMGam, LogitGam | B-splines, cubic and cyclic cubic splines, polynomial smoothers, penalties, alpha selection by CV | W9 | SM |
| `miscmodels.nonlinls` NonlinearLS | (SM sandbox-quality); replaced by the R/MATLAB-grade NLS of W10 | W10 | NIST StRD nonlinear (certified), R `nls` |
| `regression.dimred` | SlicedInverseReg, PrincipalHessianDirections, SlicedAverageVarianceEstimation, CovarianceReduction | W11 | SM |
| `regression.process_regression` ProcessMLE (GaussianCovariance) | | W11 | SM |
| `formula` (patsy/formulaic) | formula → design matrix, contrasts, term metadata | W12 | formulaic (SM default ordering = patsy's) |

### 2.2 What R and MATLAB add beyond statsmodels

| Feature | Source | Wave |
|---|---|---|
| Aliased coefficients reported as NA (pivoted QR, `singular.ok`) | R `lm`/`glm` | W1 (option `Aliasing: AliasNA`) |
| Quasi families (`quasi`, `quasibinomial`, `quasipoisson`, variance `mu(1-mu)`, `mu`, `mu^2`, `mu^3`, `constant`) | R `glm` | W5 |
| R's GLM stopping rule and `maxit = 25` as an option | R `glm.control` | W5 |
| Analysis of deviance: sequential and multi-model, tests Chisq/LRT, Rao score, F, Cp | R `anova.glm` | W5 |
| Profile-likelihood confidence intervals for GLM and NLS | R `confint` | W5, W10 |
| `add1`/`drop1` (F, Chisq tests), `step` (AIC/BIC, both directions, scope, `steps`) | R | W9 |
| Stepwise with criteria sse/aic/bic/rsquared/adjrsquared, PEnter/PRemove, Lower/Upper, hierarchy rules | MATLAB `stepwiselm`/`stepwiseglm` | W9 |
| `robustfit` weight functions cauchy, fair, logistic, talwar, welsch (missing in SM) and its leverage-adjusted residual scaling | MATLAB | W7 |
| ANOVA sum-of-squares type `'h'` (hierarchical) | MATLAB `anova` | W3 |
| Repeated measures: between-subject factors, sphericity (Mauchly), GG/HF/LB epsilon-corrected p-values, MANOVA tables, within-model contrasts | MATLAB `fitrm`/`ranova`/`mauchly`/`epsilon` (SM `AnovaRM` has no between factors and no corrections) | W3 |
| Mixed models: covariance patterns FullCholesky, Full (log-Cholesky), Diagonal, Isotropic, CompSymm, logical pattern; ML default; DF methods residual/Satterthwaite; simulated LRT | MATLAB `fitlme` | W8 |
| GLMM by Laplace / approximate Laplace (and MPL/REMPL after bibliography) | MATLAB `fitglme`, R lme4 `glmer` (oracle) | W8 |
| `nls` Gauss–Newton with relative-offset criterion, `port` (bounds), `plinear` (Golub–Pereyra); `fitnlm` (robust options, weights) | R, MATLAB | W10 |
| Durbin–Watson test with p-value | MATLAB `dwtest` | W4 (normal approximation; exact p after Pan's paper is opened) |

### 2.3 Left out, and why

| Left out | Why | What would bring it in |
|---|---|---|
| `tsa` (ARIMA, state space, VAR…) | separate plan item (`timeseries`) | that item |
| `duration` (PHReg Cox, survival functions) | survival analysis is its own domain (censoring, risk sets); not in item 6's equivalents | a survival item |
| `imputation` (MICE, BayesGaussMI, ROS) | a missing-data framework built on models, needs a dataframe; its own design | after W12 |
| `emplike` (incl. `OLSResults.el_test`) | separate subpackage with its own theory (Owen's empirical likelihood) | its own design note |
| `sqrt_lasso` (OLS `fit_regularized(method="sqrt_lasso")`), `fit_l1_cvxopt_cp` | need a conic (SOCP) solver (SM calls cvxopt); none in the fleet | a conic solver in go-numerics/optimize |
| `robust.covariance` (MCD, OGK, Tyler, Ledoit–Wolf) | multivariate location/scatter, not regression; belongs to stats. Only the pieces RLMDetS needs are ported (W7-4) | stats multivariate design |
| `multivariate` beyond MANOVA (PCA, factor analysis, CanCorr, MultivariateLS as a public model) | not regression | stats multivariate design |
| `sandbox` (GMM, IV2SLS, …) | not public API in SM | a separate IV/GMM design |
| R `selfStart` initial values (`SSlogis` …) | the algorithms exist only as GPL R code; the documentation is not enough for parity. The model formulas themselves are provided (W10) without self-starting | — |
| MATLAB `fitglme` MPL/REMPL, `dwtest` exact p | primary sources not opened (Wolfinger & O'Connell; Pan 1968) | open them first; then a task each in W8/W4 |
| Plots (`plot_regress_exog`, influence plots, partial regression plots…) | graphics; models returns the data the plots use | go-plot side |
| Summary text tables (`summary()` exact layout) | presentation; a plain-text `Summary()` is provided, not byte parity with SM's layout | — |

## 3. Waves

Each wave is a set of PRs (one per task, §15). A wave may start only when
the layer it depends on is merged.

| Wave | Content | Depends on | Validation |
|---|---|---|---|
| W0 | repository, CI, oracle tools, dependencies in other repos | — | — |
| W1 | numerical core and least squares (OLS/WLS/GLS, four solvers, aliasing) | W0 | NIST linear (11), SM |
| W2 | inference layer: every covariance type, tests, CIs, prediction | W1 | SM |
| W3 | ANOVA family: types I/II/III/'h', nested, robust, repeated measures, MANOVA | W2 | NIST ANOVA (11), SM, R |
| W4 | diagnostics and influence | W2 | SM, R |
| W5 | GLM complete (families, links, quasi, weights, IRLS + gradient fits, deviance analysis, profile CIs) | W2, W6-1 (likelihood layer) | SM, R |
| W6 | generic likelihood layer and every discrete/ordinal/count model | W2 | SM, R |
| W7 | robust (RLM, S/MM, scale estimators, robustfit) and quantile regression | W2 | SM, MATLAB doc examples, exact LP |
| W8 | correlated and grouped data: GLSAR, FGLS, recursive/rolling LS, GEE, QIF, MixedLM, GLMM | W5, W6 | SM, R outputs |
| W9 | regularization (elastic net, L1, SCAD…), GAM, stepwise selection | W5, W6 | SM, R, MATLAB rules |
| W10 | nonlinear least squares with inference | W2 | NIST nonlinear (27), R `nls` |
| W11 | dimension-reduction regression, process regression | W1 | SM |
| W12 | formula layer | W1–W3 | formulaic/patsy |

## 4. Core architecture (W1)

### 4.1 Packages

One Go module, several packages to keep compile units and APIs readable:

```
models/            // linear models (OLS, WLS, GLS, GLSAR…), results, ANOVA, prediction
models/core        // design validation, constant detection, whitening, solvers, Term, errors
models/inference   // covariance estimators, Wald/F/t/LR/LM/score tests, multiple-testing glue
models/glm         // GLM, families, links, variance functions, IRLS, quasi
models/discrete    // likelihood layer users + discrete, count, ordinal, conditional models
models/likelihood  // generic MLE layer: Model interface, optimizers, numerical derivatives
models/robust      // norms, scale estimators, RLM, S/MM estimators, robustfit
models/quantile    // QuantReg
models/grouped     // GEE, QIF, MixedLM, BayesMixedGLM
models/penalized   // elastic net, penalties, GAM
models/selection   // add1/drop1/step/stepwise
models/nls         // nonlinear least squares
models/diagnostic  // influence and specification tests
models/dimred      // SIR, PHD, SAVE, CovarianceReduction, ProcessMLE
models/formula     // W12
```

### 4.2 Shared types

```go
// core
type Term struct {
    Name    string   // "x1", "C(g)", "x1:C(g)", "Intercept"
    Cols    [2]int   // [start, stop) in X
    Factors []string // variables involved; drives type II/'h' containment and stepwise hierarchy
    Categorical []bool // per factor; MATLAB 'h' uses continuous vs categorical
}
type Design struct {          // validated, immutable
    X    []float64; N, P int  // row-major, float64 copy of the input
    Y    []float64
    Terms []Term
    KConstant int; ConstIdx int // -1 if implicit or none
}
type Solver uint8 // SolverPinv, SolverQR, SolverPivotedQR, SolverExtended
type Aliasing uint8 // AliasMinNorm (SM), AliasError, AliasNA (R)

// inference
type CovSpec struct {
    Type   CovType // Nonrobust, FixedScale, HC0..HC3, HAC, HACPanel, HACGroupsum, Cluster, ClusterCRV3, ClusterJK
    Scale  float64 // FixedScale
    MaxLags int; Kernel Kernel // HAC*: Bartlett (default), Uniform
    Groups  [][]int // Cluster: 1 or 2 group label vectors; HACPanel: groups or time
    Time    []int
    UseCorrection *bool // per-type default as SM (§5)
    DfCorrection  *bool // cluster/hac-panel/hac-groupsum: df_resid_inference = G−1
}
type Inference struct { // embedded in every results type
    Params, BSE, Stat, PValues []float64
    Cov [][]float64; UseT bool; DfResidInference float64
}
```

Every estimator returns a results struct embedding `Inference`, so W2's
tests (`TTest`, `FTest`, `WaldTest`, `ConfInt`) are written once.

### 4.3 Validation, constants, whitening (statsmodels order)

1. Validate: X is 2-D (n, p), y has length n, everything finite (SM raises
   `MissingDataError` for non-finite exog, `base/data.py:138–139`; we refuse
   any non-finite X, y, w, Σ with `ErrData`, naming the first index).
   `n ≥ 1`, `p ≥ 1`, `n·p ≤ MaxElems` (§13).
2. Constant detection, copied exactly from `base/data.py:130–190`: per column
   `max == min` marks a constant column. One such column with nonzero mean
   gives `KConstant = 1`. With several, take the first whose mean is exactly 1,
   else the first with nonzero mean. If there is none (or only zero columns)
   and `HasConst` is nil, test for an implicit constant:
   `rank([1 | X]) == rank(X)` with NumPy's `matrix_rank` tolerance
   (`S.max·max(M,N)·eps`) gives `KConstant = 1`, `ConstIdx = −1`.
3. Whitening: OLS uses the identity. WLS multiplies row i by `√wᵢ`
   (`linear_model.py:873–877`). GLS takes Σ as a scalar (repeated), (n,) with
   `cholsigmainv = 1/√σ`, or (n,n) with `L = chol(Σ)`, `cholsigmainv = L⁻¹`
   (`linear_model.py:208–235`). For (n,n), compute `wX` by forward
   substitution `L·wX = X` rather than forming L⁻¹ (same to rounding,
   tolerance-validated).

### 4.4 Solvers

**SolverPinv** (SM default, `linear_model.py:383–400`, `tools.py:188–224`):
thin SVD `wX = U·diag(s)·Vᵀ`, cutoff `rcond·max(s)` with `rcond = 1e-15`,
`pinv = V·diag(s⁺)·Uᵀ`, `NormalizedCov = pinv·pinvᵀ`, `b = pinv·wy`. Rank is
`np.linalg.matrix_rank(np.diag(s))`, so the tolerance is `max(s)·p·eps` (a
p×p diagonal matrix), **not** `max(s)·max(n,p)·eps`. Rank < p gives the
minimum-norm solution, `RankDeficient = true`, no error. Performance variant
(§12): QR of wX first, then the SVD of the p×p R. This gives the same
singular values and pseudo-inverse to rounding when n ≫ p.

**SolverQR** (`linear_model.py:402–411`): `wX = Q·R` (go-ndarray
`QR(ModeRaw)`), `effects = Qᵀwy`, back substitution `R·b = effects[:p]`.
Rank uses `matrix_rank(R)`. Rank < p gives `ErrSingular`. **Deliberate
deviation:** SM forms `inv(RᵀR)` (l. 404), which squares κ. We form `R⁻¹` by
triangular inversion and set `NormalizedCov = R⁻¹R⁻ᵀ`. The effects vector is
kept because type-I ANOVA reads it.

**SolverPivotedQR** (R semantics): Householder QR with column pivoting
under R's documented "limited pivoting" rule (bib/r.md l. 199). A column
whose remaining norm drops below `tol·(its original norm)` (R's `tol = 1e-7`)
is moved to the end and declared aliased. With `Aliasing = AliasNA`, aliased
coefficients are NaN and flagged, and the rank is the number of unaliased
columns. Fidelity to R is by oracle only. The exact LINPACK-derived pivot
order is reproduced from outputs, never from R source.

**SolverExtended** (certification path). No float64 method can promise all
15 NIST digits on Filip and Wampler4/5. A backward-stable LS solver has a
forward error of order `κ·u + κ²·u·‖r‖/(‖X‖‖b‖)` (κ is the condition number
of the column-equilibrated design, `u = 2⁻⁵³`), and NIST classes these sets
"higher difficulty … multicollinear". Normal equations square κ and are
never offered. Algorithm, entirely in double-double (`u_dd ≈ 2⁻¹⁰⁴`, products
by error-free `TwoProd`, deterministic on every GOARCH with or without FMA):

1. `dⱼ = 2^round(log2‖wX_j‖₂)` (exact scaling), `A = wX·D`.
2. Unblocked Householder QR of `[A | wy]` in dd: for each column,
   `α = −sign(a_kk)·‖a_k:n,k‖`, `v`, `β = 2/vᵀv`, apply `I − βvvᵀ` to the
   remaining columns and to wy.
3. `effects = Qᵀwy`. Back-substitute `R·c = effects[:p]`.
4. Iterative refinement (1 to 3 steps, Björck-style, reusing the dd
   factors): residual `r = wy − A·c` accumulated in dd from the original
   data, `R·δ = (Qᵀr)[:p]`, `c ← c + δ`. Stop when `‖δ‖∞ ≤ 2⁻¹⁰⁰‖c‖∞`.
   `SSR = rᵀr` in dd.
5. `b = D·c`. Compute `R⁻¹` in dd. `NormalizedCov = D·R⁻¹R⁻ᵀ·D`.
6. Means and centered sums of squares in dd. Round each output once.

Rank: compare the dd `R` diagonal against `p·u·max|r_kk|`; rank deficiency
gives `ErrSingular`. Requires the exported dd package (§14.1).

### 4.5 Results of least squares (`linear_model.py:1957–2141`)

`DfModel = rank − KConstant`, `DfResid = n − rank` (l. 413–416).
`WResid = wy − wX·b`, `Resid = y − X·b`, `SSR = WResidᵀWResid`.
`CenteredTSS` is `Σ(wy − mean(wy))²` for OLS and `Σwᵢ(yᵢ − ȳ_w)²` for WLS.
For GLS, with `ι_w = whiten(1)` and `m = wyᵀι_w/ι_wᵀι_w`, it is
`Σ(whiten(y − m))²`. `UncenteredTSS = wyᵀwy`.
`ESS = (KConstant ? CenteredTSS : UncenteredTSS) − SSR`, `R² = 1 − SSR/TSS`
with the same switch, `R²adj = 1 − (n − KConstant)/DfResid·(1 − R²)`.
`mse_model = ESS/DfModel` and `mse_resid = SSR/DfResid` are NaN when the
denominator is 0. `F = mse_model/mse_resid` (nonrobust) or the robust Wald
form (l. 2083–2105; NaN with an implicit constant). `Scale = SSR/DfResid`.
`LLF = −n/2·log(2π) − n/2·log(SSR/n) − n/2` (l. 1066–1074); WLS adds
`½Σlog wᵢ`, GLS adds `Σlog|cholsigmainv_ii|`.
`AIC = −2LLF + 2(DfModel + KConstant)`, BIC likewise with `log n`, plus
AICc and HQIC (`info_criteria`, l. 2143–2168). `condition_number` comes from
the singular values. `UseT` defaults to true (l. 1874–1875).

### 4.6 NIST linear targets (LRE = −log10|x − c|/|c|, absolute when c = 0, capped at 15)

| Dataset | Level | n, p | SolverExtended | SolverQR / SolverPinv |
|---|---|---|---|---|
| Norris | lower | 36, 2 | 15 | ≥ 12 (gate) |
| Pontius | lower | 40, 3 | 15 | ≥ 12 (gate) |
| NoInt1 | average | 11, 1 (no intercept, uncentered R²) | 15 | ≥ 12 (gate) |
| NoInt2 | average | 3, 1 | 15 | ≥ 12 (gate) |
| Longley | higher | 16, 7 | 15 | ≥ 9 (gate) |
| Wampler1–3 | higher | 21, 6 (Wampler1 exact fit) | 15 | ≥ 9 (gate) |
| Wampler4, Wampler5 (R² = 0.0022) | higher | 21, 6 | 15 | measured, published, ≥ SM's LRE |
| Filip | higher | 82, 11 | 15 | measured, published, ≥ SM's LRE |

The targets cover every certified quantity: estimates, their SDs, residual
SD, R², regression/residual SS, mean squares and F. The float64 floors are
hypotheses consistent with the bib's "≥ 9 on easy". W1-3 measures κ of each
equilibrated design (mpmath SVD at 50 digits in the oracle) and the actual
LREs, then commits per-dataset budgets that may only rise. A miss of 15 by
SolverExtended is a defect, not a budget. The remedies, in order, are more
refinement steps, then a triple-double residual.

## 5. Inference layer (W2)

### 5.1 Covariance estimators (`linear_model.py:2213–2258, 2834–3015`; `sandwich_covariance.py`)

Notation: `e = WResid`; `P = pinv(wX)` on SolverPinv, or `B·…` via `R⁻¹` on
QR paths; `h = diag(wX·NC·wXᵀ)` (`_abat_diagonal`, row by row, never n×n);
`xu = wX ⊙ e` (scores, `_get_sandwich_arrays` l. 311–365); `H⁻¹ = NC`.
For likelihood models (W5–W8), `xu = score_obs(params)` and
`H⁻¹ = inv(hessian)` (same function, l. 342–347). Freq weights divide xu by
`√freq` except for cluster (l. 353–359).

| Type | Formula | Default correction (SM) |
|---|---|---|
| Nonrobust | `Scale·NC` | — |
| FixedScale | `scale·NC` (user scale, default 1) | — |
| HC0 | `P·diag(e²)·Pᵀ` | — |
| HC1 | HC0 with `n/DfResid·e²` | — |
| HC2 | `e²/(1−h)` | — |
| HC3 | `(e/(1−h))²` | — |
| HAC | `H⁻¹·S·H⁻¹`, `S = Γ₀ + Σ_{l=1}^{L} w_l(Γ_l + Γ_lᵀ)`, `Γ_l = Σ_t xu_t xu_{t−l}ᵀ`; Bartlett `w_l = 1 − l/(L+1)`, uniform `w_l = 1` (l. 523–566); `maxlags` required for OLS results (l. 2880); in `S_hac_simple` the default is `floor(4(T/100)^(2/9))` (l. 607) | `use_correction = False` → no factor; True → `n/(n−k)` (l. 2884, 976–978) |
| HACPanel | Newey–West within each group (`cov_nw_panel`), groups from a `groups` change or a `time` reset (l. 2957–2969) | `"hac"` (l. 2953) |
| HACGroupsum (Driscoll–Kraay) | HAC on the per-time sums of scores (`cov_nw_groupsum`) | `"cluster"` (l. 2989) |
| Cluster (1-way) | `H⁻¹·(Σ_g u_g u_gᵀ)·H⁻¹`, `u_g = Σ_{i∈g} xu_i` | factor `G/(G−1)·(n−1)/(n−k)` (l. 776–779) |
| Cluster (2-way) | `cov₀ + cov₁ − cov₀₁` (intersection groups), CRV1 only (l. 843–886) | same factor in each |
| ClusterCRV3 / ClusterJK | cluster jackknife centred at the estimate / at the jackknife mean (l. 781–792); G ≥ 2 | `(G−1)/G` (l. 794–795) |

With cluster, hac-panel and hac-groupsum, `DfCorrection` (default true) sets
`DfResidInference = G − 1` for t and F (l. 2842–2857, 3011–3013). HC2/HC3
with `hᵢ = 1` give IEEE Inf/NaN, as in SM, and are flagged.

### 5.2 Tests and intervals (`base/model.py:1749–2386`; `linear_model.py:2387–2634`)

- `TTest(L, q)`: `t = (Lb − q)/√diag(L·V·Lᵀ)`, t or normal per UseT.
- `FTest(L, q)`: `F = (Lb − q)ᵀ(L·V·Lᵀ)⁻¹(Lb − q)/r` with `r = rank(L)`. The
  inverse uses Cholesky with a pinv fallback, as SM's `f_test` does.
  df = (r, DfResidInference).
- `WaldTest`: chi² form when UseT is false. `WaldTestTerms`: one joint test
  per Term.
- `CompareFTest(restricted)`, `CompareLRTest(restricted, largeSample)`,
  `CompareLMTest(restricted, demean, useLR)`: formulas at l. 2387–2633.
- `ConfInt(alpha)`: `b ± q·BSE` with q from t(DfResidInference) or normal.
  R-style profile intervals live in W5/W10.
- `Predict(Xnew, alpha, weights)`: mean CI `x₀ᵀb ± q·√(x₀ᵀVx₀)` and
  observation PI with `+ Scale/w₀` (`_prediction.py`, `get_prediction`).

## 6. ANOVA family (W3)

### 6.1 Single model (`stats/anova.py`)

All types need `Terms`. Without them they return `ErrParam`.

- **Type I** (l. 147–179): `effects = Qᵀwy` of a non-pivoted QR in column
  order. `sum_sqₜ = Σ_{j∈cols(t)} effectsⱼ²`, `df = |cols|`,
  `F = (ss/df)/(SSR/DfResid)`. SM uses the **unwhitened** exog/endog unless
  the fit was `method="qr"` (l. 74–75, 149–152). We use the whitened design
  and test WLS/GLS against SM fitted with `method="qr"` (documented
  divergence).
- **Type II** (l. 222–291): for term t, `L1` holds the rows of I for cols(t)
  plus the cols of every term whose factor set strictly contains t's (l.
  241–248), and `L2` the containing terms only. If L2 is empty, `L12 = L1`.
  Otherwise `Q = qr(L1·V·L2ᵀ)` (complete mode), `r = |L1| − |L2|`,
  `L12 = Q[:, −r:]ᵀ·L1`. Then take the Wald F with V,
  `sum_sq = F·r·SSR/DfResid`, and sort rows by first column (l. 277). The
  completion is not unique; test F, SS and p, never L12.
- **Type III** (l. 304–346): `L12` = rows of I for cols(t), including the
  intercept row.
- **Type 'h'** (MATLAB): type II whose containment test uses all factors,
  continuous ones included (`linearmodel.anova` definition). Validate
  against the documented MATLAB examples' printed tables.
- **Robust**: `robust ∈ {HC0..HC3}` selects V (l. 13–25). Type I ignores it.
- **Nested models** (l. 446–477): type I only.
  `scale = last.Scale` unless given, `df_diff = −diff(df_resid)` (error if
  negative), `ss_diff = −diff(ssr)`, `F = ss_diff/df_diff/scale`.
- **NIST summary** (`ANOVATable`): Regression row `ESS`, `DfModel`, MS, F;
  Residual row `SSR`, `DfResid`, `Scale`. Uncentered when `KConstant = 0`.
  For NoInt1 this was checked: 200457.727272727 + 127.272727272727 = Σy².

### 6.2 NIST one-way ANOVA certification

`OneWayANOVA(groups [][]float64)` (and through type I with group dummies)
computes between/within SS in dd. Group means come from compensated sums.
Then `SS_within = Σ(y − ȳ_g)²` and `SS_between = Σ n_g(ȳ_g − ȳ)²`, both in
dd. The 11 sets (SiRstv, SmLs01–09, AtmWtAg) have up to 13 constant leading
digits (SmLs07–09: `1000000000000.4`). Target: 15 digits (LRE ≥ 15, cap)
for SS, MS, F, R² and residual SD on all 11, with the float64 type-I path
measured and published alongside. SmLs03/06/09 have 18 009 observations,
which also makes them a size test. stats' existing `FOneway` is
cross-checked on the same files; any loss there is filed against stats.

### 6.3 Repeated measures and MANOVA

- `AnovaRM` parity (l. 523–723): balanced within-subject designs only,
  aggregation optional. OLS on Sum-coded factors × subject without the full
  interaction. Each within term is tested against its term×subject error
  stratum by reduced-model SSR (`_ssr_reduced_model`). Collinear → error.
  Unbalanced → error.
- `FitRM` (MATLAB `fitrm` scope): wide data `Y` (n × r repeated measures),
  between-subject design `X`, within design (`separatemeans` default, or a
  contrast matrix C, or within Terms). `RANOVA` gives SumSq, DF, MeanSq, F,
  p, and p corrected with `F(εv₁, εv₂)` for Greenhouse–Geisser, Huynh–Feldt
  and lower-bound ε (`ranova` page). `Mauchly` and `Epsilon` follow. The ε
  and Mauchly formulas come from the primary sources, which the task opens
  first (not obtained here). `MANOVA` gives Wilks, Pillai, Hotelling–Lawley
  and Roy with F approximations, ported from SM `multivariate/manova.py`
  (`MultivariateLS`).
- Validation: SM AnovaRM fixtures; R outputs (`aov` with `Error()`
  strata; `car::Anova` for GG/HF on the same data, oracle only); the printed
  numbers of MATLAB's documented `ranova`/`mauchly` examples as small
  fixed fixtures.

## 7. Diagnostics and influence (W4)

- `OLSInfluence` (`outliers_influence.py:809–1299`): `hat_matrix_diag`,
  `resid_press`, `ess_press`, `influence`, `resid_studentized_internal/external`
  (external via the leave-one-out variance formula, never n refits),
  `cooks_distance` (+p from F), `dffits`, `dffits_internal`, `dfbeta`,
  `dfbetas`, `cov_ratio`, `det_cov_params_not_obsi`, `params_not_obsi`,
  `resid_var`, `resid_std`, `summary_frame` data. Every quantity uses the
  closed-form leave-one-out update, O(np²) total. `MLEInfluence`/
  `GLMInfluence` (l. 388–809, 1458–1640): one-step approximations exactly
  as SM. `VIF` (l. 165, with `standardize`). `OutlierTest` (Bonferroni and
  the other `multipletests` methods: needs §14.4). `ResetRamsey`.
- `diagnostic.py` and `stattools.py`: every function in §2.1, each a small
  auxiliary regression or a closed form. The time-series-flavoured ones
  (`acorr_ljungbox`, `het_arch`, `acorr_breusch_godfrey`) are here because
  they test regression residuals. `breaks_hansen` and `breaks_cusumolsresid`
  need their critical-value tables, copied from SM.
- `DurbinWatson` statistic; MATLAB `dwtest` normal-approximation p-value.
  The exact p (Pan) is left out until the paper is opened.
- Validation: SM fixtures per function; R `influence.measures` cross-check
  on the same data (R's `dffit` and `covratio` definitions agree with SM's).

## 8. GLM (W5)

### 8.1 Families, links, variance (`genmod/families/*.py`)

Families: Gaussian, Binomial, Poisson, Gamma, InverseGaussian,
NegativeBinomial(α), Tweedie(var_power, link power, `eql`), plus R quasi
families (`Quasi{Variance, Link}`, `QuasiBinomial`, `QuasiPoisson`). Quasi
families have estimated dispersion, LLF/AIC = NaN as R reports, and use the
same IRLS. Links: Identity, Log, Logit, Probit, CLogLog, LogLog, Cauchy,
LogC, Power(p), InversePower, InverseSquared, Sqrt, NegativeBinomial(α),
plus a user link (Eval, Inverse, Deriv, Deriv2, InverseDeriv). Variance
functions: constant, mu, mu², mu³, binary/binomial(n), NegativeBinomial(α),
Power(p). Every helper's clipping is copied exactly (`FLOAT_EPS = 2⁻⁵²`):
Logit/CDF links and binomial variance clip to `[eps, 1−eps]` (`links.py:161`,
`varfuncs.py:200`), Log clips to `[eps, ∞)` (`links.py:494`), deviance
ratios `y/mu` are clipped, and Binomial adds `1e-20` to denominators
(`family.py:377, 1028–1029`). Each family's `_resid_dev`, `loglike_obs`,
`starting_mu` (`(y+ȳ)/2`, Binomial `(y+0.5)/2`; l. 121, 967) and allowed
links (`safe_links`) are ported line by line.

### 8.2 IRLS (`_fit_irls`, l. 1402–1506)

1. Validate endog per family. Binomial (n,2) counts become `y/n` with
   `n_trials` = row sums (`family.py:991–1003`). Exposure is only allowed
   with the Log link (l. 421–423). `offset_exposure = offset + log(exposure)`.
2. `DfModel = rank(X) − 1`. `DfResid = n − DfModel − 1`, or
   `Σfreq − DfModel − 1` (l. 403–412). This assumes an intercept regardless;
   reproduce it.
3. Start: `mu = StartingMu(y)`, `eta = link(mu)`. Or, from `StartParams`,
   `eta = X·b₀ + offset_exposure`.
4. `scale = estimate_scale(mu)` (l. 909–962): 1 for Binomial/Poisson/NB;
   otherwise Pearson `Σ(y−mu)²·iw/V(mu)/DfResid`; `"dev"` uses
   deviance/DfResid; a float is fixed.
5. `dev₀`; NaN gives an error (l. 1437–1442). `history = [+Inf, dev₀]`.
6. Loop: `w = iweights·n_trials/(g′(mu)²V(mu))`;
   `z = eta + g′(mu)(y − mu) − offset_exposure`; WLS of z on X (`lstsq`, the
   SVD cutoff `eps·max(n,p)`; `_tools.py:97–102`); update eta and mu; append
   the deviance at the previous scale (l. 897–906); re-estimate scale; set
   `PerfectSeparation` if `allclose(mu − y, 0)`. Converged iff
   `|h[k] − h[k+1]| ≤ atol + rtol·|h[k+1]|` (`np.allclose`, l. 60–63;
   `atol = tol = 1e-8`, `rtol = 0`; criterion deviance or params).
7. Final WLS by `pinv` gives `NormalizedCov` (l. 1483–1486).
8. LLF (concentrated for Gaussian with identity, docstring l. 2113–2117;
   read `llf_scaled`), `AIC = −2LLF + 2(DfModel+1)` (l. 2156), BIC
   variants, null deviance from the intercept-only fit with the same
   offset/weights (l. 2031–2056), Pearson chi², pseudo R² (Cox–Snell,
   McFadden). UseT defaults to false (l. 1868–1869).
9. Option `StopRule: RStop` → `|dev − dev_old|/(|dev| + 0.1) < ε`,
   `MaxIter = 25` (R `glm.control`), for R parity fixtures.

### 8.3 The rest of GLM

- `Method` ≠ IRLS (`fit`, l. 1161–1300): newton/bfgs/lbfgs/nm/… via the W6
  likelihood layer, `max_start_irls = 3` IRLS warm-up steps, Hessian
  `oim` or `eim`.
- `FitConstrained(R, q)` (l. 1687–1735): reparameterize, fit, transform back.
- `ScoreTest`, `lm_test_glm`; GLM covariance types via §5.1 with
  `score_obs`.
- Analysis of deviance (R `anova.glm`): sequential over Terms and across
  nested models; tests Chisq/LRT, Rao score, F (moment dispersion from the
  largest model), Cp; the default is chi² for known dispersion, F otherwise.
- Profile-likelihood CIs (R `confint.glm`): for each coefficient, refit
  with it fixed on a grid of ±k·SE, signed root deviance
  `sign·√(ΔD/φ)`, and invert by interpolation at the normal (or t)
  quantile. The grid and interpolation follow `MASS`'s documented behaviour
  and are tolerance-validated against R outputs.
- `Predict` with `which = mean|linear|var`, delta-method CIs.

## 9. Generic likelihood layer and discrete models (W6)

### 9.1 Likelihood layer (`base/model.py`, `base/optimizer.py`)

```go
type LikelihoodModel interface {
    NParams() int
    LogLike(params []float64) float64
    LogLikeObs(params, dst []float64)
    Score(params, dst []float64)            // analytic where SM has it
    ScoreObs(params []float64, dst [][]float64)
    Hessian(params []float64, dst [][]float64)
    StartParams() []float64
}
func Fit(m LikelihoodModel, o *FitOptions) (*MLEResults, error)
```

Optimizers: `newton` (SM's own `_fit_newton`, l. 457–570: `H += 1e-10·I`
ridge, step `−H⁻¹g`, stop when `max|Δ| ≤ tol = 1e-8` or `maxiter`), plus
`bfgs`, `lbfgs`, `nm`, `powell`, `cg`, `ncg` mapped to go-numerics/optimize
`Minimize` with SM's defaults (`gtol = 1e-5`, `xtol = ftol = 1e-4` for
nm/powell, `avextol = 1e-5`; l. 635–1198). `basinhopping` and scipy
`minimize` pass-through map to optimize when available, else `ErrParam`.
Generic results: `Cov = −H⁻¹`, every §5 covariance type, LR tests against
the null model, `llnull`, `prsquared`, AIC/BIC. `GenericLikelihoodModel`
takes a user log-likelihood with numerical score and Hessian (§14.3).

### 9.2 Models (each ported from its SM class; `discrete_model.py` line ranges in §2.1)

- Binary: Logit, Probit (`newton`, `maxiter = 35`); `pred_table`;
  perfect-separation detection as SM.
- MNLogit: J−1 parameter blocks, base category 0, `K·(J−1)` params.
- Count: Poisson (offset, exposure), NegativeBinomial (loglike_method nb2,
  nb1, geometric; `bfgs` default), NegativeBinomialP (p), GeneralizedPoisson.
  Zero-inflated (ZIP, ZIGP, ZINB-P; inflation model logit/probit),
  truncated (left-truncated Poisson, NB-P, GP), Hurdle (zero model +
  truncated count model). Count diagnostics (`_diagnostics_count`).
- Conditional: ConditionalLogit, ConditionalPoisson, ConditionalMNLogit
  (group-conditional likelihoods; ConditionalLogit's recursive
  denominator, as in SM).
- Ordinal: OrderedModel (distr probit/logit/cloglog/custom; thresholds
  parameterized as the first cut plus exp increments; default fit `nm`,
  `maxiter = 500`, then the numerical Hessian).
- TLinearModel (t errors, df estimated or fixed).
- `GetMargEff(at = overall|mean|median|zero|all, method = dydx|eyex|dyex|eydx,
  atexog, dummy, count)` with delta-method SEs.
- `FitRegularized` L1 (`method = "l1"`, SM `fit_l1_slsqp`) via optimize
  SLSQP, `trim_mode` auto/size/off, `auto_trim_tol = 0.01`,
  `size_trim_tol = 1e-4`, `qc_tol = 0.03`.
- Validation: SM fixtures (each SM test class above is a case source:
  `discrete/tests/test_*.py` list the datasets and options), R `glm`/
  `MASS::polr`/`pscl` outputs for logit/probit/ordinal/zero-inflated
  (oracle only).

## 10. Robust and quantile (W7)

- RLM (`robust_linear_model.py:223–414`): start from OLS (or
  `StartParams`), scale = `mad(resid, center=0)` (`GAUSSIAN_3_4 = Φ⁻¹(3/4)`)
  or HuberScale or a callable; loop: `w = M.weights(resid/scale)`, WLS,
  update scale, append the criterion (dev = `ΣM(resid/scale)`, coefs,
  weights, sresid). Converged when `max|crit_k − crit_{k−1}| ≤ tol` or
  `iteration ≥ maxiter` (l. 32–34; `maxiter = 50`, `tol = 1e-8`). Scale 0
  stops with a warning. Covariance H1/H2/H3 exactly as `bcov_scaled`
  (l. 542–580) with `k = 1 + (DfModel+1)/n·var(ψ′)/m²`. Inference is
  normal (l. 583–584).
- Norms with SM defaults: HuberT t = 1.345, RamsayE a = 0.3, AndrewWave
  a = 1.339, TrimmedMean c = 2, Hampel (2, 4, 8), TukeyBiweight c = 4.685,
  TukeyQuartic (c = 3.61752, k = 4), StudentT (c = 2.3849, df = 4),
  MQuantileNorm(q, base), LeastSquares (`norms.py` `__init__` lines in §2.1).
  Each has ρ, ψ, ψ′ and weights.
- `RobustFit` (MATLAB): weight functions andrews 1.339, bisquare 4.685,
  cauchy 2.385, fair 1.400, huber 1.345, logistic 1.205, ols, talwar 2.795,
  welsch 2.985; `r = resid/(tune·s·√(1−h))`; `s = MAD/0.6745` with the p
  smallest absolute deviations excluded; outputs `ols_s`, `robust_s`,
  `mad_s`, `s`, `se`, `covb = inv(XᵀX)·s²`. The combination rule for `s` and
  the stopping rule are not on the page. The task opens the primary source
  (DuMouchel & O'Brien 1989, cited by MATLAB) before implementing, and
  validates against the doc example's printed output.
- Scale estimators: mad, iqr, qn_scale (O(n log n) algorithm, naive version
  as test oracle), Huber proposal 2 (c = 1.5, tol 1e-8, maxiter 30),
  HuberScale (d = 2.5), MScale, scale_tau, scale_trimmed.
- RLMDetS (`fit(h, maxiter=100, maxiter_step=5)`) and RLMDetSMM
  (`h = max(n//2+1, p+1)`): deterministic starts, S-step, MM step with
  TukeyBiweight efficiency constants from SM `robust/tools.py`.
- QuantReg (`quantile_regression.py:90–303`): IRLS from OLS with
  `pinv(XᵀX*)`, residual floor `1e-6`, `diff = max|Δβ| ≤ p_tol = 1e-6`,
  `max_iter = 1000`, cycle check every 100 after 300; bandwidth
  hsheather/bofinger/chamberlain, `h = min(std(y), IQR/1.34)·(Φ⁻¹(q+h) − Φ⁻¹(q−h))`;
  kernels epa/biw/cos/gau/par; vcov robust (sandwich with
  `d = (q/f̂)²` or `((1−q)/f̂)²`) or iid; `prsquared`. Also an **exact** mode:
  solve the LP `min Σ q·u⁺ + (1−q)·u⁻` with optimize `Linprog` (interior
  point), the true minimizer that IRLS approximates. It is the second
  reference and the R-quantreg-comparable answer.
- Validation: SM fixtures (`robust/tests/test_rlm.py` cases, the 15
  kernel×bandwidth QuantReg cases); the LP optimum as an independent check
  (objective value equal, coefficients equal when the solution is unique).

## 11. Correlated and grouped data (W8)

- GLSAR: `iterative_fit(maxiter)`, ρ by `yule_walker` (adjusted/mle) or
  `burg` (`linear_model.py:1351–1775`), whitening by the AR filter.
  GLSHet/GLSHet2 (`feasible_gls.py`).
- RecursiveLS: SM uses the state-space Kalman filter with diffuse
  initialization. We implement the algebraically equivalent recursive LS
  update (rank-one QR/Cholesky update per observation) and expose recursive
  coefficients, recursive residuals, CUSUM and CUSUM of squares with SM's
  significance bounds. Tolerance-validated against SM. The diffuse-period
  outputs must match SM's first k observations convention.
- RollingOLS/RollingWLS: window QR down/up-dates (`rolling.py`), with
  `min_nobs`, `expanding`, and per-window cov types nonrobust/HC0.
- GEE (`generalized_estimating_equations.py:1296–1478`): per-iteration
  Gauss–Newton update of the mean params; stop when
  `‖score‖ < ctol = 1e-6` after at least one dependence update; dependence
  update every `params_niter` from `first_dep_update`; `maxiter = 60`; cov
  robust (sandwich), naive, bias_reduced (Mancl–DeRouen, `_bc_covmat`);
  scale by `ddof_scale`. All cov_struct classes (§2.1) with their
  `update`/`covariance_matrix`/`covariance_matrix_solve`. OrdinalGEE and
  NominalGEE expand the response as SM does. Constraints via
  `_handle_constraint`.
- QIF (`qif.py:378`): `maxiter = 100`, `tol = 1e-6`, `gtol = 1e-4`.
- MixedLM (`mixed_linear_model.py:2249–2460`): profile (RE)ML over
  `cov_re` (Cholesky "sqrt" parameterization) and variance components with
  fixed effects and scale profiled out; optimizer sequence
  `["bfgs","lbfgs","cg"]` retried until converged (l. 2332–2405); start
  `cov_re = I`, `vcomp = 1`; boundary warning when a variance < 0.01
  (l. 2433–2437); Hessian in the natural parameterization for SEs; BLUPs
  (`random_effects`), `fe_pen`/`cov_pen`, `free` patterns. Per-group work
  uses the Woodbury identity on small `k_re` blocks; there is no sparse
  matrix anywhere. MATLAB additions: covariance patterns (FullCholesky,
  Full log-Cholesky, Diagonal, Isotropic, CompSymm, logical) as `free`
  masks or reparameterizations; ML default in the `FitLME` entry point,
  REML in the SM-compatible `MixedLM` entry point; Satterthwaite DF (open
  the source first); LRT `Compare`.
- GLMM: BinomialBayesMixedGLM and PoissonBayesMixedGLM (`fit_map` by
  Laplace/MAP, `fit_vb` mean-field VB, as SM). Plus `FitGLME` by Laplace
  approximation of the marginal likelihood (MATLAB `'Laplace'`,
  `'ApproximateLaplace'`). MPL/REMPL are left out until their source is
  opened.
- Validation: SM fixtures (GEE/MixedLM test datasets in `genmod/tests`,
  `regression/tests/test_lme.py` cases); R lme4 `lmer`/`glmer` outputs on
  the same data as a second oracle (tolerance on parameters,
  log-likelihood, SEs; optimizers differ, so it is never exact).

## 12. Regularization, GAM, selection (W9) — and performance plan for all waves

### 12.1 W9 content

- Elastic net (`elastic_net.py:227`): coordinate descent with SM's
  `maxiter = 100`, `cnvrg_tol = 1e-7`, `zero_tol = 1e-8`, `refit`,
  `check_step`, applied to OLS/WLS (closed-form coordinate updates) and to
  GLM/discrete models (one-dimensional penalized likelihood updates via the
  likelihood layer).
- Penalties (`_penalties.py`): NonePenalty, L2, L2Univariate, PseudoHuber,
  SCAD, SCADSmoothed, L2ConstraintsPenalty, with `PenalizedMixin` fitting.
- GAM (`gam/`): smoothers (BSplines, CubicSplines, CyclicCubicSplines,
  PolynomialSmoother, GenericSmoothers), univariate/multivariate penalties,
  `GLMGam` and `LogitGam` fit by penalized IRLS, alpha selection by K-fold
  CV over a path (`gam_cross_validation`). Needs a B-spline basis (§14.5).
- Selection: R `add1`/`drop1` (F and Chisq), `step` (direction both by
  default, `k = 2` or `log n`, scope upper/lower, `steps = 1000`,
  `extractAIC` semantics); MATLAB `stepwiselm`/`stepwiseglm` (criteria
  sse/aic/bic/rsquared/adjrsquared with the PEnter/PRemove defaults from
  the doc; Lower `'constant'`, Upper `'interactions'`; hierarchy rules:
  never add a term before its lower-order parts, never remove a part of a
  present higher-order term; drop linearly dependent terms).
- Validation: SM fixtures (elastic net, GAM test cases), R `step` traces
  (oracle), MATLAB doc example selections.

### 12.2 Performance plan (every wave)

Reference: statsmodels 0.15.0 on the same linux/amd64 host, single core
(`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`), timed by the
oracle's `bench` subcommand (best of 7 after warm-up), against Go
`testing.B` with `GOMAXPROCS=1`. Interleaved rounds, never at 1-min load
> 8, allocation counts asserted exactly. Each wave's last task publishes its
table in `docs/perf.md`. Slower than SM = open defect. Grids:
OLS/WLS/GLS/HC/HAC/cluster n ∈ {10², 10⁴, 10⁶} × p ∈ {5, 50, 200};
GLM/discrete to convergence on the same; RLM/QuantReg n ∈ {10³, 10⁵};
MixedLM/GEE with {10, 1000} groups × {5, 50} per group.

Levers, algorithm first:

1. Small problems are dominated by SM's Python and pandas overhead. Go wins
   without tuning; the target is zero allocations per fit after the first
   (workspace reuse).
2. Full-rank SolverPinv: QR (≈2np² flops) then the SVD of p×p R, instead of
   SM's thin SVD of n×p plus `pinv·pinvᵀ`.
3. IRLS loops (GLM, RLM, QuantReg, GEE): one workspace per fit; row
   scaling `√w ⊙ X` and working responses with go-simd unfused kernels
   (`AxpyNoFMA`, `Axpby3NoFMA`), sums by `SumPairwise` where SM uses
   `np.sum`.
4. Never form n×n: leverage row by row; leave-one-out by closed forms;
   cluster sums by one pass over sorted groups; HAC Γ_l by lagged dot
   products.
5. MixedLM/GEE: per-group small dense blocks, groups processed in
   parallel only when the user asks (single-core parity is the reference).
6. Missing kernels go to go-simd/floats or go-ndarray (§14), never into
   models.

## 13. Security and robustness (every wave)

- Untrusted sizes: refuse before allocating when `n·p > MaxElems` (default
  2²⁸ float64 = 2 GiB; 2²⁶ on 32-bit), computed in int64 with overflow
  checks. Also dense GLS Σ when `n² > MaxElems`; MNLogit `p·(J−1)`; the
  QuantReg LP size; MixedLM `Σ_g n_g·k_re` and `k_re ≤ 64`; GEE cluster
  sizes (`max n_g ≤ 2¹⁴` for the dense working covariance); HAC
  `maxlags < n`; number of groups ≤ n; GAM basis size; formula expansion
  size (W12).
- Non-finite input anywhere → `ErrData`. Negative weights → `ErrData`.
  Zero weights are allowed. Invalid endog for a family or model (binary
  not in {0,1}, counts negative, ordinal with < 2 levels…) → `ErrData`.
- Singular designs: SolverPinv min-norm (DfResid ≤ 0 gives NaN statistics,
  never a panic); QR and Extended → `ErrSingular`; AliasNA → NaN with flags.
- Iterative methods: every loop bounded by MaxIter (≤ 10⁴ enforced);
  NaN/Inf in an iterate stops with `Converged = false` and the last finite
  iterate; exp overflow in Poisson/NB/logit uses SM's clipping, so no NaN.
- No panic on any input: every exported entry point is fuzzed.
- Fuzz targets (corpus committed under `testdata/fuzz/`): NIST parser,
  fixture decoder, OLS, GLM, each discrete model family, RLM, QuantReg,
  MixedLM (small), NLS model evaluation, formula parser (W12). Invariants:
  no panic; full rank ⇒ `‖wXᵀ·WResid‖ ≤ tol·‖wX‖‖wy‖`; deviances ≥ 0; LLF
  finite when Converged.
- Bomb tests assert the refusal under `GOMEMLIMIT=256MiB` and `ulimit -v`.
- govulncheck in CI. The oracle tools are separate modules that run Python
  and R only with `-I` (Python) or `--vanilla` (R) from stdin, never
  imported by the library.

## 14. Dependencies to add elsewhere (one PR each, before the task that needs them)

1. **go-numerics/special**: export `internal/dd` as `dd` (API unchanged),
   then switch go-statistics/stats to it (one implementation). For W1-3,
   W3-2, W10.
2. **go-ndarray/linalg**: public `SolveTriangular` (scipy
   `solve_triangular`) and `InvTriangular` (LAPACK `trtri`) over the
   internal trsm. For W1.
3. **go-numerics/optimize**: numerical Hessians `ApproxHess1/2/3` and
   complex-step variants (port of SM `tools/numdiff.py`), and Jacobian of
   vector functions. For W6 (OrderedModel, GenericLikelihoodModel),
   W8, W10.
4. **go-statistics/stats**: `MultipleTests` (Bonferroni, Šidák, Holm,
   Holm–Šidák, Simes–Hochberg, Hommel, BH, BY, two-stage BH; SM
   `stats/multitest.py`). For W4 `OutlierTest`.
5. **go-numerics/interpolate** (or go-ndarray): B-spline design matrix
   (scipy `BSpline.design_matrix`). For W9 GAM.
6. **go-ndarray/linalg** (W1 SolverPivotedQR can live in models first,
   then move down): column-pivoted QR `geqp3` and `ormqr`.
7. **go-simd/floats** (only if the perf tables show the need): row-scale
   kernel and batched quadratic forms.
8. Landing (`go-statistics.github.io`), docs site and stats README
   "see also": one PR each at every release of models.

## 15. Implementation tasks, in order (one PR each)

Fixture tooling: `tools/smoracle` (statsmodels 0.15.0, NumPy and SciPy
pinned) and `tools/roracle` (Rscript with pinned R and package versions,
outputs only). Both are cobra (`SilenceUsage`, `RunE`, `osExit`) with `gen`,
`diff`, `bench` and `--mp` (mpmath 50-digit re-solve), copied from
go-numerics/optimize `tools/slsqporacle`. Committed fixtures are generated
inside `docker run --platform linux/amd64` (macOS arm64 wheels use FMA).
Inputs are generated in Go with go-ndarray `random.DefaultRNG(seed)`, NumPy's
Generator bit for bit, so fixtures store seeds and outputs. Comparison rules
(all waves): integers and decisions (rank, df, KConstant, aliasing,
convergence flags, iteration counts on the fixture set) are compared
exactly. Floating outputs use relative tolerance `64·κ_s·2⁻⁵³` for direct
solves. Iterative fits use `max(64·κ_s·2⁻⁵³, 10·tol_of_the_method)`. No bit
identity with statsmodels is claimed anywhere: LAPACK `gesdd`/`geqrf` and
OpenBLAS blocking, reduction order and FMA differ from go-ndarray. When a
case exceeds tolerance, the `--mp` reference decides who is right.

**W0** — 0.1 repository and CI skeleton (amd64, arm64, 386; 100 % coverage;
govulncheck; fuzz smoke); 0.2 `tools/smoracle`; 0.3 `tools/roracle`; 0.4
NIST loader + all 49 `.dat` files (linear 11, ANOVA 11, nonlinear 27) +
`SOURCES` with SHA-256; plus dependency PRs §14.1, §14.2.

**W1** — 1.1 `core`: validation, constant detection, whitening, Term;
1.2 SolverPinv + SolverQR + OLS results (§4.5), NIST float64 budgets
measured; 1.3 SolverExtended, NIST linear 15 digits, negative controls;
1.4 WLS and GLS (known Σ) with their TSS/LLF; 1.5 SolverPivotedQR and
AliasNA (R oracle); 1.6 W1 perf table.

**W2** — 2.1 HC0–HC3, fixed scale, robust F; 2.2 HAC, HAC-panel,
Driscoll–Kraay; 2.3 cluster 1-/2-way, CRV3, jackknife, df correction;
2.4 t/F/Wald/WaldTestTerms, compare F/LR/LM; 2.5 ConfInt, Predict;
2.6 perf.

**W3** — 3.1 ANOVA I/II/III + robust + nested; 3.2 one-way NIST ANOVA
certification (dd), FOneway cross-check; 3.3 type 'h'; 3.4 AnovaRM;
3.5 MANOVA statistics; 3.6 FitRM/RANOVA/Mauchly/epsilon (open primary
sources first); 3.7 perf.

**W4** — 4.1 OLSInfluence (closed forms); 4.2 VIF, OutlierTest (needs
§14.4), ResetRamsey; 4.3 heteroscedasticity tests; 4.4 autocorrelation
tests and Durbin–Watson; 4.5 specification and non-nested tests (RESET,
rainbow, Harvey–Collier, linear_lm, spec_white, Cox, J, encompassing);
4.6 structural-break tests and recursive residuals; 4.7 robust LM and
conditional-moment tests; 4.8 normality/robust-moment stattools.

**W6-1 before W5** — 6.1 likelihood layer: interface, SM Newton, optimize
mapping, generic MLE results with all covariance types (needs §14.3).

**W5** — 5.1 families, links and variance functions with clipping (unit
fixtures per function); 5.2 IRLS core (Gaussian, Binomial, Poisson);
5.3 Gamma, InverseGaussian, NB, Tweedie; offset/exposure/weights/(n,2);
5.4 quasi families and the R stop rule; 5.5 gradient methods, fit
constrained, score test; 5.6 GLM covariance types and GLMInfluence;
5.7 analysis of deviance; 5.8 profile CIs; 5.9 perf.

**W6** — 6.2 Logit/Probit (+pred_table, separation); 6.3 MNLogit;
6.4 Poisson, NB (nb1/nb2/geometric), NB-P, GeneralizedPoisson;
6.5 zero-inflated models; 6.6 truncated and hurdle models;
6.7 conditional models; 6.8 OrderedModel; 6.9 TLinearModel and
GenericLikelihoodModel; 6.10 marginal effects; 6.11 L1 regularized fits;
6.12 count diagnostics; 6.13 perf.

**W7** — 7.1 norms and scale estimators; 7.2 RLM (H1–H3, all conv
criteria); 7.3 RobustFit (open DuMouchel & O'Brien first); 7.4 RLMDetS
and RLMDetSMM; 7.5 QuantReg IRLS; 7.6 QuantReg exact LP; 7.7 perf.

**W8** — 8.1 GLSAR, yule_walker, burg; 8.2 GLSHet/GLSHet2; 8.3 RecursiveLS
+ CUSUM; 8.4 Rolling OLS/WLS; 8.5 GEE core + Independence/Exchangeable/
Autoregressive; 8.6 the remaining cov_structs, Ordinal/Nominal GEE,
bias-reduced cov; 8.7 QIF; 8.8 MixedLM random intercepts; 8.9 random
slopes, variance components, penalties, BLUPs; 8.10 FitLME patterns, ML
default, Satterthwaite (source first), LRT compare; 8.11 BayesMixedGLM MAP
and VB; 8.12 FitGLME Laplace; 8.13 perf.

**W9** — 9.1 elastic net for OLS/WLS; 9.2 elastic net for GLM and
discrete models; 9.3 penalties and PenalizedMixin; 9.4 B-spline and cubic
smoothers (needs §14.5); 9.5 GLMGam/LogitGam + CV path; 9.6 add1/drop1/
step; 9.7 stepwiselm/stepwiseglm; 9.8 perf.

**W10** — nonlinear least squares, `models/nls`:
10.1 model and Jacobian interface (analytic or numerical, §14.3), the 10 R
`SS*` model formulas without self-start (§2.3); 10.2 Gauss–Newton with
step halving (`minFactor`) and the relative-offset criterion with
`scaleOffset` (documented in `nls`); Levenberg–Marquardt via optimize
`LeastSquares(LM)`; bounds via `LeastSquares(TRF)` (the `port`
counterpart); 10.3 `plinear` (variable projection, Golub–Pereyra);
10.4 inference: `Cov = s²(JᵀJ)⁻¹` from the QR of J, t tests, profile CIs;
weights; MATLAB `fitnlm` robust options (via 7.1 weights); 10.5 NIST
nonlinear certification: all 27 sets from both starts; target all 11
certified digits for estimates, SDs, residual SS and residual SD from at
least Start 2. Polish: after convergence, two Gauss–Newton steps with
residuals and JᵀJ accumulated in dd. Failures from Start 1 are published,
not hidden; 10.6 perf vs R `nls` and SciPy `curve_fit`.

**W11** — 11.1 SIR, PHD, SAVE (eigh-based); 11.2 CovarianceReduction
(Grassmann optimization as SM); 11.3 ProcessMLE.

**W12** — formula layer: 12.1 parser (patsy grammar: `~ + - : * / ** ( )`,
`0`/`1`/`-1`, `I()`, `C(x, Treatment|Sum|Helmert|Poly|Diff|Reference)`),
fuzzed; 12.2 materializer over go-dataframe with patsy column ordering
(the order SM's FormulaManager imposes for formulaic, `formula/_manager.py`
l. 389–462), NA handling `drop`/`raise`, Term metadata; 12.3 MATLAB
Wilkinson strings and named specs (`'linear'`, `'interactions'`,
`'purequadratic'`, `'quadratic'`, `'polyIJK'`); 12.4 formula entry points
for every model, `AnovaRM`/`FitRM` on long data. Validation: column-for-
column equality with formulaic/patsy outputs (exact: these are 0/1/data
values and contrast matrices).

**Release cadence**: v0.1.0 after W2 (OLS/WLS/GLS with inference and NIST
certification), then a MINOR per wave via `ghrelease`, each with the
README, landing and docs PRs (§14.8).
