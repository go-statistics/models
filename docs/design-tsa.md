# Design: time-series analysis — state space, ARIMA, smoothing, VAR, unit roots, filters (`go-statistics/models/tsa`)

Status: design, no code. The go-scikit roadmap (`COVERAGE.md`, l. 63) lists
time series (statsmodels `tsa`) as "left out of the item 6 design and to be
planned as [its] own item". This document is that plan. It covers the **whole
public surface of `statsmodels.tsa` 0.15.0**, the unit-root and cointegration
tests that statsmodels does not have but MATLAB's Econometrics Toolbox does
(taken from the `arch` package, NCSA licence), and what MATLAB and Mathematica
add. It mirrors the regression design ([go-statistics/stats
`docs/design-models.md`](https://github.com/go-statistics/stats/blob/main/docs/design-models.md),
"the regression design" below): sources, inventory with exactness tier per
function, waves, oracle plan, prerequisites.

Home: this repository, as a set of packages under `models/tsa/…` in the same
module `github.com/go-statistics/models`. Reasons: `AutoReg`, `ARDL`, `adfuller`,
`coint`, `grangercausalitytests`, `VAR` and `hamilton_filter` are OLS fits and
reuse the regression core (W1) and its covariance layer (W2); MLE state-space
models reuse the likelihood layer (W6-1); `RecursiveLS` and `GLSAR`
(regression W8) are state-space and AR-filter users. A separate module would
import all of that back.

Exactness policy (user instruction, same as go-ndarray `docs/design-linalg.md`
§4): **port the reference exactly, so that it is the equality oracle; then
measure; then optimise where the table shows a gap.**

## 1. Sources

### Opened

| Id | Source | What was taken |
|---|---|---|
| B1 | go-scikit/roadmap `COVERAGE.md` (fetched 2026-11-07), l. 3, 5, 24, 63 | scope ("whole mathematical surface"), MATLAB/Mathematica columns "from knowledge, re-check", tsa left out of item 6 |
| B2 | go-statistics/stats `docs/design-models.md` (main, fetched 2026-11-07), all 908 lines | structure, packages, W0–W12, oracle tooling (`smoracle`, linux/amd64 fixtures), comparison rules, prerequisites §14 |
| B3 | go-ndarray `docs/design-linalg.md` (main), §1, §4, §10; v0.34.0 `linalg/` file list (module cache) | exactness tiers, reference-LAPACK C oracle (`refscipy`), tasks 0–20; existing files `cholesky eig eigh lstsq lu qr svd norm power` |
| B4 | statsmodels **v0.15.0** (BSD-3), shallow clone of tag (commit 278ff99) at `~/Documents/VCS/GIT/localhost/_bibliography/statsmodels-v0.15.0` | `docs/source/tsa.rst` (whole), `docs/source/statespace.rst` and `vector_ar.rst` (autosummary lists), `tsa/api.py` (whole), every top-level `def`/`class` of `tsa/**/*.py` outside tests (file:line in §3), and the passages cited in §3–§5 |
| B5 | arch **v8.0.0** (NCSA, `pyproject.toml:60`), shallow clone at `~/Documents/VCS/GIT/localhost/_bibliography/arch-v8.0.0` | `arch/unitroot/*` class/function list; `phillips_ouliaris` signature (`_phillips_ouliaris.py:134–143`) |
| B6 | go-numerics/optimize main: file tree, `numdiff.go` exports | present: `lbfgsb`, `bfgs`, `slsqp`, `trustconstr`, `powell`, `neldermead`, `basinhopping`, `brute`, `leastsquares`/`trf`; `ApproxFprime` (`numdiff.go:212`), `CheckGrad` (l. 225); **absent: TNC, numerical Hessian, complex-step derivatives** |
| B7 | go-dsp/signal main: file tree | `filter.go`, `convolve.go`, `freq.go`, `poly.go`, `spectral.go`; oracle fixtures named `LFilter`, `LFilterZIAndIC`, `Convolve`, `Deconvolve`, `Freqz`, `Detrend`, `Periodogram`, `Roots` (exported names to be confirmed at T0-3) |
| B8 | go-statistics/stats main: file tree | distributions `normal`, `studentt`, `chi2`, `fdist`, `gamma`…; **no Box–Cox file, no Jarque–Bera file** |
| B9 | go-statistics/models main: file tree | `LICENSE`, `README.md` only (regression W0 not started) |

### Not obtained (no number below comes from them)

- Books and papers named in the statsmodels docstrings: Durbin & Koopman,
  *Time Series Analysis by State Space Methods* (2012; the filter flags cite
  its chapters, `kalman_filter.py:22–29`); Harvey (1989); Hamilton (1994);
  Lütkepohl (2005) (cited by `coint_johansen`, `vecm.py:629`); Hyndman et al.
  (2008, ETS); Cleveland et al. (1990, STL); Bandara et al. (MSTL); MacKinnon
  (1994, 2010) (tables come from `adfvalues.py`); Johansen (1991);
  Osterwald-Lenum (1992); Kim & Nelson (1999); Bańbura & Modugno (2014,
  DFM-MQ); Herbst (2015, Chandrasekhar, `kalman_filter.py:31`); Koopman (1997,
  exact diffuse); Durbin & Koopman (2002, simulation smoother); Chan &
  Jeliazkov (2009, CFA); Pesaran, Shin & Smith (2001, bounds test); Monahan
  (1984, cited at `statespace/tools.py:534`); Assimakopoulos & Nikolopoulos
  (2000); Hodrick & Prescott; Baxter & King; Christiano & Fitzgerald; Hamilton
  (2018); Zivot & Andrews (1992); Leybourne & McCabe; Aparicio et al. (RUR);
  Diebold & Mariano; Broock et al. (BDS); Phillips & Ouliaris (1990). The port
  follows the code; each wave's first task opens the primary source of what it
  ports and records discrepancies (bibliography first).
- Cython kernels read only in part: `statespace/_tools.pyx.in` l. 154–273
  (Lyapunov). The other `.pyx.in` files (filters, smoothers, simulation
  smoothers, Hamilton filter, Kim smoother, ETS and Holt-Winters kernels,
  `_innovations.pyx`, `_stl.pyx` beyond l. 109–363) were located by file
  listing only. Each porting task reads its kernel completely first.
- MATLAB Econometrics Toolbox and Mathematica documentation: **not opened**
  (client-rendered vendor pages). The MATLAB and Mathematica columns in §3 are
  **from knowledge, re-check** before any MATLAB- or Mathematica-driven task.
- US Census X-13ARIMA-SEATS reference manual: not opened (§3.13).

## 2. Exactness tiers

Same three tiers as the regression design §15 and go-ndarray `design-linalg.md`
§4, stated per function in §3:

- **E0 — integers and decisions, exact against the statsmodels 0.15.0 wheel**:
  selected lags (`autolag`, `ic`), ranks (`select_coint_rank`), break dates
  (Zivot–Andrews), STL/LOESS window lengths, model-selection argmins, number of
  diffuse periods, `nobs_diffuse`, convergence flags, reject/accept at the
  stated levels. Each decision compares a value with a threshold; the Go code
  records the smallest relative margin and the gate **reports instead of
  failing** below 1e-12 (go-ndarray rule).
- **E1 — values bit-exact against the reference oracle**: statsmodels 0.15.0,
  NumPy, SciPy and arch 8.0.0 built from source against Reference-LAPACK
  v3.12.1's BLAS and LAPACK, `-O2 -ffp-contract=off` (the Cython extensions
  included), NumPy SIMD dispatch off, CORE-MATH correctly rounded `exp`,
  `log`, `pow`, `hypot` (the `refscipy` driver of go-ndarray task 0, extended
  in T0-2). The Go side ports operation for operation, writes every `x*y + z`
  as `float64(x*y) + z`, keeps statsmodels' column-major buffers (the Cython
  filters take Fortran-ordered arrays) and reduction order. Variants:
  - **E1-fft**: `acovf`/`ccovf`/`acf` default to `fft=True`
    (`_stattools.py:488, 610–612`), `bkfilter` uses `fftconvolve`
    (`bk_filter.py:98`). Bit parity holds only if go-fft reproduces NumPy's
    and SciPy's pocketfft operation order for the lengths `_next_regular`
    picks. T0-3 checks it; until then these rows are E2. The direct path
    (`fft=False`) is plain E1.
  - **E1-opt**: fits that call an optimizer (L-BFGS-B, BFGS, SLSQP, Powell,
    Nelder–Mead, trust-constr, least_squares) are E1 when go-numerics/optimize
    reproduces SciPy's iterate trace for the same objective (its `lbfgsb`,
    `bfgs`, `slsqp` already carry SciPy trace fixtures, B6) and every objective
    evaluation is E1. Iteration counts are then E0 against the oracle only.
    Where the trace diverges, the case falls to E2 and is listed in
    `testdata/budgets/` with the reason.
  - **E1-rng**: simulation (`simulate`, simulation smoothers, Monte Carlo
    error bands, ETS simulated intervals, Markov `search_reps`) uses
    go-ndarray `random`, which reproduces NumPy's `Generator` bit for bit
    (regression design §15). Same seed ⇒ same draws ⇒ E1.
  - **E1-cplx**: complex-step derivatives run the filter in `complex128`
    (§4.2). Bit parity follows go-ndarray's gfortran complex-arithmetic rule
    (`design-linalg.md` §4) and go-numerics/special `cmplx` for `exp`/`log`.
- **E2 — values against the wheel, by tolerance**: always run in addition to
  E1. Direct computations: `64·κ·2⁻⁵³` relative (regression design §15).
  Iterative fits: `max(64·κ·2⁻⁵³, 10·tol_of_the_method)` on parameters,
  log-likelihood to `1e-9` relative. When a case exceeds tolerance, the E1
  oracle (or an mpmath re-solve for closed forms) decides which side is right.

## 3. Coverage inventory

Legend. **Id** numbers the rows (207 in total, §3.15). **SM** = statsmodels
0.15.0, file:line of the definition. **MATLAB** and **Mathematica** = nearest
equivalent, **from knowledge, re-check** (none of these pages was opened); `—`
= none known. **Tier** per §2. **Wave** per §6. **Dep** = what the row needs
beyond the tsa core (prerequisites P1–P12 in §8; `reg Wn` = regression wave).

### 3.1 Descriptive statistics and tests (`tsa/stattools/_stattools.py`, `_arma_order_selection.py`)

| Id | SM | MATLAB | Mathematica | Go (`tsa`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| D1 | `acovf` (l. 488; `adjusted`, `demean`, `fft=True`, `missing`) | `autocorr` (cov via output) | `CovarianceFunction` | `ACovF` | E1-fft / E1 direct | T2 | P5 |
| D2 | `acf` (l. 847; `nlags = min(⌊10·log10 n⌋, n−1)`, l. 1003; Bartlett CI, Ljung–Box) | `autocorr` | `CorrelationFunction` | `ACF` | E1-fft; CI E1 | T2 | P5 |
| D3 | `pacf` (l. 1315; methods yw, ywa (default `ywadjusted`), ywm, ols, ols-inefficient, ols-adjusted, ld, lda, ldb, burg; l. 1318–1337) | `parcorr` | `PartialCorrelationFunction` | `PACF` | E1 (ols: E1 via reg W1) | T2 | reg W1 |
| D4 | `pacf_yw` (l. 1063) | — | — | `PACFYW` | E1 | T2 | — |
| D5 | `pacf_ols` (l. 1207) | — | — | `PACFOLS` | E1 | T2 | reg W1 |
| D6 | `pacf_burg` (l. 1135) | — | — | `PACFBurg` | E1 | T2 | — |
| D7 | `ccovf` (l. 1492; `scipy.signal.correlate`, l. 1539) | `crosscorr` | `CrossCorrelationFunction`? | `CCovF` | E1-fft | T2 | P5, P6 |
| D8 | `ccf` (l. 1562) | `crosscorr` | — | `CCF` | E1-fft | T2 | P5, P6 |
| D9 | `pccf` (l. 1792; yw/ols paths l. 1666, 1739) | — | — | `PCCF` | E1 | T2 | reg W1 |
| D10 | `q_stat` (l. 775) | `lbqtest` | `AutocorrelationTest` | `QStat` | E1 | T2 | stats χ² |
| D11 | `levinson_durbin` (l. 2029) | `levinson` (Signal) | — | `LevinsonDurbin` | E1 | T2 | — |
| D12 | `levinson_durbin_pacf` (l. 2119) | — | — | `LevinsonDurbinPACF` | E1 | T2 | — |
| D13 | `innovations_algo` (`tsa/_innovations.pyx`) | — | — | `InnovationsAlgo` | E1 | T2 | — |
| D14 | `innovations_filter` (`tsa/_innovations.pyx`) | — | — | `InnovationsFilter` | E1 | T2 | — |
| D15 | `block_jackknife` (l. 642) | — | — | `BlockJackknife` | E1 (user statistic) | T2 | — |
| D16 | `bds` (`tsa/_bds.py:171`) | — | — | `BDS` | E1 | T2 | stats normal |
| D17 | `breakvar_heteroskedasticity_test` (l. 2195) | — | — | `BreakvarHeteroskedasticity` | E1 | T2 | stats F, χ² |
| D18 | `grangercausalitytests` (l. 2368; four tests) | `gctest` | — | `GrangerCausalityTests` | E1 | T2 | reg W1, W2 |
| D19 | `diebold_mariano_test` (l. 2756; `criterion`, `harvey_adj`, `horizon`) | — | — | `DieboldMariano` | E1 | T2 | stats t, normal |
| D20 | `arma_order_select_ic` (`_arma_order_selection.py:45`; `max_ar=4`, `max_ma=2`, `ic="bic"`) | `aicbic` (loop by user) | `TimeSeriesModelFit` (auto) | `ARMAOrderSelectIC` | argmin E0; ICs E1-opt | T9 | M1 |

### 3.2 Unit roots and cointegration (`_stattools.py`, `_leybourne.py`, `adfvalues.py`, `coint_tables.py`, `vector_ar/vecm.py`; arch 8.0.0 `arch/unitroot/`)

| Id | SM / arch | MATLAB | Mathematica | Go (`tsa/unitroot`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| U1 | `adfuller` (l. 233; `maxlag = ⌈12(n/100)^{1/4}⌉` capped at `n//2 − ntrend − 1`, l. 373–375; `autolag` AIC/BIC/t-stat via `_autolag`, l. 87) | `adftest` | `UnitRootTest` ("AugmentedDickeyFuller") | `ADFuller` | lag E0; stat E1 | T3 | reg W1 |
| U2 | `mackinnonp` (`adfvalues.py:225`) | (inside `adftest`) | — | `MacKinnonP` | E1 | T3 | stats normal |
| U3 | `mackinnoncrit` (`adfvalues.py:414`) | (inside) | — | `MacKinnonCrit` | E1 | T3 | — |
| U4 | `kpss` (l. 3051; `nlags="auto"` Hobijn autolag l. 3297, `"legacy"` l. 3197–3198) | `kpsstest` | `UnitRootTest` ("KPSS")? | `KPSS` | lag E0; stat E1 | T3 | — |
| U5 | `zivot_andrews` (`ZivotAndrewsUnitRoot.run`, l. 3864; `trim=0.15`, `regression="c"`, `autolag="AIC"`) | — | — | `ZivotAndrews` | break E0; stat E1 | T3 | reg W1 |
| U6 | `range_unit_root_test` (l. 3374; `interp1d` of tables, l. 3476) | — | — | `RangeUnitRoot` | E1 | T3 | — |
| U7 | `leybourne` (`_leybourne.py:295`) | `lmctest` | — | `Leybourne` | E1-opt (ARIMA fit inside) | T9 | M1 |
| U8 | `coint` (l. 2571; `trend="c"`, `method="aeg"`, `autolag="aic"`; `mackinnonp(N=k_vars)`, l. 2716) | `egcitest` | — | `Coint` | E1 | T3 | reg W1 |
| U9 | `coint_johansen` (`vecm.py:594–723`) | `jcitest` | — | `CointJohansen` | order/sign E0; stats E1 (needs exact `dgeev`) | T3 | P1 (task 9) |
| U10 | `select_coint_rank` (`vecm.py:531`) | `jcitest` (rank) | — | `SelectCointRank` | E0 | T3 | U9 |
| U11 | `select_order` VECM (`vecm.py:36`) | — | — | `VECMSelectOrder` | E0 / E1 | T11 | V1 |
| U12 | `c_sja`, `c_sjt` (`coint_tables.py:89, 232`) | (inside `jcitest`) | — | tables | exact copy | T3 | — |
| U13 | arch `PhillipsPerron` (`unitroot.py:1004`) | `pptest` | `UnitRootTest` ("PhillipsPerron")? | `PhillipsPerron` | E1 | T3 | P10 |
| U14 | arch `DFGLS` (`unitroot.py:832`) | — | — | `DFGLS` | E1 | T3 | P10 |
| U15 | arch `VarianceRatio` (`unitroot.py:1594`) | `vratiotest` | — | `VarianceRatio` | E1 | T3 | P10 |
| U16 | arch `phillips_ouliaris` (`_phillips_ouliaris.py:134`; Za/Zt/Pu/Pz, Bartlett kernel) | — | — | `PhillipsOuliaris` | E1 | T3 | P10 |
| U17 | arch `engle_granger` (`_engle_granger.py:25`) | `egcitest` | — | `EngleGranger` | E1 | T3 | P10 |
| U18 | arch `DynamicOLS` (`cointegration.py:453`) | — | — | `DynamicOLS` | E1 | T3 | P10, reg W2 |
| U19 | arch `FullyModifiedOLS` (`cointegration.py:981`) | — | — | `FullyModifiedOLS` | E1 | T3 | P10 |
| U20 | arch `CanonicalCointegratingReg` (`cointegration.py:1139`) | — | — | `CanonicalCointegratingReg` | E1 | T3 | P10 |

Why arch: the brief asks for Phillips–Ouliaris and MATLAB has `pptest` and
`vratiotest`; statsmodels 0.15.0 has none of them (`git grep -i ouliaris` over
`statsmodels/` returns nothing). arch is NCSA-licensed (permissive), so it is
ported, not only used as an oracle. arch's own `ADF`, `KPSS` and
`ZivotAndrews` are not ported twice; they serve as second oracles for U1, U4,
U5.

### 3.3 Lag tools and deterministic terms (`tsatools.py`, `deterministic.py`)

| Id | SM | MATLAB | Mathematica | Go (`tsa`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| L1 | `lagmat` (`tsatools.py:315`) | `lagmatrix` | — | `LagMat` | exact (copies) | T1 | — |
| L2 | `lagmat2ds` (l. 520) | — | — | `LagMat2DS` | exact | T1 | — |
| L3 | `add_lag` (l. 174) | — | — | `AddLag` | exact | T1 | — |
| L4 | `add_trend` (l. 42) | — | — | `AddTrend` | E1 | T1 | — |
| L5 | `detrend` (l. 253) | `detrend` | — | `Detrend` | E1 | T1 | reg W1 |
| L6 | `vec` (l. 601) | — | `Flatten`∘`Transpose` | `Vec` | exact | T1 | — |
| L7 | `vech` (l. 626) | — | — | `Vech` | exact | T1 | — |
| L8 | `unvec` (l. 670) | — | — | `Unvec` | exact | T1 | — |
| L9 | `unvech` (l. 695) | — | — | `Unvech` | exact | T1 | — |
| L10 | `duplication_matrix` (l. 730) | — | — | `DuplicationMatrix` | exact | T1 | — |
| L11 | `elimination_matrix` (l. 750) | — | — | `EliminationMatrix` | exact | T1 | — |
| L12 | `commutation_matrix` (l. 770) | — | — | `CommutationMatrix` | exact | T1 | — |
| L13 | `unintegrate_levels` (l. 901) | — | — | `UnintegrateLevels` | E1 | T1 | — |
| L14 | `unintegrate` (l. 927) | — | — | `Unintegrate` | E1 | T1 | — |
| L15 | `freq_to_period` (l. 961) | — | — | `FreqToPeriod` | exact | T1 | — |
| L16 | `DeterministicProcess` (`deterministic.py:1049`) | — | — | `DeterministicProcess` | E1 | T1 | — |
| L17 | `TimeTrend` (l. 226) | — | — | `TimeTrend` | E1 | T1 | — |
| L18 | `Seasonality` (l. 315) | — | — | `Seasonality` | exact | T1 | — |
| L19 | `Fourier` (l. 464) | — | — | `Fourier` | E1 (libm sin/cos per §2) | T1 | — |
| L20 | `CalendarTimeTrend` (l. 877) | — | — | `CalendarTimeTrend` | E1 | T1 | §4.5 calendar |
| L21 | `CalendarSeasonality` (l. 703) | — | — | `CalendarSeasonality` | exact | T1 | §4.5 |
| L22 | `CalendarFourier` (l. 608) | — | — | `CalendarFourier` | E1 | T1 | §4.5 |
| L23 | `DeterministicTerm` (abstract, l. 42) | — | — | `DeterministicTerm` interface | — | T1 | — |

### 3.4 Lag polynomials and processes (`arima_process.py`, `varma_process.py`, `sandbox/tsa/fftarma.py`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/process`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| P1 | `ArmaProcess` (l. 679; roots, stationarity, invertibility, acf/pacf, periodogram, generate) | `arima` object, `LagOp` | `ARMAProcess` | `ARMAProcess` | E1 (roots via exact `dgeev`) | T1 | P1 task 9 |
| P2 | `arma_generate_sample` (l. 55; `signal.lfilter`, l. 127) | `simulate` | `RandomFunction` | `ARMAGenerateSample` | E1-rng | T1 | P6 |
| P3 | `arma_acovf` (l. 130; `lfiltic`/`lfilter`, l. 203–204) | — | `CovarianceFunction[proc]` | `ARMAACovF` | E1 | T1 | P6 |
| P4 | `arma_acf` (l. 211) | — | `CorrelationFunction[proc]` | `ARMAACF` | E1 | T1 | — |
| P5 | `arma_pacf` (l. 239) | — | `PartialCorrelationFunction[proc]` | `ARMAPACF` | E1 | T1 | — |
| P6 | `arma_periodogram` (l. 274; `signal.freqz`, l. 308) | — | `PowerSpectralDensity[proc]` | `ARMAPeriodogram` | E1 | T1 | P6 |
| P7 | `arma_impulse_response` (l. 322) | `armairf`, `impulse` | `ImpulseResponse`? | `ARMAImpulseResponse` | E1 | T1 | P6 |
| P8 | `arma2ma` (l. 381) | `arma2ma` | `ToInvertibleTimeSeries`? | `ARMA2MA` | E1 | T1 | P6 |
| P9 | `arma2ar` (l. 406) | `arma2ar` | — | `ARMA2AR` | E1 | T1 | P6 |
| P10 | `ar2arma` (l. 432; `scipy.optimize` least squares) | — | — | `AR2ARMA` | E1-opt | T1 | optimize `leastsq` |
| P11 | `lpol2index` (l. 499) | — | — | `LPol2Index` | exact | T1 | — |
| P12 | `index2lpol` (l. 523) | — | — | `Index2LPol` | exact | T1 | — |
| P13 | `lpol_fima` (l. 545; `gammaln`) | — | `FARIMAProcess` (coefficients) | `LPolFIMA` | E1 | T1 | special `lgamma` |
| P14 | `lpol_fiar` (l. 571) | — | — | `LPolFIAR` | E1 | T1 | special |
| P15 | `lpol_sdiff` (l. 604) | — | — | `LPolSDiff` | exact | T1 | — |
| P16 | `deconvolve` (l. 622; `lfilter` + `convolve`, l. 665–666) | `deconv` | — | `Deconvolve` | E1 | T1 | P6 |
| P17 | `VarmaPoly` (`varma_process.py:534`) | — | — | `VARMAPoly` | E1 | T1 | P1 |
| P18 | `varfilter` (l. 41) | — | — | `VARFilter` | E1 | T1 | P6 |
| P19 | `varinversefilter` (l. 137) | — | — | `VARInverseFilter` | E1 | T1 | — |
| P20 | `vargenerate` (l. 192) | — | — | `VARGenerate` | E1 | T1 | — |
| P21 | `sandbox.tsa.fftarma.ArmaFft` (listed in `tsa.rst:257–262`) | — | — | `ARMAFFT` | E1-fft | T1 | P5 |

### 3.5 Filters, decomposition, interpolation (`filters/`, `seasonal/`, `stl/`, `interp/`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/filters`, `tsa/seasonal`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| F1 | `hpfilter` (`hp_filter.py:11`, `lamb=1600`; `scipy.sparse` + `spsolve`, l. 4–5, 105–112) | `hpfilter` | — | `HPFilter` | E2 (SuperLU order not reproduced; see §5.9); E1 via banded oracle | T4 | P1 task 4 |
| F2 | `bkfilter` (`bk_filter.py:7`, `low=6, high=32, K=12`; `fftconvolve` valid, l. 98) | `bkfilter`? | — | `BKFilter` | E1-fft | T4 | P5 |
| F3 | `cffilter` (`cf_filter.py:16`, `drift=True`) | `cffilter`? | — | `CFFilter` | E1 | T4 | — |
| F4 | `hamilton_filter` (`hamilton_filter.py:17`, `h=8, p=4`; AutoReg, l. 13) | `hfilter`? | — | `HamiltonFilter` | E1 | T5 | A1 |
| F5 | `convolution_filter` (`filtertools.py:279`; `signal.convolve`) | `filter` | `ListConvolve` | `ConvolutionFilter` | E1 | T1 | P6 |
| F6 | `recursive_filter` (l. 226; `lfiltic` + `lfilter`, l. 265–269) | `filter` | `RecurrenceFilter` | `RecursiveFilter` | E1 | T1 | P6 |
| F7 | `miso_lfilter` (l. 364; `correlate` + `lfilter`) | — | — | `MISOLFilter` | E1 | T1 | P6 |
| F8 | `fftconvolve3` (l. 143) | — | — | `FFTConvolve3` | E1-fft | T1 | P5 |
| F9 | `fftconvolveinv` (l. 83) | — | — | `FFTConvolveInv` | E1-fft | T1 | P5 |
| F10 | `seasonal_decompose` (`_seasonal.py:92`; additive/multiplicative, `convolution_filter`) | `trenddecomp`? | — | `SeasonalDecompose` | E1 | T4 | F5 |
| F11 | `seasonal_mean` (`_seasonal.py:73`) | — | — | `SeasonalMean` | E1 | T4 | — |
| F12 | `STL` (`_stl.pyx:109`; `seasonal=7`, trend = smallest odd > 1.5p/(1−1.5/seasonal), `low_pass` = smallest odd > p, degrees, jumps, `robust`; inner 2/5, outer 15/0, l. 111–130, 336–339) | `trenddecomp` (STL)? | — | `STL` | E1 | T4 | — |
| F13 | `MSTL` (`mstl.py:38`; `iterate=2`, windows `7+4i`, Box–Cox `lmbda="auto"`, periods ≥ n/2 dropped; l. 117, 145–149, 219–227, 282) | — | — | `MSTL` | E1; λ E1-opt | T4 | P3 (Box–Cox) |
| F14 | `dentonm` (`interp/denton.py:86`; `numpy.linalg.solve`) | — | — | `DentonM` | E1 | T4 | P1 task 3 |

### 3.6 Autoregressions and ARDL (`ar_model.py`, `ardl/`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/ar`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| A1 | `AutoReg` (`ar_model.py:103`; trend n/c/t/ct/ctt, seasonal, exog, `hold_back`; results l. 932: predict, forecast, `get_prediction`, simulate, tests, roots) | `arima` with `MA` empty + `estimate` | `TimeSeriesModelFit[…, "AR"]` | `AutoReg` | E1 | T5 | reg W1, W2, W4 |
| A2 | `ar_select_order` (l. 2070) | — | — | `ARSelectOrder` | E0 / E1 | T5 | A1 |
| A3 | `ARDL` (`ardl/model.py:203`; fixed, causal, seasonal) | — | — | `ARDL` | E1 | T5 | reg W1, W2 |
| A4 | `ardl_select_order` (l. 1504) | — | — | `ARDLSelectOrder` | E0 / E1 | T5 | A3 |
| A5 | `UECM` (l. 1765; results l. 2244) | — | — | `UECM` | E1 | T5 | A3 |
| A6 | `UECMResults.bounds_test` (`BoundsTestResult`, l. 69) | — | — | `(*UECMResults).BoundsTest` | E1 | T5 | A7 |
| A7 | PSS critical values (`ardl/_pss_critical_values/pss.py:11`, `pss_critical_values.py`) | — | — | tables + `pssBlock` | exact copy | T5 | — |

### 3.7 ARIMA estimators, ARIMA, innovations likelihood (`arima/`, `innovations/`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/arima`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| R1 | `yule_walker` (`estimators/yule_walker.py:14`) | `aryule` (Signal) | `EstimatedProcess[…, "YuleWalker"]`? | `YuleWalker` | E1 | T5 | — |
| R2 | `burg` (`burg.py:16`) | `arburg` (Signal) | `EstimatedProcess[…, "Burg"]`? | `Burg` | E1 | T5 | — |
| R3 | `durbin_levinson` (`durbin_levinson.py:16`) | — | — | `DurbinLevinson` | E1 | T5 | — |
| R4 | `innovations` (`innovations.py:27`) | — | `EstimatedProcess[…, "Innovations"]`? | `Innovations` | E1 | T5 | — |
| R5 | `innovations_mle` (`innovations.py:103`; `scipy.optimize.minimize`, l. 317) | — | — | `InnovationsMLE` | E1-opt | T5 | optimize |
| R6 | `hannan_rissanen` (`hannan_rissanen.py:19`) | — | `EstimatedProcess[…, "HannanRissanen"]`? | `HannanRissanen` | E1 | T5 | reg W1 |
| R7 | `gls` (`gls.py:27`; regression with ARMA errors) | `regARIMA` | — | `GLS` | E1-opt | T9 | M1 |
| R8 | `statespace` (`statespace.py:16`) | `estimate` | — | `StateSpaceEstimator` | E1-opt | T9 | M1 |
| R9 | `ARIMA` (`arima/model.py:26`; `order=(0,0,0)`, `enforce_*=True`, l. 145–150; `fit(method=…)` statespace / innovations_mle / hannan_rissanen / burg / innovations / yule_walker, l. 258–263; `gls` default False for statespace, l. 267–270) | `arima` + `estimate`/`forecast`/`infer` | `ARIMAProcess`, `TimeSeriesModelFit` | `ARIMA` | E1-opt | T9 | M1, R1–R8 |
| R10 | `SARIMAXSpecification` (`specification.py:24`) | — | — | `SARIMAXSpecification` | exact | T5 | — |
| R11 | `SARIMAXParams` (`params.py:15`) | — | — | `SARIMAXParams` | E1 | T5 | — |
| R12 | `arma_innovations` (`innovations/arma_innovations.py:16`) | — | — | `ARMAInnovations` | E1 | T5 | S28 |
| R13 | `arma_loglike` (l. 111) | — | `LogLikelihood[proc, data]` | `ARMALogLike` | E1 | T5 | R12 |
| R14 | `arma_loglikeobs` (l. 140) | — | — | `ARMALogLikeObs` | E1 | T5 | R12 |
| R15 | `arma_score` (l. 183) | — | — | `ARMAScore` | E1-cplx | T5 | P2 |
| R16 | `arma_scoreobs` (l. 231) | — | — | `ARMAScoreObs` | E1-cplx | T5 | P2 |

### 3.8 State-space framework (`statespace/`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/statespace`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| S1 | `Representation`, `FrozenRepresentation` (`representation.py:86, 1164`; `initial_variance=1e6`, l. 249) | `ssm` | `StateSpaceModel` (control)? | `Representation`, `Frozen` | exact | T6 | — |
| S2 | `Initialization` (`initialization.py:17`; known, stationary, approximate_diffuse (`1e6`, l. 174–220), diffuse, fixed; per-block mixes) | `ssm` `Mean0`/`Cov0`, `dssm` | — | `Initialization` | E1 | T6 | S28 |
| S3 | `KalmanFilter` conventional (`kalman_filter.py:64`; `FILTER_CONVENTIONAL`, l. 22; `_filters/_conventional.pyx.in`) | `filter` (ssm) | `KalmanEstimator`? | `KalmanFilter` | E1 | T6 | P1 tasks 2, 3 |
| S4 | univariate filter (`FILTER_UNIVARIATE`, l. 26; `_filters/_univariate.pyx.in`) | `filter` (`Univariate`) | — | method flag | E1 | T7 | S3 |
| S5 | exact diffuse initial filter (`FILTER_EXACT_INITIAL`, l. 23; `_univariate_diffuse.pyx.in`) | `dssm` | — | method flag | E1 | T7 | S4 |
| S6 | Chandrasekhar recursions (`FILTER_CHANDRASEKHAR`, l. 31; in `_kalman_filter.pyx.in`) | — | — | method flag | E1 | T7 | S3 |
| S7 | collapsed filtering (`FILTER_COLLAPSED`, l. 27; `_representation.pyx.in`) | — | — | method flag | E1 | T7 | S3 |
| S8 | concentrated scale (`FILTER_CONCENTRATED`, l. 30) | — | — | method flag | E1 | T6 | S3 |
| S9 | inversion methods (`INVERT_UNIVARIATE`, `SOLVE_LU`, `INVERT_LU`, `SOLVE_CHOLESKY`, `INVERT_CHOLESKY`, l. 33–37; default `INVERT_UNIVARIATE \| SOLVE_CHOLESKY`, l. 296; `_filters/_inversions.pyx.in`) | — | — | `Inversion` flags | E1 | T6 | P1 task 3 |
| S10 | stability, memory conservation, timing (`STABILITY_FORCE_SYMMETRY`, `MEMORY_*`, `TIMING_*`, l. 39–61, 298–302; `tolerance=1e-19`, `loglikelihood_burn=0`, l. 306) | — | — | option flags | E1 | T6 | — |
| S11 | missing data (`reorder_missing_*`, `copy_missing_*`, `copy_index_*`, `tools.py:1585–1824`; `_tools.pyx.in:807–1153`) | (NaN handling) | — | internal | exact | T6 | — |
| S12 | `KalmanSmoother` conventional (`kalman_smoother.py:40`; `SMOOTH_CONVENTIONAL`, l. 34) | `smooth` | — | `KalmanSmoother` | E1 | T7 | S3 |
| S13 | classical smoother (`SMOOTH_CLASSICAL`, l. 35) | — | — | method flag | E1 | T7 | S12 |
| S14 | alternative smoother (`SMOOTH_ALTERNATIVE`, l. 36) | — | — | method flag | E1 | T7 | S12 |
| S15 | univariate and univariate-diffuse smoothers (`SMOOTH_UNIVARIATE`, l. 37) | — | — | method flag | E1 | T7 | S4, S5 |
| S16 | disturbance smoother, state autocovariance (`SMOOTHER_DISTURBANCE*`, `SMOOTHER_STATE_AUTOCOV`, l. 24–29) | `smooth` outputs | — | output flags | E1 | T7 | S12 |
| S17 | `SimulationSmoother` (`simulation_smoother.py:23`; `SIMULATION_STATE`, `_DISTURBANCE`, l. 18–20) | `simsmooth` | — | `SimulationSmoother` | E1-rng | T7 | S12 |
| S18 | `CFASimulationSmoother` (`cfa_simulation_smoother.py:13`) | — | — | `CFASimulationSmoother` | E1-rng | T7 | P1 tasks 3, 4 |
| S19 | `compute_smoothed_state_weights` (`tools.py:2124`) | — | — | `SmoothedStateWeights` | E1 | T7 | S12 |
| S20 | `news`, `NewsResults` (`news.py:18`, `mlemodel.py:4544`; `get_impact_dates`, `tools.py:2254`) | — | — | `News` | E1 | T7 | S19 |
| S21 | `companion_matrix` (`tools.py:142`) | — | — | `CompanionMatrix` | exact | T6 | — |
| S22 | `diff` (`tools.py:263`) | — | `Differences` | `Diff` | E1 | T6 | — |
| S23 | `is_invertible` (`tools.py:398`, `threshold=1−1e-10`) | `isStable` (LagOp) | — | `IsInvertible` | E0 / E1 | T6 | P1 task 9 |
| S24 | `constrain_stationary_univariate` (`tools.py:514`; Monahan recursion l. 540–547) | — | — | `ConstrainStationaryUnivariate` | E1 | T6 | — |
| S25 | `unconstrain_stationary_univariate` (l. 550) | — | — | `UnconstrainStationaryUnivariate` | E1 | T6 | — |
| S26 | `constrain_stationary_multivariate` (l. 912; Python l. 806; Cython `_compute_coefficients_from_multivariate_pacf`, `_tools.pyx.in:402`) | — | — | `ConstrainStationaryMultivariate` | E1 | T6 | P1 task 3 |
| S27 | `unconstrain_stationary_multivariate` (l. 1418) | — | — | `UnconstrainStationaryMultivariate` | E1 | T6 | P1 task 3 |
| S28 | `solve_discrete_lyapunov` (`tools.py:469–511`; Cython `_tools.pyx.in:154`: `getrf`/`getrs`, then `gees` + `trsyl`, `lwork = 3n`) | `dlyap` (Control) | `LyapunovSolve`/`DiscreteLyapunovSolve` | `SolveDiscreteLyapunov` (real, complex) | E1 / E1-cplx | T6 | P1 tasks 3, 7 + new |
| S29 | `validate_matrix_shape`, `validate_vector_shape` (`tools.py:1489, 1540`) | — | — | validation in `Representation` | exact | T6 | — |
| S30 | `MLEModel.fit` (`mlemodel.py:532`; `method="lbfgs"`, `maxiter=50`), `fit_constrained` (l. 790), `fix_params`, `transform_params`, `transform_jacobian` (l. 1946) | `estimate` (ssm) | `FindProcessParameters`? | `MLEModel` | E1-opt | T8 | reg W6-1, P2 |
| S31 | `loglike`, `loglikeobs` (l. 1016) | — | — | `LogLike`, `LogLikeObs` | E1 | T8 | S3 |
| S32 | `score` (l. 1597): `harvey` (l. 1479), `approx` complex-step (l. 1465) or finite difference (l. 1473); `score_obs` | — | — | `Score`, `ScoreObs` | E1-cplx | T8 | P2 |
| S33 | `hessian` (l. 1777): `oim` (l. 1863), `opg` (l. 1869), `approx` (l. 1876) | — | — | `Hessian` | E1-cplx | T8 | P2 |
| S34 | covariance types (l. 2801–3072): `opg` default (`approx` if likelihood not stored), `oim`, `approx`, `robust`, `robust_oim`, `robust_approx`, `none`, `custom`; `approx_complex_step=True` (l. 2810) | — | — | `CovType` | E1-cplx | T8 | S32, S33 |
| S35 | `get_prediction`, `get_forecast` (l. 3760, 3902), `PredictionResults` (l. 5693; `tsa/base/prediction.py:8`) | `forecast` | `TimeSeriesForecast` | `GetPrediction`, `GetForecast` | E1 | T8 | stats normal |
| S36 | `simulate` (model l. 2216, results l. 4081) | `simulate` | `RandomFunction` | `Simulate` | E1-rng | T8 | — |
| S37 | `impulse_responses` (l. 2459, 4245) | `impulse`? | — | `ImpulseResponses` | E1 | T8 | — |
| S38 | `append`, `extend`, `apply` (l. 4915, 5058, 5157) | — | — | `Append`, `Extend`, `Apply` | E1 | T8 | — |
| S39 | `test_normality`, `test_heteroskedasticity`, `test_serial_correlation` (l. 3512, 3561, 3661) | — | `AutocorrelationTest` | `TestNormality`, … | E1 | T8 | P4, D17, reg W4 |

Filter flags defined but not implemented in 0.15.0 — `FILTER_AUGMENTED`,
`FILTER_SQUARE_ROOT`, `FILTER_EXTENDED`, `FILTER_UNSCENTED`
(`kalman_filter.py:24–29`): `_filters/` holds only conventional, univariate,
univariate-diffuse and inversion kernels (directory listing). They are left out
(§3.16) and kept as reserved flag values so that the bit layout matches.

### 3.9 State-space models (`statespace/`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/statespace`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| M1 | `SARIMAX` (`sarimax.py:37`; signature l. 341–349: `order=(1,0,0)`, measurement error, time-varying regression, `mle_regression`, `simple_differencing`, `enforce_stationarity`, `enforce_invertibility`, `hamilton_representation`, `concentrate_scale`, `use_exact_diffuse`; default init approximate diffuse with burn = `k_states − k_order`, l. 669–683; `start_params` by conditional sum of squares, l. 857, 948) | `arima` (seasonal), `regARIMA` | `SARIMAProcess`, `TimeSeriesModelFit` | `SARIMAX` | E1-opt | T9 | S*, P2 |
| M2 | `UnobservedComponents` (`structural.py:48`; level/trend specs, seasonal, `freq_seasonal`, cycle, autoregressive, exog, damping) | `ssm` (user-built) | `StructuralTimeSeriesModel`? | `UnobservedComponents` | E1-opt | T9 | S* |
| M3 | `VARMAX` (`varmax.py:37`; `order=(1,0)`, `error_cov_type="unstructured"`, `measurement_error`, l. 138–141) | `varm` (VAR only) | `ARMAProcess` (vector) | `VARMAX` | E1-opt | T10 | S26, S27 |
| M4 | `DynamicFactor` (`dynamic_factor.py:33`; `error_order`, `error_var`, `error_cov_type="diagonal"`, l. 147–149) | — | — | `DynamicFactor` | E1-opt | T10 | S* |
| M5 | `DynamicFactorMQ` (`dynamic_factor_mq.py:701`; `fit(method="em", maxiter=500, tolerance=1e-6, llf_decrease_action="revert", llf_decrease_tolerance=1e-4)`, l. 2278–2284; EM l. 2456–2945) | — | — | `DynamicFactorMQ` | E1 (EM is closed-form) | T10 | S16 |
| M6 | `statespace.ExponentialSmoothing` (`statespace/exponential_smoothing.py:32`; `concentrate_scale=True`, l. 157–160) | — | — | `SSExponentialSmoothing` | E1-opt | T9 | S* |

`RecursiveLS` and `GLSAR` are in the regression design (W8). The regression
design implements `RecursiveLS` by a rank-one update rather than the Kalman
filter; once T7 lands, the Kalman path becomes the E1 reference for it and the
update path is kept only if the T8 performance table justifies it.

### 3.10 Exponential smoothing and forecasting (`holtwinters/`, `exponential_smoothing/`, `forecasting/`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/smoothing`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| E1 | `ExponentialSmoothing` (Holt-Winters, `holtwinters/model.py:116`; init `known`/`estimated`/`heuristic`/`legacy-heuristic`, l. 271; Box–Cox; 13 smoother kernels `_smoothers.py:151–373`, Cython `_exponential_smoothers.pyx`; optimizers L-BFGS-B (default, l. 1063), TNC, SLSQP, Powell, trust-constr, basinhopping, least_squares, l. 1015–1022; brute start grid, l. 719–729) | — | `ExponentialMovingAverage` (fixed α only) | `HoltWinters` | E1-opt (TNC: E2 until P2b) | T12 | P2b, P3 |
| E2 | `SimpleExpSmoothing` (l. 1415) | — | `ExponentialMovingAverage` | `SimpleExpSmoothing` | E1-opt | T12 | E1 |
| E3 | `Holt` (l. 1551) | — | — | `Holt` | E1-opt | T12 | E1 |
| E4 | `HoltWintersResults` (`results.py:20`; forecast, predict, simulate add/mul/bootstrap) | — | — | `HoltWintersResults` | E1 / E1-rng | T12 | E1 |
| E5 | `ETSModel` (`ets.py:211`; error add/mul × trend × damped × seasonal = 30 models; init known/heuristic/estimated, l. 548–594; `fit(maxiter=1000)` by `lbfgs`, l. 918–921, 1037; Cython `_ets_smooth.pyx`) | — | — | `ETSModel` | E1-opt | T12 | reg W6-1 |
| E6 | `ETSResults` (l. 1360; `get_prediction` analytic or simulated, `PredictionResults` l. 2177, `simulate_repetitions=1000`, l. 2226) | — | — | `ETSResults` | E1 / E1-rng | T12 | E5 |
| E7 | `ThetaModel` (`theta.py:64`; `deseasonalize=True`, `use_test=True`, `method="auto"`, l. 144–152; `fit(use_mle=False)`, l. 213) | — | — | `ThetaModel` | E1 (E1-opt with MLE) | T12 | F10, E2, M1 |
| E8 | `ThetaModelResults` (l. 310; forecast, intervals, components) | — | — | `ThetaModelResults` | E1 | T12 | E7 |
| E9 | `STLForecast` (`forecasting/stl.py:72`, results l. 224) | — | — | `STLForecast` | E1 / E1-opt (inner model) | T12 | F12 |

### 3.11 Vector autoregressions (`vector_ar/`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/var`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| V1 | `VAR` (`var_model.py:570`; `fit(maxlags, method="ols", ic, trend="c")`, l. 676) | `varm` + `estimate` | `ARProcess` (vector), `TimeSeriesModelFit` | `VAR` | E1; lag E0 | T11 | reg W1 |
| V2 | `VAR.select_order`, `LagOrderResults` (l. 846, 508) | `aicbic` | — | `SelectOrder` | E0 / E1 | T11 | V1 |
| V3 | `VARProcess` (l. 903): `is_stable`, `simulate_var`, `mean`, `intercept_longrun`, `ma_rep`, `orth_ma_rep`, `long_run_effects`, `acf`, `acorr`, `forecast`, `mse`, `forecast_interval`, `to_vecm` (l. 965–1357) | `simulate`, `forecast`, `isStable` | — | `VARProcess` | E1 / E1-rng | T11 | P1 tasks 3, 9 |
| V4 | `VARResults` inference (l. 1382): `cov_params`, `stderr*`, `tvalues*`, `pvalues*`, `llf`, `info_criteria` (aic, bic, hqic, fpe), `roots`, `sample_acov`, `resid_acov`, `resid_corr`, `sigma_u_mle`, `cov_ybar`, `detomega` (l. 1494–2544) | `summarize` | — | `VARResults` | E1 | T11 | V1 |
| V5 | `forecast_cov` (`method="mse"` or `"auto"`, l. 1788) | — | — | `ForecastCov` | E1 | T11 | V3 |
| V6 | `test_causality` (l. 2119; F and Wald) | `gctest` | — | `TestCausality` | E1 | T11 | stats F, χ² |
| V7 | `test_inst_causality` (l. 2249) | — | — | `TestInstCausality` | E1 | T11 | — |
| V8 | `test_whiteness` (l. 2370; Portmanteau, `adjusted`) | — | — | `TestWhiteness` | E1 | T11 | — |
| V9 | `test_normality` (l. 2464; Jarque–Bera multivariate) | — | — | `TestNormality` | E1 | T11 | — |
| V10 | `irf`, `IRAnalysis` (l. 2047; `irf.py:394`: asymptotic `cov`, `cum_effect_cov`, `lr_effect_cov`, stderrs, l. 439–1069) | `irf` (varm) | — | `IRF` | E1 | T11 | V3 |
| V11 | `irf_errband_mc`, `irf_resim`, `IRAnalysis.errband_mc`, `cum_errband_mc` (l. 1833, 1903; `irf.py:471, 988`; `repl=1000`, `burn=100`) | — | — | `IRFErrBandMC` | E1-rng | T11 | V10 |
| V12 | `fevd`, `FEVD` (l. 2068, 2581) | `fevd` (varm) | — | `FEVD` | E1 | T11 | V10 |
| V13 | `SVAR` (`svar_model.py:33`; types A, B, AB; MLE fit, l. 123; `SVARProcess`, `SVARResults` l. 547, 607) | (structural via `varm` `Covariance`)? | — | `SVAR` | E1-opt | T11 | V1, optimize |
| V14 | `VECM` (`vecm.py:823`; `fit(method="ml")`, l. 964; deterministic terms `co`/`ci`/`lo`/`li`/seasonal) | `vecm` + `estimate` | — | `VECM` | E1 | T11 | U9 |
| V15 | `VECMResults` (l. 1165): `llf`, `cov_params_*`, `stderr_*`, `tvalues_*`, `pvalues_*`, `conf_int_*`, `var_rep`, `cov_var_repr`, `ma_rep`, `orth_ma_rep`, `predict`, `test_granger_causality`, `test_inst_causality`, `irf`, `test_normality`, `test_whiteness` (l. 1461–2228) | `forecast`, `irf` (vecm) | — | `VECMResults` | E1 | T11 | V14 |
| V16 | `LocalProjections` (`local_proj.py:212`; `horizons=12`, `nw_lags`, l. 304–312; results l. 78) | — | — | `LocalProjections` | E1 | T11 | reg W2 (HAC) |
| V17 | module functions `ma_rep`, `is_stable`, `var_acf`, `forecast_cov`, `forecast`, `forecast_interval`, `var_loglike`, `orth_ma_rep`, `test_normality` (`var_model.py:50–459`) | — | — | package funcs | E1 | T11 | V3 |
| V18 | `util.varsim`, `comp_matrix`, `seasonal_dummies`, `get_var_endog` (`util.py:312, 165, 498, 20`) | — | — | internal + `VARSim` | E1 / E1-rng | T11 | — |

### 3.12 Regime switching (`regime_switching/`)

| Id | SM | MATLAB | Mathematica | Go (`tsa/regime`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| K1 | `MarkovSwitching` base (`markov_switching.py:492`; `fit(method="bfgs", maxiter=100, cov_type="approx", em_iter=5, search_reps=0)`, l. 1179–1196; EM skipped with TVTP, l. 1291; EM stop `2(ℓ_k − ℓ_{k−1})/abs(ℓ_k + ℓ_{k−1})`, l. 1407; random start search l. 1501–1579) | `msVAR` + `estimate` | `HiddenMarkovProcess` (+`EstimatedProcess`)? | `MarkovSwitching` | E1-opt; EM E1 | T13 | reg W6-1, P2 |
| K2 | Hamilton filter (`cy_hamilton_filter_log`, l. 111; `_hamilton_filter.pyx.in`) | `filter` (msVAR) | — | `HamiltonFilter` | E1 | T13 | — |
| K3 | Kim smoother (`cy_kim_smoother_log`, l. 258; `_kim_smoother.pyx.in`) | `smooth` (msVAR) | — | `KimSmoother` | E1 | T13 | K2 |
| K4 | `MarkovRegression` (`markov_regression.py:13`; trend n/c/ct/t, switching trend/exog/variance, l. 96–99) | `msVAR` (univariate) | — | `MarkovRegression` | E1-opt | T13 | K1 |
| K5 | `MarkovAutoregression` (`markov_autoregression.py:20`; `switching_ar=True`, l. 111–114) | `msVAR` | — | `MarkovAutoregression` | E1-opt | T13 | K1 |
| K6 | time-varying transition probabilities (`exog_tvtp`) | — | — | option | E1-opt | T13 | K1 |
| K7 | `MarkovSwitchingResults` (l. 1881; smoothed/filtered/predicted probabilities, expected durations, predict) | — | — | `MarkovSwitchingResults` | E1 | T13 | K3 |

### 3.13 X-13ARIMA-SEATS (`x13.py`) — external binary

| Id | SM | MATLAB | Mathematica | Go (`tsa/x13`) | Tier | Wave | Dep |
|---|---|---|---|---|---|---|---|
| X1 | `x13_arima_analysis` (l. 419) | — | — | `X13ARIMAAnalysis` | output parsing exact; numbers are the binary's | T13 | external `x13as` |
| X2 | `x13_arima_select_order` (l. 716) | — | — | `X13ARIMASelectOrder` | as X1 | T13 | external `x13as` |

statsmodels does not compute anything here: it writes a spec file, runs the
US Census binary in a subprocess (`x13.py:12, 160`; names `x13as`, `x12a`,
`x13as_ascii`, `x13as_html`, l. 28; located on `PATH` or via `X13PATH`/
`X12PATH`, l. 52–110) and parses its output. The Go port does the same and is
**not** a re-implementation of X-13. Security rules in §9.

### 3.14 Additions beyond statsmodels (MATLAB, Mathematica — from knowledge, re-check)

| Id | Feature | Source | Go | Tier | Wave |
|---|---|---|---|---|---|
| Z1 | Multivariate Markov-switching VAR (switching coefficients and covariance for k > 1 series) | MATLAB `msVAR` | `MarkovSwitchingVAR` (extends K1–K3 to vector observations) | E2 vs MATLAB doc examples; internal E1 vs K* for k = 1 | T13 |
| Z2 | Fractionally integrated ARMA (ARFIMA) exact likelihood | Mathematica `FARIMAProcess` | `ARFIMA` (state-space truncation via P13, P14) | E2 vs R `fracdiff` outputs (oracle only) | T9 |
| Z3 | Automatic model selection across AR/MA/ARMA/SARIMA by IC | Mathematica `TimeSeriesModelFit` | `AutoSelect` built on D20, M1 | E0 (selected model) | T9 |
| Z4 | Integration/stationarity test batteries over a grid of specifications | MATLAB `i10test` | `I10Test` over U1, U4, U13 | E0 / E1 | T3 |
| Z5 | User parameter-to-matrices map for state-space models (time-varying or nonlinear maps) | MATLAB `ssm(@ParamMap)` | `Model` interface with `Update(params, *Representation)` (the Go form of subclassing `MLEModel`) | E1 | T8 |

### 3.15 Counts

| Section | Rows |
|---|---|
| 3.1 D | 20 |
| 3.2 U | 20 (12 SM, 8 arch) |
| 3.3 L | 23 |
| 3.4 P | 21 |
| 3.5 F | 14 |
| 3.6 A | 7 |
| 3.7 R | 16 |
| 3.8 S | 39 |
| 3.9 M | 6 |
| 3.10 E | 9 |
| 3.11 V | 18 |
| 3.12 K | 7 |
| 3.13 X | 2 |
| 3.14 Z | 5 |
| **Total** | **207** |

Rows count functions, classes, or one option family (S4–S16 are filter and
smoother method families that statsmodels selects by flag).

### 3.16 Left out, and why

| Left out | Why | What would bring it in |
|---|---|---|
| `graphics.tsaplots`, `vector_ar/plotting.py`, every `plot_*` | graphics; the Go results expose the data | go-plot side |
| `summary()` text layout, `vector_ar/output.py` | presentation; a plain-text `Summary()` without byte parity | — |
| pandas index machinery (`tsa/base/tsa_model.py` `get_index_loc`, `get_prediction_index`; `base/datetools.py`) | pandas-specific; Go uses integer positions plus an optional calendar index (§4.5) | go-dataframe integration task after T13 |
| `from_formula` (VAR) | formula layer is regression W12 | after W12 |
| `arima_model.ARMA`, `arima_model.ARIMA`, `ar_model.AR` | removed in statsmodels: their constructors raise `NotImplementedError` (`ar_model.py:903–907`, `arima_model.py:14–62`) | — |
| `TsaDescriptive` (`descriptivestats.py:18`), `mlemodel.TSMLEModel` (`tsa/mlemodel.py:23`) | undocumented legacy wrappers, not in `tsa.rst` | — |
| `FILTER_AUGMENTED`, `FILTER_SQUARE_ROOT`, `FILTER_EXTENDED`, `FILTER_UNSCENTED` | flags without implementation in 0.15.0 (§3.8) | an implementation upstream, or a design note with its source opened |
| single-precision (`s`, `c`) instantiations of the Cython templates | reached only for float32 input; the Go kernels are generic over `float64` and `complex128` (§4.2) and can add `float32` and `complex64` without API change | a user need |
| GARCH and volatility models (arch `arch_model`) | not in statsmodels `tsa`; a separate surface (MATLAB `garch`, `egarch`, `gjr`; Mathematica `GARCHProcess`) | its own design item |
| Bayesian VAR (MATLAB `bayesvarm`), MATLAB `dtmc` Markov chains | outside statsmodels; Bayesian sampling layer not designed | separate items |
| Re-implementing X-13ARIMA-SEATS | ~100k lines of Fortran/C by the US Census; statsmodels itself wraps the binary | — |
| `x13` binary distribution | not ours to ship | — |

## 4. Architecture

### 4.1 Packages

```
models/tsa             // stattools (D*), tsatools and deterministic terms (L*)
models/tsa/process     // lag polynomials, ArmaProcess, VARMA helpers (P*)
models/tsa/filters     // HP, BK, CF, Hamilton, convolution/recursive filters (F1–F9)
models/tsa/seasonal    // seasonal_decompose, STL, MSTL, Denton (F10–F14)
models/tsa/unitroot    // ADF, KPSS, ZA, RUR, Leybourne, coint, Johansen, arch tests (U*)
models/tsa/ar          // AutoReg, ARDL, UECM, bounds test (A*)
models/tsa/arima       // ARMA estimators, innovations likelihood, ARIMA (R*)
models/tsa/statespace  // Representation, Initialization, filters, smoothers, simulation smoothers,
                       // MLEModel layer and the state-space models (S*, M*)
models/tsa/smoothing   // Holt-Winters, ETS, Theta, STLForecast (E*)
models/tsa/var         // VAR, SVAR, VECM, IRF, FEVD, local projections (V*)
models/tsa/regime      // Markov switching (K*, Z1)
models/tsa/x13         // X-13 wrapper (X*)
```

### 4.2 Scalar genericity and complex step

statsmodels computes scores, Hessians and covariance matrices by complex-step
differentiation by default (`approx_complex_step=True`, `mlemodel.py:2810`;
`complex_step=True` path through `ssm.filter`, l. 1185–1196). Its Cython
templates are instantiated for `s`, `d`, `c`, `z`. The Go kernels are written
once over

```go
type Scalar interface{ ~float64 | ~complex128 }
```

and instantiated for `float64` (fitting) and `complex128` (complex-step). This
is what makes S32–S34 E1: the complex filter runs the same operations in
complex arithmetic. Every linear-algebra call inside the filter (Cholesky,
LU, triangular solve, Schur, Sylvester) is therefore needed in both types
(P1). Non-conjugating transposes are used under complex step, as
`solve_discrete_lyapunov(…, complex_step=True)` does (`tools.py:506–511`).

### 4.3 Representation and memory

`Representation` holds `design` (k_endog × k_states), `obs_intercept`,
`obs_cov`, `transition`, `state_intercept`, `selection`, `state_cov`, each
either constant or time-varying (third dimension `nobs`), column-major as
statsmodels' Fortran arrays. Outputs follow the `MEMORY_*` flags
(`kalman_filter.py:41–58`): with `MEMORY_CONSERVE` only the last step is
kept, so the log-likelihood path allocates O(k²) not O(n·k²). Missing
observations (NaN in `endog`) are handled by the same reorder/copy routines
as statsmodels (S11), which permute rows and columns of the per-time
matrices rather than branching in the filter.

### 4.4 Model interface (Go form of `MLEModel` subclasses)

```go
type Model[T Scalar] interface {
    KParams() int
    StartParams() []float64
    ParamNames() []string
    TransformParams(unconstrained, dst []T)   // Monahan, exp/square for variances…
    UntransformParams(constrained, dst []T)
    Update(params []T, ssm *Representation[T]) error
}
```

`MLEModel` wraps a `Model` with `Fit`, `LogLike`, `Score`, `Hessian`,
covariance types and results (S30–S39). SARIMAX, UnobservedComponents,
VARMAX, DynamicFactor(MQ), statespace ExponentialSmoothing and user models
(Z5) implement it. The optimizer mapping (`lbfgs` → optimize L-BFGS-B, `bfgs`,
`nm`, `powell`, `cg`, `ncg`, `newton`) and statsmodels' defaults come from the
regression likelihood layer (regression W6-1, §9.1), not a second copy.

### 4.5 Time index

Integer positions everywhere. Calendar deterministic terms (L20–L22) and
`freq_to_period` (L15) take an optional `[]time.Time` plus a frequency code
(`D`, `W`, `M`, `Q`, `A`/`Y`, `H`…; the subset statsmodels maps in
`freq_to_period`), parsed with a fuzzed parser. No pandas-style offsets.

## 5. Algorithms that fix the port (statsmodels 0.15.0 line references)

### 5.1 Kalman filter defaults

Filter method conventional; inversion `INVERT_UNIVARIATE | SOLVE_CHOLESKY`;
stability `STABILITY_FORCE_SYMMETRY`; memory `MEMORY_STORE_ALL`; timing
`TIMING_INIT_PREDICTED`; `tolerance = 1e-19`; `loglikelihood_burn = 0`
(`kalman_filter.py:296–306`). Each flag combination statsmodels accepts is a
fixture axis in T6–T7. The univariate inversion branch (scalar `F_t`) is the
common case and is ported first.

### 5.2 Initialization

`initial_variance = 1e6` (`representation.py:249`) for approximate diffuse.
SARIMAX's default initialization (`sarimax.py:669–690`): exact diffuse only
with `use_exact_diffuse`, else approximate diffuse with
`loglikelihood_burn = k_states − k_order` when `enforce_stationarity`, the
stationary block (ARMA part) initialised by the discrete Lyapunov solution.
The number of diffuse periods under exact initialization is E0.

### 5.3 Discrete Lyapunov

Bilinear transformation to a continuous Lyapunov equation, then Sylvester
(`tools.py:499–511`). The Cython version used by the filter
(`_tools.pyx.in:154–273`) solves `(aᴴ + I) bᴴ = (a − I)` by `getrf`/`getrs`,
then forms `c = 2(a + I)⁻¹ q (aᴴ + I)⁻¹` by two further solves, then solves
the Sylvester equation via `gees` and `trsyl` with `lwork = 3n` (l. 184). The
Go port follows the Cython path (it is the one the filter calls); the Python
path is kept as the `SolveDiscreteLyapunov` public function and both are
tested against each other and against the oracle.

### 5.4 MLE fit and covariance

`fit(method="lbfgs", maxiter=50)` (`mlemodel.py:532–540`); `cov_type` default
`opg`, `approx` when the per-observation likelihood is not stored
(l. 2801); complex-step on by default (l. 2810). Scores: Harvey's analytic
score (l. 1479) for `oim`, complex step otherwise. Optimizer defaults beyond
`maxiter` are those of the regression likelihood layer (regression design
§9.1).

### 5.5 Unit-root lag rules and p-values

ADF: `maxlag = ⌈12(n/100)^{1/4}⌉`, capped at `n//2 − ntrend − 1`
(`_stattools.py:373–375`); p-values and critical values from MacKinnon's
response surfaces (`adfvalues.py:225, 414`), with `N = k_vars` for `coint`
(l. 2712–2716). KPSS: Hobijn et al. automatic bandwidth (`_kpss_autolag`,
l. 3297) or the legacy `⌈12(n/100)^{1/4}⌉` capped at `n − 1` (l. 3197–3198).
RUR: linear interpolation of tables (`interp1d`, l. 3476), ported as a plain
linear interpolation with the same out-of-range behaviour. The tables
(`adfvalues.py`, `coint_tables.py`, PSS tables, RUR tables, arch's
critical-value modules) are copied digit for digit, with a SHA-256 of each
source file in `testdata/SOURCES`.

### 5.6 Johansen

`coint_johansen` (`vecm.py:594–723`): detrend by OLS on a Vandermonde basis
of `linspace(−1, 1, n)` (l. 650–653), residuals by `pinv` (l. 655–659),
moment matrices, `eig(inv(skk)·sk0·inv(s00)·sk0ᵀ)` — a **non-symmetric**
eigenproblem (`np.linalg.eig`, l. 692) — then Cholesky normalisation
(l. 694), descending sort (l. 698–701), sign fixed by the first non-zero
element (l. 704–706), trace and max-eigenvalue statistics (l. 715–721).
Critical values exist for `det_order ∈ {−1, 0, 1}` and at most 12 variables
(l. 634–646): outside that the Go code returns NaN critical values and a
flag, as statsmodels warns. E1 needs go-ndarray's exact `dgeev` path (task 9).

### 5.7 STL and MSTL

STL defaults and iteration counts as in §3.5 F12 (`_stl.pyx:111–130,
336–339`); robustness weights updated between outer iterations (l. 341–354).
MSTL: periods sorted, windows `7 + 4i`, one pass for a single period else
`iterate = 2` passes, each STL fit on the series deseasonalised by the other
components (`mstl.py:141–175`); Box–Cox with `lmbda="auto"` estimates λ by
SciPy's `boxcox` MLE (l. 145–147) — prerequisite P3.

### 5.8 Exponential smoothing

Holt-Winters: starting values by a brute grid when `use_brute` and no start
is given (`holtwinters/model.py:719–729`), then the chosen optimizer
(default L-BFGS-B, l. 1063). ETS: 30 model variants, concentrated scale
(`ets.py:1370–1372`), `lbfgs` with `maxiter = 1000` (l. 918–921, 1037),
simulated prediction intervals with 1000 repetitions by default
(l. 2226). TNC is not in go-numerics/optimize (B6): prerequisite P2b.

### 5.9 HP filter

statsmodels builds the sparse pentadiagonal `I + λKᵀK` and calls
`scipy.sparse.linalg.spsolve` (SuperLU, `hp_filter.py:105–112`). The matrix
is symmetric positive definite and banded; the Go port solves it with
go-ndarray `SolvehBanded` (banded Cholesky, task 4). SuperLU's pivoting and
elimination order are not reproduced, so HP is **E2** against the wheel; the
E1 oracle for it is SciPy's `solveh_banded` on the same matrix, built in the
reference oracle (a documented departure from statsmodels' solver, not from
its result).

### 5.10 Markov switching

Log-space Hamilton filter and Kim smoother (`markov_switching.py:111, 258`).
Fit: `em_iter = 5` EM steps for starting values unless TVTP (l. 1291–1298),
then BFGS with `maxiter = 100`; `cov_type = "approx"`; optional random start
search (`search_reps`, l. 1501–1579), E1-rng.

## 6. Waves

Each task is one PR. A wave starts when the waves and prerequisites it names
are merged.

| Wave | Content | Depends on | Validation |
|---|---|---|---|
| T0 | oracle, harness, datasets, prerequisite PRs | regression W0 | — |
| T1 | lag tools, deterministic terms, lag polynomials, processes, basic filters (L*, P*, F5–F9) | T0, P1 tasks 2–3, P6 | oracle E1, wheel E2 |
| T2 | descriptive statistics and tests (D1–D19) | T1, reg W1, P5 | oracle, wheel, R `stats::acf`/`pacf` (oracle only) |
| T3 | unit roots and cointegration (U1–U6, U8–U10, U12–U20, Z4) | T2, reg W1–W2, P1 task 9, P10 | oracle, wheel, arch second oracle, `Matlab_results/test_coint.csv` |
| T4 | HP, BK, CF, seasonal_decompose, STL, MSTL, Denton (F1–F3, F10–F14) | T1, P1 task 4, P3, P5 | oracle, wheel, STL netlib Fortran outputs (oracle only) |
| T5 | AutoReg, ARDL, UECM, bounds test, ARMA estimators, innovations likelihood (A*, R1–R6, R10–R16, F4) | T2, reg W1, W2, W4, P2 | oracle, wheel, `arima/estimators/tests/results/*.R` |
| T6 | state-space core I: Representation, Initialization, Lyapunov, conventional filter, inversions, memory, missing data, concentrated scale, constraint transforms (S1–S3, S8–S11, S21–S29) | T1, P1 tasks 3, 7, P1-new | oracle E1 (float64 and complex128) |
| T7 | state-space core II: univariate, exact diffuse, Chandrasekhar, collapsed filters; all smoothers; simulation smoothers; weights; news (S4–S7, S12–S20) | T6 | oracle E1; `statespace/tests/results` (R KFAS exact-diffuse and smoothing CSVs, MATLAB `ssm` WPI CSVs, CFA MATLAB outputs) |
| T8 | MLE layer and results (S30–S39, Z5) | T7, reg W6-1, P2, P4 | oracle E1-opt/E1-cplx, wheel |
| T9 | SARIMAX, ARIMA (all methods), statespace ExponentialSmoothing, UnobservedComponents, `gls`, Leybourne, ARMA order selection, ARFIMA, auto selection (M1, M2, M6, R7–R9, U7, D20, Z2, Z3) | T5, T8 | oracle, wheel, `statespace/tests/results` (Stata, R) |
| T10 | VARMAX, DynamicFactor, DynamicFactorMQ (M3–M5) | T8 | oracle, wheel |
| T11 | VAR family: VAR, VARProcess/Results, tests, IRF/FEVD (asymptotic and MC), SVAR, VECM and results, local projections (V*, U11) | T3, reg W1–W2 | oracle, wheel, `vector_ar/tests/JMulTi_results`, `results/results_var.py`, `results_svar*.py`, `results_local_proj.R` |
| T12 | Holt-Winters, ETS, Theta, STLForecast (E*) | T4, T8, T9, P2b, P3 | oracle, wheel, `holtwinters/tests/results` |
| T13 | Markov switching (K*, Z1), X-13 wrapper (X*) | T8 | oracle, wheel; X* against a pinned `x13as` binary in the oracle container |

### 6.1 Tasks

**T0** — 0.1 `tools/tsaoracle` (cobra: `SilenceUsage`, `RunE`, `osExit`;
subcommands `gen`, `diff`, `bench`), copied from the regression `smoracle`
pattern, wheels pinned: statsmodels 0.15.0, arch 8.0.0, NumPy, SciPy, pandas;
generated inside `docker run --platform linux/amd64`. 0.2 reference oracle:
extend go-ndarray's `refscipy` driver (task 0) to build statsmodels 0.15.0
(including its Cython extensions) and arch 8.0.0 from source with the same
flags; record the build in `testdata/ORACLE`. 0.3 confirm the exported names
of go-dsp/signal (`LFilter`, `LFiltIC`, `Convolve`, `Correlate`, `Freqz`,
`Deconvolve`) and whether go-fft matches pocketfft (decides E1-fft vs E2 for
D1, D2, D7, D8, F2, F8, F9, P21). 0.4 datasets used by statsmodels tests
(`macrodata`, `sunspots`, Lütkepohl `e1–e6.dat`, Brockwell–Davis `dowj`,
`lake`, `oshorts`, `sbl`, `housing-data.csv`) with SHA-256 in
`testdata/SOURCES`. 0.5 prerequisite PRs P1-new, P2, P3 (§8).

**T1** — 1.1 `LagMat`, `LagMat2DS`, `AddLag`, `AddTrend`, `Detrend`, vec/vech
family, duplication/elimination/commutation, `Unintegrate*`. 1.2
deterministic terms and `DeterministicProcess`, calendar index parser
(fuzzed). 1.3 lag polynomials, `ARMAProcess` (roots after P1 task 9),
`ARMAGenerateSample`, `ARMAACovF/ACF/PACF`, periodogram, impulse response,
`ARMA2MA/AR`, `AR2ARMA`, `LPol*`, `Deconvolve`. 1.4 VARMA helpers,
`ARMAFFT`. 1.5 `ConvolutionFilter`, `RecursiveFilter`, `MISOLFilter`,
`FFTConvolve3/Inv`. 1.6 perf table.

**T2** — 2.1 `ACovF` (direct and FFT), `ACF` with Bartlett/Ljung–Box,
`QStat`. 2.2 `PACF` all eleven method spellings, `PACFYW/OLS/Burg`,
`LevinsonDurbin(PACF)`. 2.3 `CCovF`, `CCF`, `PCCF`. 2.4 innovations
algorithm and filter. 2.5 `BlockJackknife`, `BDS`, `Breakvar…`,
`DieboldMariano`, `GrangerCausalityTests`. 2.6 perf.

**T3** — 3.1 MacKinnon tables and `ADFuller` (all autolag modes). 3.2 `KPSS`
(auto and legacy), `RangeUnitRoot`. 3.3 `ZivotAndrews`. 3.4 `Coint`. 3.5
`CointJohansen`, `SelectCointRank`, tables. 3.6 arch `PhillipsPerron`,
`DFGLS`, `VarianceRatio` with arch's covariance kernels. 3.7 arch
`PhillipsOuliaris`, `EngleGranger`. 3.8 `DynamicOLS`, `FullyModifiedOLS`,
`CanonicalCointegratingReg`. 3.9 `I10Test`. 3.10 perf.

**T4** — 4.1 `HPFilter` (banded), `BKFilter`, `CFFilter`. 4.2
`SeasonalDecompose`, `SeasonalMean`. 4.3 `STL` (read `_stl.pyx` and the
Cleveland et al. paper first). 4.4 `MSTL` (needs P3). 4.5 `DentonM`. 4.6 perf.

**T5** — 5.1 `AutoReg` and results (needs reg W4 for the residual tests),
`ARSelectOrder`, `HamiltonFilter`. 5.2 `ARDL`, `ARDLSelectOrder`. 5.3 `UECM`,
PSS tables, bounds test. 5.4 `SARIMAXSpecification`, `SARIMAXParams`. 5.5
`YuleWalker`, `Burg`, `DurbinLevinson`, `Innovations`, `HannanRissanen`. 5.6
innovations likelihood R12–R16 and `InnovationsMLE`. 5.7 perf.

**T6** — 6.1 `Representation` (constant and time-varying, validation). 6.2
`SolveDiscreteLyapunov` (Cython path and Python path), `Initialization` all
types and block mixes. 6.3 conventional filter, `float64`, univariate
inversion. 6.4 the four other inversion methods, stability, memory flags,
timing, concentrated scale. 6.5 missing data. 6.6 `complex128`
instantiation and complex-step parity test. 6.7 companion matrix, `Diff`,
`IsInvertible`, constraint transforms. 6.8 perf.

**T7** — 7.1 univariate filter. 7.2 exact diffuse (read Koopman 1997 and
`_univariate_diffuse.pyx.in` first). 7.3 Chandrasekhar (Herbst 2015 first),
collapsed. 7.4 smoothers conventional/classical/alternative/univariate(+diffuse),
disturbance smoother, state autocovariance. 7.5 `SimulationSmoother`
(Durbin–Koopman 2002 first). 7.6 `CFASimulationSmoother`. 7.7 smoothed state
weights, `News`. 7.8 perf.

**T8** — 8.1 `Model` interface, `MLEModel.Fit` via the regression likelihood
layer, parameter transforms, `FixParams`, `FitConstrained`. 8.2 `LogLike*`,
`Score*` (Harvey, complex step, finite difference). 8.3 `Hessian`
(oim/opg/approx), covariance types. 8.4 prediction/forecast/simulate/impulse
responses. 8.5 `Append`/`Extend`/`Apply`. 8.6 residual diagnostics (needs P4,
reg W4 `acorr_ljungbox`). 8.7 perf.

**T9** — 9.1 `SARIMAX` (all options, start params by CSS). 9.2 `ARIMA` and
its six methods, `GLS`, `StateSpaceEstimator`. 9.3 `UnobservedComponents`.
9.4 `SSExponentialSmoothing`. 9.5 `Leybourne`, `ARMAOrderSelectIC`,
`AutoSelect`. 9.6 `ARFIMA` (open Mathematica's and `fracdiff`'s definitions
first). 9.7 perf.

**T10** — 10.1 `VARMAX`. 10.2 `DynamicFactor`. 10.3 `DynamicFactorMQ` with EM
(open Bańbura & Modugno first). 10.4 perf.

**T11** — 11.1 `VAR` fit and `SelectOrder`. 11.2 `VARProcess` methods and
forecasting. 11.3 inference, causality, whiteness, normality. 11.4 IRF
(asymptotic) and FEVD. 11.5 Monte Carlo error bands. 11.6 `SVAR`. 11.7
`VECM`, `VECMResults`, `VECMSelectOrder`. 11.8 `LocalProjections`. 11.9 perf.

**T12** — 12.1 Holt-Winters kernels (13) and initialization methods. 12.2
Holt-Winters fitting with every optimizer (TNC after P2b), Box–Cox,
`SimpleExpSmoothing`, `Holt`, results and simulation. 12.3 `ETSModel` (30
variants) and results with simulated intervals (open Hyndman et al. first).
12.4 `ThetaModel`. 12.5 `STLForecast`. 12.6 perf.

**T13** — 13.1 Hamilton filter and Kim smoother. 13.2 `MarkovRegression`,
`MarkovAutoregression`, EM, start search, TVTP. 13.3 `MarkovSwitchingVAR`
(Z1). 13.4 X-13 wrapper with output parser (fuzzed) and the §9 rules. 13.5
perf.

**Release cadence**: tsa releases ride the models module's MINOR releases
(`ghrelease`), one MINOR per tsa wave from T2 on, each with README, landing
(`go-statistics.github.io`) and docs-site PRs.

## 7. Oracle and fixture plan

- **Wheel fixtures (E0, E2)**: `tools/tsaoracle gen` in a linux/amd64
  container with pinned wheels (macOS arm64 wheels use FMA). Inputs are
  generated in Go with go-ndarray `random.DefaultRNG(seed)` (NumPy's
  `Generator` bit for bit), so fixtures store seeds and outputs, gzipped JSON
  as in go-dsp/signal `testdata/oracle/`.
- **Reference oracle (E1)**: the T0-2 build. Every E1 row has at least one
  fixture per option axis that is compared bit for bit (`math.Float64bits`
  equality, NaN payload ignored).
- **Third-party results already in statsmodels' test tree**, used as
  cross-checks at E2: JMulTi outputs for VAR/VECM
  (`vector_ar/tests/JMulTi_results/*`), MATLAB Johansen results
  (`vector_ar/tests/Matlab_results/test_coint.csv`), R scripts for ARMA
  estimators (`arima/estimators/tests/results/*.R`), R filter results
  (`filters/tests/results/filter.R`), R local projections
  (`vector_ar/tests/results/results_local_proj.R`), Lütkepohl data
  (`vector_ar/tests/results/e1.dat`–`e6.dat`), and the state-space set
  (`statespace/tests/results/`: R KFAS exact-diffuse and smoothing CSVs,
  Stata SARIMAX/VARMAX/DFM CSVs, MATLAB `ssm` WPI CSVs, CFA simulation
  smoother outputs). These are data and expected numbers, not code; CSV and
  text files are copied with their source path and SHA-256. Pickles in that
  tree (`sm-0.9-sarimax.pkl`) are never loaded: they are executable.
- **arch** as a second oracle for U1, U4, U5 and the only one for U13–U20.
- **X-13**: a pinned `x13as` build inside the oracle container; the Go
  wrapper's parsed outputs must equal statsmodels' parsed outputs for the
  same spec (E0 on the parse).
- **Decision gates (E0)** record the smallest relative margin; below 1e-12
  they report instead of failing (§2).

## 8. Prerequisites in other repositories (one PR each, before the task that needs them)

| Id | Repo | What | Needed by |
|---|---|---|---|
| P1 | go-ndarray `linalg` | tasks of `design-linalg.md` §10: **2** (reference BLAS/LAPACK auxiliaries), **3** (`LU`, `LUFactor`, `LUSolve`, `ChoFactor`, `ChoSolve`, `SolveTriangular`), **4** (`SolvehBanded`, `CholeskyBanded`), **7** (`Schur`, z-variants), **9** (`Eig` on the exact `dgeev` path) — each in `float64` and `complex128` | T1, T3, T4, T6 |
| P1-new | go-ndarray `linalg` | **not in tasks 0–20**: export `SolveSylvester` (SciPy `solve_sylvester`: real/complex Schur + `trsyl`) and `SolveDiscreteLyapunov`/`SolveContinuousLyapunov` (SciPy `_solvers.py`), real and complex; `trsyl` exists internally in the task 7 file plan (`trsen.go`) | T6 |
| P2 | go-numerics/optimize | numerical Hessians `ApproxHess1/2/3`, complex-step `ApproxFprimeCS`/`ApproxHessCS`, Jacobian of vector functions (port of statsmodels `tools/numdiff.py`); only `ApproxFprime` and `CheckGrad` exist (`numdiff.go:212, 225`). Same item as regression design §14.3 — one PR serves both | T5, T8, T13 |
| P2b | go-numerics/optimize | `TNC` minimizer (SciPy `_tnc`), absent from the tree (B6) | T12 (Holt-Winters `method="TNC"`) |
| P3 | go-statistics/stats | `Boxcox`, `BoxcoxLLF`, `BoxcoxNormmax` (SciPy `stats.boxcox` with MLE λ); no Box–Cox file in the tree (B8). `InvBoxcox`/`Boxcox` scalar forms go to go-numerics/special if absent there (check at T0-5) | T4 (MSTL), T12 |
| P4 | go-statistics/stats | `JarqueBera` (and the skew/kurtosis it uses) for `test_normality`; no file of that name in the tree (B8) — check `stats` exports at T0-5 before filing | T8, T11 |
| P5 | go-fft/fft | pocketfft-order parity for real FFTs of `_next_regular` lengths, or a documented E2 (decided at T0-3) | T1, T2, T4 |
| P6 | go-dsp/signal | `LFilter` with `zi`, `LFiltIC`, `Convolve`, `Correlate`, `Freqz`, `Deconvolve` exported as SciPy names (fixtures exist, B7; names checked at T0-3) | T1, T2 |
| P7 | go-ndarray `random` | multivariate normal draws in NumPy `Generator` order (`multivariate_normal` with SVD/Cholesky method as NumPy) | T8, T11 |
| P8 | go-statistics/models regression | W1 (OLS, `pinv`/QR solvers), W2 (HAC, covariance types, Wald/F), W4 (`acorr_ljungbox`, `het_arch`), W6-1 (likelihood layer and optimizer mapping), W8 (`GLSAR`, `yule_walker`, `RecursiveLS`) | T2, T3, T5, T8, T11 |
| P9 | go-numerics/special | `lgamma` (`gammaln`) and `cmplx` `exp`/`log` for E1-cplx | T1, T6 |
| P10 | (in this repo, T3) | arch covariance kernels (Bartlett, Parzen, quadratic spectral…) and automatic bandwidth (`auto_bandwidth`, `unitroot.py:1967`) ported from arch with the tests that need them | T3 |
| P11 | go-statistics/stats | `MultipleTests` (regression design §14.4) — used only by results helpers that adjust p-values | T11 |
| P12 | docs | README, landing (`go-statistics.github.io`), docs site: one PR each per release | every release |

P10 is listed so that the arch-derived code has an owner; it is not an
external dependency.

## 9. Security and robustness

- Sizes refused before allocation, in int64 with overflow checks:
  `nobs·k_states²` per stored matrix family (unless `MEMORY_CONSERVE`),
  `nobs·k_endog²`, `k_states ≤ 2¹⁰`, VAR `neqs·p ≤` the OLS design bound of
  the regression design §13, IRF/FEVD `periods·neqs²`, MC `repl·periods·neqs²`,
  STL `n·(outer+1)·inner`, MSTL `len(periods)·n`, ETS simulation
  `repetitions·horizon`, Markov `k_regimes^(order+1)·nobs`. 32-bit builds use
  limits a quarter as large.
- Lag and window arguments validated against `nobs` (`maxlag < nobs`, odd STL
  windows ≥ 3, `period ≥ 2`, `trim ∈ (0, ½)`); statsmodels' errors become
  `ErrParam`.
- Non-finite input: NaN in `endog` is a missing observation only in the
  state-space models (as statsmodels); elsewhere non-finite input → `ErrData`.
- Every iterative method bounded (`maxiter ≤ 10⁴`), NaN iterates stop with
  `Converged = false`.
- **X-13 wrapper**: the binary path comes from an explicit option or
  `X13PATH`; `PATH` search only when the caller opts in. The spec text is
  generated from typed Go values; any user-supplied string field is checked
  against the X-13 spec grammar's allowed characters (no `{`, `}`, newline or
  quote) before writing — the injection surface statsmodels leaves open by
  formatting user strings into the spec. The process runs with a context
  timeout, an empty environment plus `PATH`, a fresh `os.MkdirTemp` working
  directory removed afterwards, stdout/stderr capped. The output parser is
  fuzzed.
- Fuzz targets (corpus in `testdata/fuzz/`): calendar/frequency parser,
  fixture decoder, X-13 output parser, `ACF`/`PACF`, `STL`, Kalman filter
  with random NaN masks, `SARIMAX.LogLike`, `ADFuller`, `CointJohansen`,
  `HoltWinters` objective. Invariant: no panic, finite log-likelihood for
  finite input and positive-definite covariances.
- Bomb tests assert the refusal under `GOMEMLIMIT=256MiB` and `ulimit -v`.
- govulncheck in CI; oracle tools are separate modules.

## 10. Performance plan

Reference: statsmodels 0.15.0 wheel on the same linux/amd64 host, single core
(`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), `tsaoracle bench` (best of
7 after warm-up) against Go `testing.B` with `GOMAXPROCS=1`; interleaved
rounds, never at 1-min load > 8; allocation counts asserted exactly. Each
wave's last task publishes its table in `docs/perf.md`; slower than
statsmodels = open defect.

Grids: `ACF`/`STL`/`HPFilter` n ∈ {10³, 10⁵, 10⁶}; Kalman log-likelihood for
SARIMAX(1,1,1)(1,1,1,12) and UnobservedComponents (local linear trend +
seasonal 12) at n ∈ {10², 10⁴}; filter with `k_states ∈ {2, 10, 50}`; full
SARIMAX fit; VAR k ∈ {3, 10} × p ∈ {2, 8}; DynamicFactorMQ EM on the
statsmodels test panel; ETS fit on the 30 variants; Markov 2–3 regimes.

Levers, algorithm first: statsmodels already picks the fast filter variants
by flag (univariate, Chandrasekhar, collapsed, memory conservation); the
port keeps them identical so the E1 reference and the fast path are the same
code. Then: zero allocations per log-likelihood call after the first
(workspace reuse across optimizer iterations, the dominant cost of every
fit); small dense k×k kernels via go-simd unfused routines where the tables
show a gap, with any missing kernel added to go-simd or go-ndarray, never
here.
