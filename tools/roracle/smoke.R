# W0 smoke case: one small lm() fit on fixed data, to prove the pipeline end
# to end (pinned image, Rscript --vanilla on stdin, JSON out by hand so that no
# package is needed). The real cases (NIST, seeded designs, aliasing with
# lm(), sandwich, car) arrive with W1 and later; their package versions are
# pinned in rocker's snapshot date below.
x <- c(1, 2, 3, 4, 5, 6)
y <- c(1.1, 1.9, 3.2, 3.9, 5.1, 6.0)
fit <- lm(y ~ x)
s <- summary(fit)
num <- function(v) paste0("[", paste(sprintf("%.17g", v), collapse = ","), "]")
cat(sprintf(
  '{"tool":"roracle","r":"%s","packages":{"stats":"%s"},"machine":"%s","cases":[{"name":"smoke-lm","seed":0,"ints":{"nobs":%d,"df_resid":%d,"rank":%d},"floats":{"coef":%s,"se":%s,"rsquared":%s}}]}\n',
  paste(R.version$major, R.version$minor, sep = "."),
  as.character(packageVersion("stats")),
  R.version$arch,
  length(y), fit$df.residual, fit$rank,
  num(coef(fit)), num(s$coefficients[, "Std. Error"]), num(s$r.squared)
))
