# Reference answers for SCORE2-CKD from RiskScorescvd (vendored), rounding disabled.
# Run from repo root: Rscript reference/score2_ckd/run_reference.R
suppressMessages(library(dplyr))
source("reference/vendor/RiskScorescvd/R/13_SCORE2_CKD_func.R")
round <- function(x, digits = 0) x  # the function rounds to 0.1%
fx <- "reference/score2_ckd/fixtures"
p <- read.csv(file.path(fx, "patients.csv"))
region_r <- c(low = "Low", moderate = "Moderate", high = "High", very_high = "Very high")
fmt <- function(x) ifelse(is.na(x), "", sprintf("%.17g", x))
run <- function(acr, trace, addon) mapply(function(r, a, g, s, b, d, t, h, e, ac, tr)
  SCORE2_CKD(r, a, g, s, b, d, t, h, e, ACR = ac, trace = tr, addon = addon, classify = FALSE),
  region_r[p$region], p$age, p$sex, p$smoker, p$sbp, p$diabetes, p$total_chol, p$hdl_chol,
  p$egfr, acr, trace) / 100
out <- data.frame(id = p$id,
  egfr = fmt(run(NA, NA, "ACR")),
  acr = fmt(run(p$acr, NA, "ACR")),
  dipstick = fmt(run(NA, p$dipstick, "dip")))
write.csv(out, file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
writeLines(c(
  "reference: RiskScorescvd SCORE2_CKD (vendored at reference/vendor/RiskScorescvd)",
  paste("package_version:", read.dcf("reference/vendor/RiskScorescvd/DESCRIPTION")[, "Version"]),
  paste("R:", R.version.string), paste("generated:", format(Sys.Date())),
  "rounding: disabled (round replaced by identity)"
), file.path(fx, "VERSION"))
