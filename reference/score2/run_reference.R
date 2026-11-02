# Compute reference answers for the SCORE2 family with the third-party R
# package RiskScorescvd (vendored at reference/vendor/RiskScorescvd).
# The package rounds risks to 0.1%; rounding is disabled here so answers can be
# compared at machine precision. Run from the repo root:
#   Rscript reference/score2/run_reference.R
suppressMessages(library(dplyr))
pkg <- "reference/vendor/RiskScorescvd"
for (f in c("rounding.R", "11_SCORE2_func.R", "15_SCORE2-Diabetes_func.R",
            "16_SCORE2-Asia-Pacific_func.R")) {
  source(file.path(pkg, "R", f))
}
round <- function(x, digits = 0) x                    # 11_SCORE2_func.R
round_to_nearest_digit <- function(number, digits = 0) number  # 15_, 16_

fx <- "reference/score2/fixtures"
region_r <- c(low = "Low", moderate = "Moderate", high = "High", very_high = "Very high")
fmt <- function(x) ifelse(is.na(x), "", sprintf("%.17g", x))

run <- function(variant, fn) {
  p <- read.csv(file.path(fx, paste0(variant, "_patients.csv")))
  risk <- fn(p) / 100
  write.csv(data.frame(id = p$id, risk = fmt(risk)),
            file.path(fx, paste0(variant, "_reference.csv")), row.names = FALSE, quote = FALSE)
  cat(variant, nrow(p), "rows\n")
}

s2 <- function(p) mapply(function(r, a, g, s, b, d, t, h)
  SCORE2(r, a, g, s, b, d, t, h, classify = FALSE),
  region_r[p$region], p$age, p$sex, p$smoker, p$sbp, p$diabetes, p$total_chol, p$hdl_chol)
run("score2", s2)
run("score2_op", s2)  # same R function; it switches to SCORE2-OP at age >= 70
run("score2_diabetes", function(p) mapply(function(r, a, g, s, b, t, h, d, da, a1c, e)
  SCORE2_Diabetes(r, a, g, s, b, t, h, d, da, a1c, e, classify = FALSE),
  region_r[p$region], p$age, p$sex, p$smoker, p$sbp, p$total_chol, p$hdl_chol,
  p$diabetes, p$age_at_diabetes_diagnosis, p$hba1c, p$egfr))

run("score2_asia_pacific", function(p) mapply(function(r, a, g, s, b, t, h, d)
  SCORE2_Asia_Pacific(r, a, g, s, b, t, h, d, classify = FALSE),
  region_r[p$region], p$age, p$sex, p$smoker, p$sbp, p$total_chol, p$hdl_chol, p$diabetes))

writeLines(c(
  "reference: RiskScorescvd (CRAN) — vendored source, see reference/vendor/RiskScorescvd/DESCRIPTION",
  paste("package_version:", read.dcf(file.path(pkg, "DESCRIPTION"))[, "Version"]),
  paste("R:", R.version.string),
  paste("dplyr:", as.character(packageVersion("dplyr"))),
  paste("generated:", format(Sys.Date())),
  "rounding: disabled (round and round_to_nearest_digit overridden with identity)"
), file.path(fx, "VERSION"))
