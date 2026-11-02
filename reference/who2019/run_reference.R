# Reference answers for the 2019 WHO models from WHORiskCalculator (CRAN).
# Run from repo root: Rscript reference/who2019/run_reference.R
suppressMessages(library(WHORiskCalculator))
fx <- "reference/who2019/fixtures"
p <- read.csv(file.path(fx, "patients.csv"), na.strings = "")  # keep "NAM" (Namibia)
fmt <- function(x) sprintf("%.17g", x)
w <- function(expr) suppressWarnings(expr)  # warnings: age > 80 notices, country fallbacks
out <- data.frame(id = p$id)
for (rc in c("region", "country")) {
  out[[paste0("lab_", rc)]] <- fmt(w(calculate_who_risk(
    p$age, p$sex, p$sbp, p$total_chol, p$smoker == 1, p$diabetes == 1,
    country = p$country, recalibration = rc, year = p$year)))
  out[[paste0("nonlab_", rc)]] <- fmt(w(calculate_who_risk_nonlab(
    p$age, p$sex, p$sbp, p$bmi, p$smoker == 1,
    country = p$country, recalibration = rc, year = p$year)))
}
write.csv(out, file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
writeLines(c(
  "reference: WHORiskCalculator (CRAN), R implementation of the 2019 WHO CVD risk models",
  paste("package_version:", as.character(packageVersion("WHORiskCalculator"))),
  paste("R:", R.version.string),
  paste("generated:", format(Sys.Date())),
  "rounding: none (package returns unrounded risks)"
), file.path(fx, "VERSION"))
