# HEART reference answers from RiskScorescvd (vendored at reference/vendor/RiskScorescvd).
# Its HEART() sums risk-factor vectors, so it is called one row at a time.
# Run from the repo root: Rscript reference/heart/run_reference.R
suppressMessages(library(dplyr))
pkg <- "reference/vendor/RiskScorescvd"
source(file.path(pkg, "R", "01_HEART_func.R"))
fx <- "reference/heart/fixtures"
p <- read.csv(file.path(fx, "patients.csv"))
score <- mapply(function(sy, en, ar, st, a, d, s, ht, hl, fh, ath, tn, g)
  suppressWarnings(HEART(sy, en, ar, st, a, d, s, ht, hl, fh, ath, tn, g, classify = FALSE)),
  p$symptoms, p$ecg_normal, p$abn_repolarisation, p$ecg_st_depression, p$age, p$diabetes, p$smoker,
  p$hypertension, p$hyperlipidaemia, p$family_history, p$atherosclerosis, p$hstni, p$sex)
write.csv(data.frame(id = p$id, score = ifelse(is.na(score), "", score)),
          file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
writeLines(c(
  "reference: RiskScorescvd (CRAN), HEART() - vendored source, see reference/vendor/RiskScorescvd/DESCRIPTION",
  paste("package_version:", read.dcf(file.path(pkg, "DESCRIPTION"))[, "Version"]),
  paste("R:", R.version.string),
  paste("dplyr:", as.character(packageVersion("dplyr"))),
  paste("generated:", format(Sys.Date())),
  "rounding: round(score, 0) on integer points, i.e. none"
), file.path(fx, "VERSION"))
cat(nrow(p), "rows\n")
