# EDACS reference answers from RiskScorescvd (vendored at reference/vendor/RiskScorescvd), one row at a time.
# ECG and troponin inputs only affect its classification, not the score; they are set to 0.
# Run from the repo root: Rscript reference/edacs/run_reference.R
suppressMessages(library(dplyr))
pkg <- "reference/vendor/RiskScorescvd"
source(file.path(pkg, "R", "03_EDACS_func.R"))
fx <- "reference/edacs/fixtures"
p <- read.csv(file.path(fx, "patients.csv"))
score <- mapply(function(a, g, d, s, ht, hl, fh, sw, pr, pl, pa)
  suppressWarnings(EDACS(a, g, d, s, ht, hl, fh, sw, pr, pl, pa, 0, 0, 0, 0, classify = FALSE)),
  p$age, p$sex, p$diabetes, p$smoker, p$hypertension, p$hyperlipidaemia, p$family_history,
  p$diaphoresis, p$radiates, p$pleuritic, p$palpation)
write.csv(data.frame(id = p$id, score = score), file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
writeLines(c(
  "reference: RiskScorescvd (CRAN), EDACS() - vendored source, see reference/vendor/RiskScorescvd/DESCRIPTION",
  paste("package_version:", read.dcf(file.path(pkg, "DESCRIPTION"))[, "Version"]),
  paste("R:", R.version.string),
  paste("dplyr:", as.character(packageVersion("dplyr"))),
  paste("generated:", format(Sys.Date())),
  "rounding: none (integer points)"
), file.path(fx, "VERSION"))
cat(nrow(p), "rows\n")
