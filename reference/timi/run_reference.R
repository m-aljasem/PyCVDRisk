# TIMI reference answers from RiskScorescvd (vendored at reference/vendor/RiskScorescvd), one row at a time.
# Run from the repo root: Rscript reference/timi/run_reference.R
suppressMessages(library(dplyr))
pkg <- "reference/vendor/RiskScorescvd"
source(file.path(pkg, "R", "07_TIMIUAN_func.R"))
fx <- "reference/timi/fixtures"
p <- read.csv(file.path(fx, "patients.csv"))
score <- mapply(function(a, ht, hl, fh, d, s, pci, cabg, asp, ep, st, tn, g)
  TIMI(a, ht, hl, fh, d, s, pci, cabg, asp, ep, st, tn, g, classify = FALSE),
  p$age, p$hypertension, p$hyperlipidaemia, p$family_history, p$diabetes, p$smoker, p$previous_pci,
  p$previous_cabg, p$aspirin, p$episodes, p$st_deviation, p$hstni, p$sex)
write.csv(data.frame(id = p$id, score = score), file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
writeLines(c(
  "reference: RiskScorescvd (CRAN), TIMI() - vendored source, see reference/vendor/RiskScorescvd/DESCRIPTION",
  paste("package_version:", read.dcf(file.path(pkg, "DESCRIPTION"))[, "Version"]),
  paste("R:", R.version.string),
  paste("dplyr:", as.character(packageVersion("dplyr"))),
  paste("generated:", format(Sys.Date())),
  "rounding: none (integer points)"
), file.path(fx, "VERSION"))
cat(nrow(p), "rows\n")
