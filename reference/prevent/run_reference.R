# Reference answers for PREVENT from the R package preventr (CRAN). preventr
# rounds risks to 3 decimals via round_half_up(); that is replaced by identity
# so answers compare at machine precision. Run from repo root:
#   Rscript reference/prevent/run_reference.R
suppressMessages(library(preventr))
assignInNamespace("round_half_up", function(x, digits = 0) x, "preventr")
fx <- "reference/prevent/fixtures"
p <- read.csv(file.path(fx, "patients.csv"))
fmt <- function(x) ifelse(is.na(x), "", sprintf("%.17g", x))
outcomes <- c("total_cvd", "ascvd", "heart_failure", "chd", "stroke")
rows <- list()
for (model in c("base", "uacr", "hba1c", "sdi", "full")) {
  for (i in seq_len(nrow(p))) {
    r <- p[i, ]
    est <- suppressMessages(estimate_risk(
      age = r$age, sex = r$sex, sbp = r$sbp, bp_tx = r$bp_treated == 1,
      total_c = r$total_chol, hdl_c = r$hdl_chol, statin = r$statin == 1,
      dm = r$diabetes == 1, smoking = r$smoker == 1, egfr = r$egfr, bmi = r$bmi,
      hba1c = if (model %in% c("hba1c", "full")) r$hba1c else NULL,
      uacr = if (model %in% c("uacr", "full")) r$uacr else NULL,
      zip = if (model %in% c("sdi", "full")) r$sdi else NULL,  # preventr >= 0.12.2: SDI decile via zip
      model = model, time = "both", chol_unit = "mmol/L", quiet = TRUE))
    for (h in c("10", "30")) {
      e <- est[[paste0("risk_est_", h, "yr")]]
      rows[[length(rows) + 1]] <- data.frame(
        id = r$id, model = model, years = as.integer(h),
        as.list(setNames(fmt(unlist(e[outcomes])), outcomes)))
    }
  }
  cat(model, "done\n")
}
out <- do.call(rbind, rows)
write.csv(out, file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
writeLines(c(
  "reference: preventr (CRAN), R implementation of the AHA PREVENT equations",
  paste("package_version:", as.character(packageVersion("preventr"))),
  paste("R:", R.version.string),
  paste("generated:", format(Sys.Date())),
  "rounding: disabled (preventr:::round_half_up replaced by identity)",
  "models: base, uacr, hba1c, sdi, full (SDI decile passed via zip); HbA1c erratum 2026-10-14 included"
), file.path(fx, "VERSION"))
