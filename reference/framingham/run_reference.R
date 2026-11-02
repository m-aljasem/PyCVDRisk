# Reference answers for Framingham 2008 from CVrisk (ascvd_10y_frs, ascvd_10y_frs_simple), with
# output rounding (2 dp of %), the 1% floor and (BMI model) the 30% cap removed. Run from repo root:
# Rscript reference/framingham/run_reference.R
suppressMessages(library(CVrisk))
fx <- "reference/framingham/fixtures"
p <- read.csv(file.path(fx, "patients.csv"))
raw <- function(f) {
  b <- body(f)
  stopifnot(deparse(b[[length(b)]]) %in% c("ifelse(risk_score < 1, 1, risk_score)",
    "ifelse(risk_score < 1, 1, ifelse(risk_score > 30, 30, risk_score))"))  # 1% floor; frs_simple also caps at 30%
  b[[length(b)]] <- quote(risk_score)
  body(f) <- b
  environment(f) <- list2env(list(round = function(x, digits = 0) x), parent = asNamespace("CVrisk"))
  f
}
fmt <- function(x) ifelse(is.na(x), "", sprintf("%.17g", x))
lab <- raw(CVrisk::ascvd_10y_frs)(gender = p$sex, age = p$age, hdl = p$hdl_chol_mgdl, totchol = p$total_chol_mgdl,
  sbp = p$sbp, bp_med = p$bp_treated, smoker = p$smoker, diabetes = p$diabetes) / 100
bmi <- raw(CVrisk::ascvd_10y_frs_simple)(gender = p$sex, age = p$age, bmi = p$bmi, sbp = p$sbp,
  bp_med = p$bp_treated, smoker = p$smoker, diabetes = p$diabetes) / 100
write.csv(data.frame(id = p$id, lab = fmt(lab), bmi = fmt(bmi)), file.path(fx, "reference.csv"),
          row.names = FALSE, quote = FALSE)
writeLines(c(paste("reference: CVrisk", packageVersion("CVrisk"), "(rounding, 1% floor and 30% cap removed)"),
             "official: Framingham Heart Study spreadsheets gencardio_lipids.xls, gencardio_bmi.xls (cached examples)",
             paste("R:", R.version.string), paste("generated:", format(Sys.Date()))),
           file.path(fx, "VERSION"))
