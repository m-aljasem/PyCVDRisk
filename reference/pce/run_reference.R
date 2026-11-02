# Reference answers for the 2013 PCE. Run from repo root: Rscript reference/pce/run_reference.R
# 1. CVrisk::ascvd_10y_accaha, with its output rounding (2 dp of %) and its floor at 1% removed:
#    the function body is evaluated with `round` as identity and the final floor skipped.
# 2. PooledCohort::predict_10yr_ascvd_risk (Goff_2013), unmodified.
suppressMessages({library(CVrisk); library(PooledCohort)})
fx <- "reference/pce/fixtures"
p <- read.csv(file.path(fx, "patients.csv"))
fmt <- function(x) sprintf("%.17g", x)

f <- CVrisk::ascvd_10y_accaha
b <- body(f)
stopifnot(identical(deparse(b[[length(b)]]), "ifelse(risk_score < 1, 1, risk_score)"))
b[[length(b)]] <- quote(risk_score)  # drop the 1% floor
body(f) <- b
environment(f) <- list2env(list(round = function(x, digits = 0) x), parent = asNamespace("CVrisk"))
race <- ifelse(p$race == "african_american", "aa", "white")
cvrisk <- f(race = race, gender = p$sex, age = p$age, totchol = p$total_chol_mgdl, hdl = p$hdl_chol_mgdl,
            sbp = p$sbp, bp_med = p$bp_treated, smoker = p$smoker, diabetes = p$diabetes) / 100

yn <- function(x) ifelse(x == 1, "yes", "no")
pc <- predict_10yr_ascvd_risk(age_years = p$age, race = ifelse(p$race == "african_american", "black", "white"),
  sex = p$sex, smoke_current = yn(p$smoker), chol_total_mgdl = p$total_chol_mgdl,
  chol_hdl_mgdl = p$hdl_chol_mgdl, bp_sys_mmhg = p$sbp, bp_meds = yn(p$bp_treated),
  diabetes = yn(p$diabetes), equation_version = "Goff_2013")

write.csv(data.frame(id = p$id, cvrisk = fmt(cvrisk), pooledcohort = fmt(pc)),
          file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
writeLines(c(paste("reference: CVrisk", packageVersion("CVrisk"), "(rounding and 1% floor removed)"),
             paste("secondary: PooledCohort", packageVersion("PooledCohort"), "(equation_version = Goff_2013)"),
             paste("R:", R.version.string), paste("generated:", format(Sys.Date()))),
           file.path(fx, "VERSION"))
