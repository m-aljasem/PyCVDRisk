# Secondary reference: the CRAN package QRISK3 (Li et al.), a third-party R port.
# Run from repo root after run_reference.py: Rscript reference/qrisk3/run_reference_cran.R
suppressMessages(library(QRISK3))
fx <- "reference/qrisk3/fixtures"
p <- read.csv(file.path(fx, "patients.csv"))
eth <- c("white_or_not_stated", "indian", "pakistani", "bangladeshi", "other_asian",
         "black_caribbean", "black_african", "chinese", "other")
d <- data.frame(id = p$id, gender = as.integer(p$sex == "female"), age = floor(p$age),
  af = p$af, aty = p$atypical_antipsychotics, ster = p$corticosteroids,
  ed = p$erectile_dysfunction, mig = p$migraine, ra = p$rheumatoid_arthritis, ckd = p$ckd,
  smi = p$severe_mental_illness, sle = p$sle, bpt = p$bp_treated,
  t1 = as.integer(p$diabetes == "type1"), t2 = as.integer(p$diabetes == "type2"),
  weight = p$bmi, height = 100, eth = match(p$ethnicity, eth), fh = p$family_history,
  rati = p$chol_hdl_ratio, sbp = p$sbp, sbps5 = p$sbp_sd,
  smoke = match(p$smoking, c("non", "ex", "light", "moderate", "heavy")), town = p$townsend)
r <- suppressMessages(QRISK3_2017(d, "id", "gender", "age", "af", "aty", "ster", "ed", "mig", "ra",
  "ckd", "smi", "sle", "bpt", "t1", "t2", "weight", "height", "eth", "fh", "rati", "sbp", "sbps5",
  "smoke", "town"))
ref <- read.csv(file.path(fx, "reference.csv"), colClasses = "character")
stopifnot(all(r$id == ref$id))
ref$qrisk3_cran <- sprintf("%.17g", r$QRISK3_2017 / 100)
write.csv(ref, file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
cat(paste("secondary: QRISK3 (CRAN)", packageVersion("QRISK3"), "-", R.version.string, "\n"),
    file = file.path(fx, "VERSION"), append = TRUE)
