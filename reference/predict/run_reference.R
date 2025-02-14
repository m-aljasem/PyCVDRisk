# Run the authors' PREDICT functions (vendored, unmodified: reference/vendor/PredictRiskScores/R) on
# reference/predict/fixtures/<model>_patients.csv; write <model>_reference.csv. Run from repo root.
# dp = 12 instead of the default 4 so the comparison is not limited by R's rounding.
for (f in list.files("reference/vendor/PredictRiskScores/R", full.names = TRUE)) source(f)
FX <- "reference/predict/fixtures"
rd <- function(m) read.csv(file.path(FX, paste0(m, "_patients.csv")), stringsAsFactors = FALSE)
wr <- function(m, risk) write.csv(data.frame(id = seq_along(risk) - 1, risk = sprintf("%.12f", risk)),
                                  file.path(FX, paste0(m, "_reference.csv")), row.names = FALSE, quote = FALSE)
smk <- function(d) { d$exsmoker <- as.integer(d$smoking == "ex"); d$smoker <- as.integer(d$smoking == "current"); d }

d <- smk(rd("cvd"))
wr("cvd", NoPriorCVDRisk(dat = d, sex = sex, age = age, eth = ethnicity, nzdep = nzdep, exsmoker = exsmoker,
                         smoker = smoker, diabetes = diabetes, af = af, familyhx = family_history, sbp = sbp,
                         tchdl = tc_hdl, bpl = bp_meds, lld = lipid_meds, athrombi = antithrombotic, dp = 12))
d <- smk(rd("cvd_bmi"))
wr("cvd_bmi", NoPriorCVDRisk_BMI(dat = d, sex = sex, age = age, eth = ethnicity, nzdep = nzdep, exsmoker = exsmoker,
                                 smoker = smoker, diabetes = diabetes, af = af, familyhx = family_history, sbp = sbp,
                                 tchdl = tc_hdl, bmi = bmi, bpl = bp_meds, lld = lipid_meds, athrombi = antithrombotic,
                                 dp = 12))
d <- rd("cvd_policy")
wr("cvd_policy", suppressWarnings(NoPriorCVDRisk_Policy(dat = d, sex = sex, age = age, eth = ethnicity, nzdep = nzdep,
                                                        diabetes = diabetes, af = af, bpl = bp_meds, lld = lipid_meds,
                                                        athrombi = antithrombotic, dp = 12)))
# PriorT2DRisk reads the ACR from a variable named `acr` in the caller, not from `dat`, and assigns
# it unsubsetted to each sex: so it is run one row at a time with `acr` set to that row's value.
d <- rd("t2d")
wr("t2d", sapply(seq_len(nrow(d)), function(i) {
  acr <- d$acr[i]
  PriorT2DRisk(dat = d[i, ], sex = sex, age = age, eth = ethnicity, nzdep = nzdep, smoker = smoker, af = af,
               familyhx = family_history, lld = lipid_meds, athrombi = antithrombotic, bpl = bp_meds,
               oral = oral_hypoglycaemic, insulin = insulin, sbp = sbp, tchdl = tc_hdl, bmi = bmi, egfr = egfr,
               acr = acr, hba1c = hba1c, years = years_since_diagnosis, dp = 12)
}))
d <- smk(rd("bleed"))
wr("bleed", NoPriorCVDBleedRisk(dat = d, sex = sex, age = age, eth = ethnicity, exsmoker = exsmoker, smoker = smoker,
                                nzdep = nzdep, diabetes = diabetes, familyhx = family_history, lld = lipid_meds,
                                bpl = bp_meds, cancer = cancer, gibleed = gi_bleed, puddiag = peptic_ulcer,
                                alcohol = alcohol, liver = liver, puddrug = peptic_ulcer_meds, nsaid = nsaid,
                                steroids = steroids, ssri = ssri, sbp = sbp, tchdl = tc_hdl, dp = 12))
d <- rd("post_cvd")
wr("post_cvd", PostCVDRisk(dat = d, sex = sex, age = age, eth = ethnicity, nzdep = nzdep, smoker = smoker,
                           diabetes = diabetes, af = af, hf = heart_failure, othervd = other_vd, bpl = bp_meds,
                           lld = lipid_meds, athrombi = antithrombotic, sbp = sbp, tchdl = tc_hdl, bmi = bmi,
                           scr = creatinine, hba1c = hba1c, cvddays = days_since_event, dp = 12))
d <- rd("post_acs")
d$acs_type[d$acs_type == "unstable_angina"] <- "0"
wr("post_acs", PostACSRisk(dat = d, sex = sex, age = age, eth = ethnicity, nzdep = nzdep, smoker = smoker,
                           diabetes = diabetes, af = af, hf = heart_failure, bpl = bp_meds, lld = lipid_meds,
                           athrombi = antithrombotic, sbp = sbp, tchdl = tc_hdl, bmi = bmi, scr = creatinine,
                           hba1c = hba1c, acsdays = days_since_acs, acstype = acs_type, dp = 12))
cat("done\n")
