# cliot 1.0.0 (CRAN) on every input combination in fixtures/*.csv (written by report.py).
# The three functions are cut unmodified into source/cliot/af_functions.R. Run from the repo root.
src <- "reference/af/source/cliot"
source(file.path(src, "af_functions.R"))
fx <- "reference/af/fixtures"
run <- function(name, f) {
  p <- read.csv(file.path(fx, paste0(name, ".csv")))
  s <- vapply(seq_len(nrow(p)), function(i) f(p[i, ]), numeric(1))
  write.csv(data.frame(id = p$id, score = s), file.path(fx, paste0("cliot_", name, ".csv")), row.names = FALSE)
}
run("chads2", function(r) chads2_score(r$heart_failure, r$hypertension, r$age, r$diabetes, r$stroke_tia)$CHADS2_Score)
run("cha2ds2_vasc", function(r) cha2ds2_vasc_score(r$age, r$sex, r$heart_failure, r$hypertension, r$stroke_tia_te,
                                                    r$vascular_disease, r$diabetes)$CHA2DS2_VASc_Score)
# cliot takes "elderly" as a flag; we pass age > 65, so its age rule is not tested here
run("has_bled", function(r) has_bled_score(r$hypertension, r$abnormal_renal, r$abnormal_liver, r$stroke, r$bleeding,
                                            r$labile_inr, as.numeric(r$age > 65), r$drugs, r$alcohol)$HAS_BLED_Score)
writeLines(c(paste("cliot", read.dcf(file.path(src, "DESCRIPTION"))[, "Version"]), R.version.string),
           file.path(fx, "VERSION_cliot"))
