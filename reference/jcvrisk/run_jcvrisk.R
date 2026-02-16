# Run the Jcvrisk 0.1.3 functions (vendored, unmodified: reference/vendor/Jcvrisk/R) on
# reference/jcvrisk/fixtures/patients.csv; write fixtures/jcvrisk_output.csv. Run from the repo root.
# For the JALS functions the total score is also recorded, by evaluating each function with its final
# return(absolute_risk) replaced by return(data$totalscore).
for (f in list.files("reference/vendor/Jcvrisk/R", full.names = TRUE)) source(f)
FX <- "reference/jcvrisk/fixtures"
d <- read.csv(file.path(FX, "patients.csv"))
score_of <- function(fn) {
  src <- gsub("return(absolute_risk)", "return(data$totalscore)", deparse(fn), fixed = TRUE)
  eval(parse(text = src))
}
out <- data.frame(id = d$id)
q <- function(x) suppressWarnings(x)
out$hisayama <- q(hisayama(d))
out$suita_wecg <- q(suita_wECG(d))
out$suita_woecg <- q(suita_woECG(d))
out$epoch_chd <- q(epoch_chd(d))
out$epoch_stroke <- q(epoch_stroke(d))
out$epoch_cvd <- q(epoch_cvd(d))
jals <- list(stroke = c("jals_stroke_wECG", "jals_stroke_woECG"), ami = c("jals_ami_wECG", "jals_ami_woECG"),
             composite = c("jals_composite_wECG", "jals_composite_woECG"), cvd_death = c("jals_cvd_wECG", "jals_cvd_woECG"))
for (o in names(jals)) for (k in 1:2) {
  fn <- get(jals[[o]][k]); tag <- paste0("jals_", o, c("_af", "_noaf")[k])
  out[[tag]] <- q(fn(d))
  out[[paste0(tag, "_score")]] <- q(score_of(fn)(d))
}
for (n in names(out)[-1]) out[[n]] <- sprintf("%.12g", out[[n]])
write.csv(out, file.path(FX, "jcvrisk_output.csv"), row.names = FALSE, quote = FALSE)
