# Reference answers from the authors' R package globorisk 1.0.2 (vendored).
# Run from repo root: Rscript reference/globorisk/run_reference.R
pkg <- "reference/vendor/globorisk"
source(file.path(pkg, "R", "globorisk.R"))
load(file.path(pkg, "R", "sysdata.rda"))
fx <- "reference/globorisk/fixtures"
p <- read.csv(file.path(fx, "patients.csv"), na.strings = "")  # "" only: keep Namibia ("NA") as a country
p$sex01 <- ifelse(p$sex == "female", 1L, 0L)  # package codes 0 = man, 1 = woman
fmt <- function(x) ifelse(is.na(x), "", sprintf("%.17g", x))
out <- data.frame(id = p$id)
for (v in c("lab", "office", "fatal")) {
  r <- rep(NA_real_, nrow(p))
  for (h in sort(unique(p$horizon))) {
    i <- which(p$horizon == h)
    r[i] <- suppressWarnings(globorisk(
      sex = p$sex01[i], age = p$age[i], sbp = p$sbp[i], tc = p$total_chol[i],
      dm = p$diabetes[i], smk = p$smoker[i], bmi = p$bmi[i], iso = p$country[i],
      year = p$year[i], time = h, version = v))
  }
  out[[v]] <- fmt(r)
  cat(v, "NA:", sum(is.na(r)), "\n")
}
write.csv(out, file.path(fx, "reference.csv"), row.names = FALSE, quote = FALSE)
writeLines(c(
  "reference: globorisk (authors' R package), vendored at reference/vendor/globorisk",
  paste("package_version:", read.dcf(file.path(pkg, "DESCRIPTION"))[, "Version"]),
  paste("R:", R.version.string),
  paste("generated:", format(Sys.Date())),
  "options: updated_lac = FALSE (default)"
), file.path(fx, "VERSION"))
