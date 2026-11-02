# Extract the Globorisk model data from the authors' R package (globorisk 1.0.2,
# MIT licence, vendored at reference/vendor/globorisk) into plain CSV.
#
# The package stores, for each (country, sex, baseline year, age), the baseline
# CVD hazard for each follow-up year t as cvd_t. These satisfy
#   cvd_t(age, year) == cvd_0(age + t, year + t),
# so they reduce to one rate per (outcome type, country, sex, calendar year,
# attained age). This script builds that table and FAILS if any two source
# values for the same key differ, so the reduction is exact.
# Run from repo root: Rscript reference/globorisk/extract_data.R
load("reference/vendor/globorisk/R/sysdata.rda")
out <- "reference/globorisk/data"
dir.create(out, showWarnings = FALSE)
fmt <- function(x) ifelse(is.na(x), "", sprintf("%.17g", x))

long <- do.call(rbind, lapply(0:13, function(t) {
  v <- cvdr[[paste0("cvd_", t)]]
  keep <- !is.na(v)
  data.frame(type = cvdr$type[keep], iso = cvdr$iso[keep], sex = cvdr$sex[keep],
             year = cvdr$year[keep] + t, age = cvdr$age[keep] + t, rate = v[keep])
}))
key <- paste(long$type, long$iso, long$sex, long$year, long$age)
spread <- tapply(long$rate, key, function(r) max(r) - min(r))
stopifnot(all(spread == 0))
rates <- long[!duplicated(key), ]
rates <- rates[order(rates$type, rates$iso, rates$sex, rates$year, rates$age), ]
rates$rate <- fmt(rates$rate)
write.csv(rates, file.path(out, "baseline_rates.csv"), row.names = FALSE, quote = FALSE)

rf2 <- rf; for (v in grep("^mean_", names(rf2), value = TRUE)) rf2[[v]] <- fmt(rf2[[v]])
write.csv(rf2, file.path(out, "mean_risk_factors.csv"), row.names = FALSE, quote = FALSE)
co <- as.data.frame(coefs); for (v in setdiff(names(co), c("type", "lac"))) co[[v]] <- fmt(co[[v]])
write.csv(co, file.path(out, "coefficients.csv"), row.names = FALSE, quote = FALSE)
cat("rates:", nrow(rates), " from source values:", nrow(long), "\n")
print(table(rates$type)); print(range(rates$year)); print(range(rates$age))
