# Export the WHO 2019 recalibration parameters and country -> region map from
# WHORiskCalculator (CRAN, MIT; data built by its authors from the WHO CVD Risk
# Chart Working Group's Stata data file for `whocvdrisk`).
# Run from repo root: Rscript reference/who2019/extract_data.R
suppressMessages(library(WHORiskCalculator))
out <- "src/cvd_risk/who2019"
p <- WHORiskCalculator:::.calibration_parameters
p <- p[order(p$level, p$code, p$sex, p$year), ]
fmt <- function(d) { for (c in names(d)) if (is.double(d[[c]])) d[[c]] <- sprintf("%.17g", d[[c]]); d }
write.csv(fmt(p), file.path(out, "recalibration.csv"), row.names = FALSE, quote = FALSE)
m <- WHORiskCalculator:::.get_country_mapping()
write.csv(data.frame(iso3 = names(m), region = unname(m)), file.path(out, "countries.csv"),
          row.names = FALSE, quote = FALSE)
cat("WHORiskCalculator", as.character(packageVersion("WHORiskCalculator")), ":", nrow(p),
    "recalibration rows,", length(m), "countries\n")
