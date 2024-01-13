#!/usr/bin/env Rscript
# ==============================================================================
# SCORE2-DM R Validation Script
# ==============================================================================
# This script calculates SCORE2-Diabetes risk scores using RiskScorescvd package
# and exports results for comparison with PyCVDRisk Python implementation.
#
# Reference:
#   SCORE2-Diabetes Working Group. (2023).
#   European Heart Journal, 44(28), 2544-2556.
# ==============================================================================

library(dplyr)
library(readr)

# Source the SCORE2-Diabetes function
source("RiskScorescvd/R/15_SCORE2-Diabetes_func.R")
source("RiskScorescvd/R/rounding.R")

cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("SCORE2-DM R VALIDATION\n")
cat("RiskScorescvd Package\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# ------------------------------------------------------------------------------
# Load Input Data
# ------------------------------------------------------------------------------
input_data <- read_csv("score2dm_input_data.csv", show_col_types = FALSE)
cat("Loaded", nrow(input_data), "patients from input data (age 40-69 with T2DM)\n")

# Map region names
region_map <- c(
  "low" = "Low",
  "moderate" = "Moderate", 
  "high" = "High",
  "very_high" = "Very high"
)

input_data <- input_data %>%
  mutate(
    Risk.region = region_map[region],
    Gender = sex,
    Age = age,
    smoker = as.numeric(smoking),
    systolic.bp = systolic_bp,
    total.chol = total_cholesterol,
    total.hdl = hdl_cholesterol,
    diabetes = 1,  # All patients have T2DM
    diabetes.age = as.numeric(diabetes_age),
    HbA1c = hba1c,
    eGFR = egfr
  )

# ------------------------------------------------------------------------------
# Calculate SCORE2-DM Risk Scores
# ------------------------------------------------------------------------------
cat("\nCalculating SCORE2-Diabetes risk scores...\n")

start_time <- Sys.time()

results <- input_data %>%
  rowwise() %>%
  mutate(
    r_risk_score = tryCatch(
      SCORE2_Diabetes(
        Risk.region = Risk.region,
        Age = Age,
        Gender = Gender,
        smoker = smoker,
        systolic.bp = systolic.bp,
        total.chol = total.chol,
        total.hdl = total.hdl,
        diabetes = diabetes,
        diabetes.age = diabetes.age,
        HbA1c = HbA1c,
        eGFR = eGFR,
        classify = FALSE
      ),
      error = function(e) NA_real_
    ),
    r_risk_category = tryCatch(
      SCORE2_Diabetes(
        Risk.region = Risk.region,
        Age = Age,
        Gender = Gender,
        smoker = smoker,
        systolic.bp = systolic.bp,
        total.chol = total.chol,
        total.hdl = total.hdl,
        diabetes = diabetes,
        diabetes.age = diabetes.age,
        HbA1c = HbA1c,
        eGFR = eGFR,
        classify = TRUE
      ),
      error = function(e) NA_character_
    )
  ) %>%
  ungroup()

end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Calculation complete!\n")
cat("Time elapsed:", round(elapsed, 2), "seconds\n")
cat("Throughput:", round(nrow(results) / elapsed), "patients/sec\n")

# ------------------------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------------------------
cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("R CALCULATION SUMMARY\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

valid_results <- results %>% filter(!is.na(r_risk_score))

cat("\nRisk Score Distribution:\n")
cat("  Min:   ", min(valid_results$r_risk_score, na.rm = TRUE), "%\n")
cat("  Max:   ", max(valid_results$r_risk_score, na.rm = TRUE), "%\n")
cat("  Mean:  ", round(mean(valid_results$r_risk_score, na.rm = TRUE), 2), "%\n")
cat("  Median:", round(median(valid_results$r_risk_score, na.rm = TRUE), 2), "%\n")

cat("\nRisk Category Distribution:\n")
cat_counts <- table(valid_results$r_risk_category)
for (cat_name in names(cat_counts)) {
  pct <- round(cat_counts[cat_name] / sum(cat_counts) * 100, 1)
  cat("  ", cat_name, ":", cat_counts[cat_name], "(", pct, "%)\n")
}

cat("\nBy Sex:\n")
for (s in c("male", "female")) {
  subset <- valid_results %>% filter(Gender == s)
  if (nrow(subset) > 0) {
    cat("  ", tools::toTitleCase(s), ": mean=", round(mean(subset$r_risk_score), 2), 
        "%, median=", round(median(subset$r_risk_score), 2), "%\n")
  }
}

# ------------------------------------------------------------------------------
# Export Results
# ------------------------------------------------------------------------------
output_data <- results %>%
  select(
    patient_id,
    age, sex, systolic_bp, total_cholesterol, hdl_cholesterol,
    smoking, hba1c, egfr, diabetes_age, region,
    r_risk_score,
    r_risk_category
  )

write_csv(output_data, "score2dm_r_results.csv")
cat("\nExported R results to: score2dm_r_results.csv\n")

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("VALIDATION COMPLETE\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
