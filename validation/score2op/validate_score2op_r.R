#!/usr/bin/env Rscript
# ==============================================================================
# SCORE2-OP R Validation Script  
# ==============================================================================
# This script calculates SCORE2-OP risk scores using the RiskScorescvd R package
# and exports results for comparison with PyCVDRisk Python implementation.
#
# Note: The SCORE2() function in RiskScorescvd automatically uses SCORE2-OP
# coefficients when Age >= 70.
#
# Reference:
#   SCORE2-OP Working Group. (2021). European Heart Journal, 42(25), 2455-2467.
# ==============================================================================

library(dplyr)
library(readr)

# Source the SCORE2 function (handles both SCORE2 and SCORE2-OP)
source("RiskScorescvd/R/11_SCORE2_func.R")
source("RiskScorescvd/R/rounding.R")

cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("SCORE2-OP R VALIDATION\n")
cat("RiskScorescvd Package\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# ------------------------------------------------------------------------------
# Load Input Data
# ------------------------------------------------------------------------------
input_data <- read_csv("score2op_input_data.csv", show_col_types = FALSE)
cat("Loaded", nrow(input_data), "patients from input data (age >= 70)\n")

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
    diabetes = as.numeric(diabetes)
  )

# ------------------------------------------------------------------------------
# Calculate SCORE2-OP Risk Scores
# ------------------------------------------------------------------------------
cat("\nCalculating SCORE2-OP risk scores...\n")

start_time <- Sys.time()

results <- input_data %>%
  rowwise() %>%
  mutate(
    r_risk_score = tryCatch(
      SCORE2(
        Risk.region = Risk.region,
        Age = Age,
        Gender = Gender,
        smoker = smoker,
        systolic.bp = systolic.bp,
        diabetes = diabetes,  # SCORE2-OP uses diabetes
        total.chol = total.chol,
        total.hdl = total.hdl,
        classify = FALSE
      ),
      error = function(e) NA_real_
    ),
    r_risk_category = tryCatch(
      SCORE2(
        Risk.region = Risk.region,
        Age = Age,
        Gender = Gender,
        smoker = smoker,
        systolic.bp = systolic.bp,
        diabetes = diabetes,
        total.chol = total.chol,
        total.hdl = total.hdl,
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

cat("\nBy Diabetes Status:\n")
for (dm in c(TRUE, FALSE)) {
  subset <- valid_results %>% filter(diabetes == dm)
  if (nrow(subset) > 0) {
    status <- ifelse(dm, "Diabetic", "Non-diabetic")
    cat("  ", status, ": mean=", round(mean(subset$r_risk_score), 2), 
        "%, n=", nrow(subset), "\n")
  }
}

# ------------------------------------------------------------------------------
# Export Results
# ------------------------------------------------------------------------------
output_data <- results %>%
  select(
    patient_id,
    age, sex, systolic_bp, total_cholesterol, hdl_cholesterol,
    smoking, diabetes, region,
    r_risk_score,
    r_risk_category
  )

write_csv(output_data, "score2op_r_results.csv")
cat("\nExported R results to: score2op_r_results.csv\n")

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("VALIDATION COMPLETE\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
