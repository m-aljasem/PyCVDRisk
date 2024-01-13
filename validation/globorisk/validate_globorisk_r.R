#!/usr/bin/env Rscript
# ==============================================================================
# Globorisk R Validation Script
# ==============================================================================
# This script calculates Globorisk CVD risk scores using the globorisk R package
# and exports results for comparison with PyCVDRisk Python implementation.
#
# Reference:
#   Ueda P, Woodward M, Lu Y, et al. (2017).
#   The Lancet Diabetes & Endocrinology, 5(3), 196-213.
# ==============================================================================

library(dplyr)
library(readr)

# Install/load the globorisk package from local source
# install.packages("globorisk-master", repos = NULL, type = "source")

# Load the package data (stored in sysdata.rda)
load("globorisk-master/R/sysdata.rda")

# Source the main function
source("globorisk-master/R/globorisk.R")

cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("GLOBORISK R VALIDATION\n")
cat("globorisk Package\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# ------------------------------------------------------------------------------
# Load Input Data
# ------------------------------------------------------------------------------
input_data <- read_csv("globorisk_input_data.csv", show_col_types = FALSE)
cat("Loaded", nrow(input_data), "patients from input data\n")

# Prepare data for Globorisk function
# Globorisk expects: sex (0=male, 1=female), age, sbp, tc, dm, smk, iso, year
input_data <- input_data %>%
  mutate(
    sex_num = ifelse(sex == "male", 0, 1),
    smk = as.numeric(smoking),
    dm = as.numeric(diabetes),
    tc = total_cholesterol,
    sbp = systolic_bp,
    iso = country,
    year = 2020,  # Use 2020 for baseline year
    bmi_val = bmi
  )

# Get unique countries
countries <- unique(input_data$iso)
cat("Countries in data:", paste(countries, collapse = ", "), "\n")

# ------------------------------------------------------------------------------
# Calculate Globorisk (Lab Version)
# ------------------------------------------------------------------------------
cat("\nCalculating Globorisk (lab version)...\n")

start_time <- Sys.time()

# Filter to age 40-80 (Globorisk limits)
lab_data <- input_data %>%
  filter(age >= 40 & age <= 80)

tryCatch({
  lab_results <- globorisk(
    sex = lab_data$sex_num,
    age = lab_data$age,
    sbp = lab_data$sbp,
    tc = lab_data$tc,
    dm = lab_data$dm,
    smk = lab_data$smk,
    iso = lab_data$iso,
    year = lab_data$year,
    version = "lab",
    type = "risk",
    updated_lac = TRUE
  )
  
  lab_data$r_risk_score_lab <- lab_results * 100  # Convert to percentage
  lab_data$r_version <- "lab"
  
}, error = function(e) {
  cat("  Error in lab version:", e$message, "\n")
  lab_data$r_risk_score_lab <- NA_real_
})

end_time <- Sys.time()
elapsed_lab <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Lab version complete! Time:", round(elapsed_lab, 2), "seconds\n")

# ------------------------------------------------------------------------------
# Calculate Globorisk (Office Version)
# ------------------------------------------------------------------------------
cat("\nCalculating Globorisk (office version)...\n")

start_time <- Sys.time()

office_data <- input_data %>%
  filter(age >= 40 & age <= 80)

tryCatch({
  office_results <- globorisk(
    sex = office_data$sex_num,
    age = office_data$age,
    sbp = office_data$sbp,
    smk = office_data$smk,
    bmi = office_data$bmi_val,
    iso = office_data$iso,
    year = office_data$year,
    version = "office",
    type = "risk",
    updated_lac = TRUE
  )
  
  office_data$r_risk_score_office <- office_results * 100
  
}, error = function(e) {
  cat("  Error in office version:", e$message, "\n")
  office_data$r_risk_score_office <- NA_real_
})

end_time <- Sys.time()
elapsed_office <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Office version complete! Time:", round(elapsed_office, 2), "seconds\n")

# Combine results
results <- lab_data %>%
  left_join(
    office_data %>% select(patient_id, r_risk_score_office),
    by = "patient_id"
  )

# ------------------------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------------------------
cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("R CALCULATION SUMMARY\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

# Lab version stats
valid_lab <- results %>% filter(!is.na(r_risk_score_lab))

cat("\nLab Version Risk Score Distribution:\n")
cat("  Min:   ", round(min(valid_lab$r_risk_score_lab, na.rm = TRUE), 2), "%\n")
cat("  Max:   ", round(max(valid_lab$r_risk_score_lab, na.rm = TRUE), 2), "%\n")
cat("  Mean:  ", round(mean(valid_lab$r_risk_score_lab, na.rm = TRUE), 2), "%\n")
cat("  Median:", round(median(valid_lab$r_risk_score_lab, na.rm = TRUE), 2), "%\n")

# Office version stats
valid_office <- results %>% filter(!is.na(r_risk_score_office))

cat("\nOffice Version Risk Score Distribution:\n")
cat("  Min:   ", round(min(valid_office$r_risk_score_office, na.rm = TRUE), 2), "%\n")
cat("  Max:   ", round(max(valid_office$r_risk_score_office, na.rm = TRUE), 2), "%\n")
cat("  Mean:  ", round(mean(valid_office$r_risk_score_office, na.rm = TRUE), 2), "%\n")
cat("  Median:", round(median(valid_office$r_risk_score_office, na.rm = TRUE), 2), "%\n")

cat("\nBy Country (Lab Version):\n")
country_stats <- valid_lab %>%
  group_by(iso) %>%
  summarise(
    mean_risk = round(mean(r_risk_score_lab, na.rm = TRUE), 2),
    n = n()
  ) %>%
  arrange(desc(mean_risk))

for (i in 1:nrow(country_stats)) {
  cat("  ", country_stats$iso[i], ": mean=", country_stats$mean_risk[i], 
      "%, n=", country_stats$n[i], "\n")
}

# ------------------------------------------------------------------------------
# Export Results
# ------------------------------------------------------------------------------
output_data <- results %>%
  select(
    patient_id,
    age, sex, systolic_bp, total_cholesterol, hdl_cholesterol,
    smoking, diabetes, bmi, country,
    r_risk_score_lab,
    r_risk_score_office
  )

write_csv(output_data, "globorisk_r_results.csv")
cat("\nExported R results to: globorisk_r_results.csv\n")

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("VALIDATION COMPLETE\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
