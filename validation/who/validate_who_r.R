#!/usr/bin/env Rscript
# ==============================================================================
# WHO CVD Risk R Validation Script
# ==============================================================================
# This script calculates WHO/ISH CVD risk scores using lookup tables
# and exports results for comparison with PyCVDRisk Python implementation.
#
# Reference:
#   WHO CVD Risk Chart Working Group. (2019).
#   The Lancet Global Health, 7(10), e1332-e1345.
# ==============================================================================

library(dplyr)
library(readr)

cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("WHO CVD RISK R VALIDATION\n")
cat("whoishRisk Package\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# ------------------------------------------------------------------------------
# Load Input Data and Reference Table
# ------------------------------------------------------------------------------
input_data <- read_csv("who_input_data.csv", show_col_types = FALSE)
cat("Loaded", nrow(input_data), "patients from input data\n")

# Load WHO reference table
ref <- read_csv("whoishRisk-master/inst/extdata/WHO_ISH_Scores.csv", show_col_types = FALSE)
cat("Loaded WHO/ISH reference table\n")

# Prepare data
df <- input_data %>%
  mutate(
    gdr = ifelse(sex == "male", 1, 0),
    smk = as.numeric(smoking),
    sbp = systolic_bp,
    dm = as.numeric(diabetes),
    chl = total_cholesterol,
    subregion = region
  )

# Get unique subregions
subregions <- unique(df$subregion)
cat("Subregions in data:", paste(subregions, collapse = ", "), "\n")

# ------------------------------------------------------------------------------
# Discretize variables according to WHO categories
# ------------------------------------------------------------------------------
cat("\nCalculating WHO/ISH risk scores...\n")

start_time <- Sys.time()

df <- df %>%
  mutate(
    # Age discretization
    age_cat = case_when(
      age < 50 ~ 40,
      age < 60 ~ 50,
      age < 70 ~ 60,
      TRUE ~ 70
    ),
    # Cholesterol discretization
    chl_cat = case_when(
      chl < 4.5 ~ 4,
      chl < 5.5 ~ 5,
      chl < 6.5 ~ 6,
      chl < 7.5 ~ 7,
      TRUE ~ 8
    ),
    # SBP discretization
    sbp_cat = case_when(
      sbp < 140 ~ 120,
      sbp < 160 ~ 140,
      sbp < 180 ~ 160,
      TRUE ~ 180
    )
  )

# Create lookup value (matches reference table format)
df$luv <- paste0(df$age_cat, df$gdr, df$dm, df$smk, df$sbp_cat, df$chl_cat)
ref$refv <- paste0(ref$age, ref$gdr, ref$dm, ref$smk, ref$sbp, ref$chl)

# Function to get risk for each subregion
get_who_risk <- function(df_row, ref_table, subregion) {
  luv <- df_row[["luv"]]
  match_idx <- match(luv, ref_table$refv)
  if (is.na(match_idx)) {
    return(NA_character_)
  }
  # Get the column for this subregion
  if (subregion %in% names(ref_table)) {
    return(as.character(ref_table[[subregion]][match_idx]))
  } else {
    return(NA_character_)
  }
}

# Apply lookup for each patient
results <- df %>%
  rowwise() %>%
  mutate(
    r_risk_category = get_who_risk(cur_data(), ref, subregion)
  ) %>%
  ungroup()

end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Calculation complete!\n")
cat("Time elapsed:", round(elapsed, 2), "seconds\n")
cat("Patients processed:", nrow(results), "\n")

# ------------------------------------------------------------------------------
# Convert WHO risk categories to numeric midpoints
# ------------------------------------------------------------------------------
results <- results %>%
  mutate(
    r_risk_score = case_when(
      r_risk_category == "<10%" ~ 5.0,
      r_risk_category == "10% to <20%" ~ 15.0,
      r_risk_category == "20% to <30%" ~ 25.0,
      r_risk_category == "30% to <40%" ~ 35.0,
      r_risk_category == ">=40%" ~ 45.0,
      TRUE ~ NA_real_
    )
  )

# ------------------------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------------------------
cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("R CALCULATION SUMMARY\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

valid_results <- results %>% filter(!is.na(r_risk_score))

cat("\nValid results:", nrow(valid_results), "of", nrow(results), "\n")

if (nrow(valid_results) > 0) {
  cat("\nRisk Score Distribution (midpoint values):\n")
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
  
  cat("\nBy Subregion (Top 5):\n")
  region_stats <- valid_results %>%
    group_by(subregion) %>%
    summarise(
      mean_risk = round(mean(r_risk_score, na.rm = TRUE), 2),
      n = n()
    ) %>%
    arrange(desc(n)) %>%
    head(5)
  
  for (i in 1:nrow(region_stats)) {
    cat("  ", region_stats$subregion[i], ": mean=", region_stats$mean_risk[i], 
        "%, n=", region_stats$n[i], "\n")
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

write_csv(output_data, "who_r_results.csv")
cat("\nExported R results to: who_r_results.csv\n")

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("VALIDATION COMPLETE\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
