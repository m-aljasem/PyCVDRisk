jals_stroke_wECG <- function(data) {
  # Check if all required columns are present in the data frame
  required_cols <- c("age", "male", "sbp", "dbp","t2dm", "smoking", "ht_medication", "af")
  if (!all(required_cols %in% names(data))) {
    stop("Data frame must contain the following columns: ", paste(required_cols, collapse = ", "))
  }

  # Check for NA values in required columns
  if (any(sapply(data[required_cols], is.na))) {
    warning("Data frame contains NA values in required columns.")
  }

  # Check for ages over 90
  if (any(data$age > 90)) {
    warning("Some individuals are aged 90 or older. results may not accurate")
  }

  # Check for ages under 40
  if (any(data$age < 40)) {
    warning("Some individuals are younger than 40. results may not accurate")
  }

  # Convert columns to numeric
  data$af <- as.numeric(data$af)
  data$sbp <- as.numeric(data$sbp)
  data$dbp <- as.numeric(data$dbp)
  data$ht_medication <- as.numeric(data$ht_medication)
  data$age <- as.numeric(data$age)
  data$t2dm <- as.numeric(data$t2dm)
  data$male <- as.numeric(data$male)
  data$smoking <- as.numeric(data$smoking)

  # dummy variable for bp_score (To specify variable class, numerical NA values were applied)
  bp_score <- rep(NA_real_, nrow(data))

  # Calculate scores
  data$af <- ifelse(data$af == 1, 21, 0)

  # Index for NA in medication
  idx_na <- is.na(data$ht_medication)

  # bp_medication == 0
  idx0 <- data$ht_medication == 0 & !idx_na
  bp_score[idx0]  <- with(data[idx0,],
                          ifelse(sbp < 120 & dbp < 80, 0,
                                 ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 8,
                                        ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 7,
                                               ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 14,
                                                      ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 21,
                                                             ifelse(sbp >= 180 | dbp >= 110, 27, NA)))))))

  # bp_medication == 1
  idx1 <- data$ht_medication == 1 & !idx_na
  bp_score[idx1] <- with(data[idx1, ], ifelse(sbp < 120 & dbp < 80, 12,
                                              ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 13,
                                                     ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 14,
                                                            ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 18,
                                                                   ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 19,
                                                                          ifelse(sbp >= 180 | dbp >= 110, 19, NA)))))))
  data$bp <- bp_score


  data$t2dm <- ifelse(data$t2dm == 1, 6, 0)
  data$male <- ifelse(data$male == 1, 4, 0)
  data$smoking <- ifelse(data$smoking == 1, 7, 0)


  # Calculate total score (under 0 = 0, over60 = 60)
  data$totalscore <- as.numeric(data$af) + as.numeric(data$bp) + as.numeric(data$t2dm) + as.numeric(data$male) + as.numeric(data$smoking)

  data$totalscore <- ifelse(data$totalscore < 0, 0,
                            ifelse(data$totalscore > 60, 60,data$totalscore))

  # Create a character vector of total scores ranging from 0 to 60.
  score_range <- as.character(0:60)


  # Define the 5y absolute risk values corresponding to each total score across age group.

  ## risk map for 40-49y person
  risk_values40 <- c(0.11,   0.12,   0.13,   0.14,   0.15,
                     0.16,   0.17,   0.18,   0.19,   0.21,
                     0.22,   0.24,   0.26,   0.28,   0.30,
                     0.32,   0.34,   0.36,   0.39,   0.42,
                     0.45,   0.48,   0.51,   0.55,   0.59,
                     0.63,   0.68,   0.73,   0.78,   0.83,
                     0.89,   0.96,   1.02,   1.10,   1.18,
                     1.26,   1.35,   1.45,   1.55,   1.66,
                     1.78,   1.90,   2.04,   2.18,   2.34,
                     2.50,   2.68,   2.87,   3.07,   3.29,
                     3.52,   3.77,   4.04,   4.32,   4.62,
                     4.94,   5.29,   5.66,   6.05,   6.47, 6.92)

  risk_map40 <- setNames(risk_values40, score_range)



  ## risk map for 50-59y person
  risk_values50 <- c(0.15,   0.16,   0.17,   0.18,   0.19,
                     0.21,   0.22,   0.24,   0.26,   0.27,
                     0.29,   0.31,   0.34,   0.36,   0.39,
                     0.42,   0.44,   0.48,   0.51,   0.55,
                     0.59,   0.63,   0.67,   0.72,   0.77,
                     0.83,   0.89,   0.95,   1.02,   1.09,
                     1.17,   1.25,   1.34,   1.44,   1.54,
                     1.65,   1.77,   1.89,   2.03,   2.17,
                     2.33,   2.49,   2.67,   2.86,   3.06,
                     3.27,   3.50,   3.75,   4.01,   4.30,
                     4.60,   4.92,   5.26,   5.63,   6.02,
                     6.44,   6.89,   7.36,   7.87,   8.41, 8.98)

  risk_map50 <- setNames(risk_values50, score_range)



  ## risk map for 60-69y person
  risk_values60 <- c(0.27,   0.29,   0.31,   0.33,   0.36,
                     0.38,   0.41,   0.44,   0.47,   0.50,
                     0.54,   0.58,   0.62,   0.66,   0.71,
                     0.76,   0.81,   0.87,   0.93,   1.00,
                     1.07,   1.15,   1.23,   1.32,   1.41,
                     1.51,   1.62,   1.74,   1.86,   1.99,
                     2.13,   2.29,   2.45,   2.62,   2.81,
                     3.01,   3.22,   3.44,   3.69,   3.95,
                     4.22,   4.52,   4.84,   5.17,   5.53,
                     5.92,   6.33,   6.77,   7.24,   7.74,
                     8.27,   8.83,   9.44,  10.08,  10.76,
                     11.49, 12.26,  13.08,  13.95,  14.87, 15.85)

  risk_map60 <- setNames(risk_values60, score_range)



  ## risk map for 70-79y person
  risk_values70 <- c(0.55,   0.59,   0.63,   0.68,   0.73,
                     0.78,   0.83,   0.89,   0.96,   1.03,
                     1.10,   1.18,   1.26,   1.35,   1.45,
                     1.55,   1.66,   1.78,   1.90,   2.04,
                     2.18,   2.34,   2.51,   2.68,   2.87,
                     3.08,   3.29,   3.52,   3.77,   4.04,
                     4.32,   4.62,   4.95,   5.29,   5.66,
                     6.06,   6.48,   6.92,   7.40,   7.91,
                     8.46,   9.03,   9.65,  10.31,  11.00,
                     11.75, 12.53,  13.37,  14.26,  15.20,
                     16.20, 17.25,  18.37,  19.55,  20.80,
                     22.11, 23.50,  24.95,  26.48,  28.09, 29.77)

  risk_map70 <- setNames(risk_values70, score_range)



  ## risk map for 80-89y person
  risk_values80 <- c(0.93,   0.99,   1.07,   1.14,   1.22,
                     1.31,   1.40,   1.50,   1.61,   1.73,
                     1.85,   1.98,   2.12,   2.27,   2.43,
                     2.60,   2.79,   2.99,   3.20,   3.42,
                     3.66,   3.92,   4.20,   4.49,   4.80,
                     5.14,   5.50,   5.88,   6.29,   6.73,
                     7.19,   7.69,   8.21,   8.78,   9.38,
                     10.02, 10.69,  11.42,  12.18,  13.00,
                     13.86, 14.78,  15.75,  16.78,  17.88,
                     19.03, 20.25,  21.53,  22.88,  24.31,
                     25.81, 27.38,  29.03,  30.75,  32.55,
                     34.43, 36.39,  38.42,  40.53,  42.71, 44.95)

  risk_map80 <- setNames(risk_values80, score_range)



  # Create a named vector that maps total scores to corresponding risk values across age.
  data$age <- as.character(cut(data$age, breaks = c(-Inf, 40, 50, 60, 70, 80, Inf), labels = c(40, 40, 50, 60, 70, 80), right = FALSE))


  # Calculate 5-year Absolute Risk
  absolute_risk <- ifelse(data$age == 40, risk_map40[as.character(data$totalscore)],
                          ifelse(data$age == 50, risk_map50[as.character(data$totalscore)],
                                 ifelse(data$age == 60, risk_map60[as.character(data$totalscore)],
                                        ifelse(data$age == 70, risk_map70[as.character(data$totalscore)],
                                               ifelse(data$age == 80, risk_map80[as.character(data$totalscore)],NA)))))
  absolute_risk <- as.numeric(absolute_risk)

  return(absolute_risk)
}
