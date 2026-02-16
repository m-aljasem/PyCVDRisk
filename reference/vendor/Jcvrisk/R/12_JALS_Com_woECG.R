jals_composite_woECG <- function(data) {
  # Check if all required columns are present in the data frame
  required_cols <- c("age", "male", "sbp", "dbp","t2dm", "hdl", "smoking", "ht_medication")
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
  data$hdl <- as.numeric(data$hdl)
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
  data$hdl <- as.character(cut(data$hdl, breaks = c(-Inf, 40, 60, Inf), labels = c(4,2,0), right = FALSE))


  # Index for NA in medication
  idx_na <- is.na(data$ht_medication)

  # bp_medication == 0
  idx0 <- data$ht_medication == 0 & !idx_na
  bp_score[idx0]  <- with(data[idx0,],
                          ifelse(sbp < 120 & dbp < 80, 0,
                                 ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 7,
                                        ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 7,
                                               ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 13,
                                                      ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 19,
                                                             ifelse(sbp >= 180 | dbp >= 110, 26, NA)))))))

  # bp_medication == 1
  idx1 <- data$ht_medication == 1 & !idx_na
  bp_score[idx1] <- with(data[idx1, ],
                         ifelse(sbp < 120 & dbp < 80, 11,
                                ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 14,
                                       ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 14,
                                              ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 17,
                                                     ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 17,
                                                            ifelse(sbp >= 180 | dbp >= 110, 18, NA)))))))
  data$bp <- bp_score

  data$t2dm <- ifelse(data$t2dm == 1, 6, 0)
  data$male <- ifelse(data$male == 1, 6, 0)
  data$smoking <- ifelse(data$smoking == 1, 7, 0)

  # Calculate total score (under 0 = 0, over60 = 60)
  data$totalscore <- as.numeric(data$hdl) + as.numeric(data$bp) + as.numeric(data$t2dm) + as.numeric(data$male) + as.numeric(data$smoking)

  data$totalscore <- ifelse(data$totalscore < 0, 0,
                            ifelse(data$totalscore > 60, 60,data$totalscore))

  # Create a character vector of total scores ranging from 0 to 60.
  score_range <- as.character(0:60)


  # Define the 5y absolute risk values corresponding to each total score across age group.

  ## risk map for 40-49y person
  risk_values40 <- c(0.12,   0.12,   0.13,   0.14,   0.15,
                     0.16,   0.17,   0.19,   0.20,   0.22,
                     0.23,   0.25,   0.27,   0.28,   0.30,
                     0.33,   0.35,   0.37,   0.40,   0.43,
                     0.46,   0.49,   0.53,   0.57,   0.61,
                     0.65,   0.70,   0.75,   0.80,   0.86,
                     0.92,   0.99,   1.06,   1.13,   1.21,
                     1.30,   1.39,   1.49,   1.60,   1.71,
                     1.83,   1.96,   2.10,   2.25,   2.41,
                     2.58,   2.76,   2.96,   3.17,   3.39,
                     3.63,   3.89,   4.16,   4.45,   4.76,
                     5.09,   5.45,   5.83,   6.23,   6.67, 7.13)

  risk_map40 <- setNames(risk_values40, score_range)



  ## risk map for 50-59y person
  risk_values50 <- c(0.17,   0.18,   0.19,   0.21,   0.22,
                     0.24,   0.26,   0.27,   0.29,   0.32,
                     0.34,   0.36,   0.39,   0.42,   0.45,
                     0.48,   0.51,   0.55,   0.59,   0.63,
                     0.68,   0.72,   0.78,   0.83,   0.89,
                     0.95,   1.02,   1.09,   1.17,   1.26,
                     1.35,   1.44,   1.54,   1.65,   1.77,
                     1.90,   2.03,   2.18,   2.33,   2.50,
                     2.67,   2.86,   3.07,   3.28,   3.51,
                     3.76,   4.03,   4.31,   4.61,   4.93,
                     5.28,   5.65,   6.04,   6.46,   6.90,
                     7.38,   7.89,   8.43,   9.01,   9.62, 10.28  )

  risk_map50 <- setNames(risk_values50, score_range)



  ## risk map for 60-69y person
  risk_values60 <- c(0.29,   0.31,   0.33,   0.36,   0.38,
                     0.41,   0.44,   0.47,   0.51,   0.54,
                     0.58,   0.62,   0.67,   0.72,   0.77,
                     0.82,   0.88,   0.94,   1.01,   1.08,
                     1.16,   1.24,   1.33,   1.43,   1.53,
                     1.64,   1.75,   1.88,   2.01,   2.15,
                     2.31,   2.47,   2.64,   2.83,   3.03,
                     3.25,   3.47,   3.72,   3.98,   4.26,
                     4.56,   4.88,   5.22,   5.58,   5.97,
                     6.39,   6.83,   7.30,   7.80,   8.34,
                     8.91,   9.52,  10.17,  10.85,  11.59,
                     12.36,  13.19,  14.07,  15.00,  15.98, 17.03 )

  risk_map60 <- setNames(risk_values60, score_range)



  ## risk map for 70-79y person
  risk_values70 <- c(0.58,   0.62,   0.67,   0.72,   0.77,
                     0.82,   0.88,   0.94,   1.01,   1.08,
                     1.16,   1.24,   1.33,   1.43,   1.53,
                     1.64,   1.75,   1.88,   2.01,   2.15,
                     2.31,   2.47,   2.65,   2.83,   3.03,
                     3.25,   3.48,   3.72,   3.98,   4.26,
                     4.56,   4.88,   5.22,   5.59,   5.97,
                     6.39,   6.83,   7.30,   7.81,   8.34,
                     8.92,   9.52,  10.17,  10.86,  11.59,
                     12.37,  13.20,  14.08,  15.01,  15.99,
                     17.04,  18.14,  19.31,  20.54,  21.84,
                     23.21,  24.65,   26.17, 27.76,  29.43, 31.17)
  risk_map70 <- setNames(risk_values70, score_range)



  ## risk map for 80-89y person
  risk_values80 <- c(0.96,   1.03,   1.10,   1.18,   1.27,
                     1.36,   1.45,   1.56,   1.67,   1.79,
                     1.91,   2.05,   2.20,   2.35,   2.52,
                     2.70,   2.89,   3.09,   3.31,   3.54,
                     3.79,   4.06,   4.34,   4.65,   4.97,
                     5.32,   5.69,   6.09,   6.51,   6.96,
                     7.44,   7.96,   8.50,   9.08,   9.70,
                     10.36,  11.06,  11.81,  12.60,  13.44,
                     14.33,  15.28,  16.28,  17.34,  18.46,
                     19.65,  20.90,  22.22,  23.61,  25.07,
                     26.61,  28.22,  29.91,  31.67,  33.52,
                     35.44,  37.43,  39.50,  41.65,  43.86, 46.14)

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
