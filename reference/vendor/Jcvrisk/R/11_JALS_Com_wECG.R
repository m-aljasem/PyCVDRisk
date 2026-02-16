jals_composite_wECG <- function(data) {
  # Check if all required columns are present in the data frame
  required_cols <- c("age", "male", "sbp", "dbp","t2dm", "hdl", "smoking", "ht_medication","af")
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
  data$af <- ifelse(data$af == 1, 20, 0)
  data$hdl <- as.character(cut(data$hdl, breaks = c(-Inf, 40, 60, Inf), labels = c(4,2,0), right = FALSE))


  # Index for NA in medication
  idx_na <- is.na(data$ht_medication)

  # bp_medication == 0
  idx0 <- data$ht_medication == 0 & !idx_na
  bp_score[idx0]  <- with(data[idx0,],
                          ifelse(sbp < 120 & dbp < 80, 0,
                                 ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 7,
                                        ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 7,
                                               ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 14,
                                                      ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 19,
                                                             ifelse(sbp >= 180 | dbp >= 110, 26, NA)))))))

  # bp_medication == 1
  idx1 <- data$ht_medication == 1 & !idx_na
  bp_score[idx1] <- with(data[idx1, ], ifelse(sbp < 120 & dbp < 80, 10,
                                              ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 13,
                                                     ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 14,
                                                            ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 17,
                                                                   ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 18,
                                                                          ifelse(sbp >= 180 | dbp >= 110, 18, NA)))))))
  data$bp <- bp_score


  data$t2dm <- ifelse(data$t2dm == 1, 6, 0)
  data$male <- ifelse(data$male == 1, 5, 0)
  data$smoking <- ifelse(data$smoking == 1, 8, 0)


  # Calculate total score
  data$totalscore <- as.numeric(data$af) + as.numeric(data$hdl) + as.numeric(data$bp) + as.numeric(data$t2dm) + as.numeric(data$male) + as.numeric(data$smoking)

  data$totalscore <- ifelse(data$totalscore < 0, 0,
                            ifelse(data$totalscore > 60, 60,data$totalscore))

  # Create a character vector of total scores ranging from 0 to 60.
  score_range <- as.character(0:60)


  # Define the 5y absolute risk values corresponding to each total score across age group.

  ## risk map for 40-49y person
  risk_values40 <- c(0.12,   0.13,   0.14,   0.14,   0.16,
                     0.17,   0.18,   0.19,   0.20,   0.22,
                     0.24,   0.25,   0.27,   0.29,   0.31,
                     0.33,   0.36,   0.38,   0.41,   0.44,
                     0.47,   0.50,   0.54,   0.58,   0.62,
                     0.66,   0.71,   0.76,   0.82,   0.87,
                     0.94,   1.00,   1.08,   1.15,   1.24,
                     1.32,   1.42,   1.52,   1.63,   1.74,
                     1.87,   2.00,   2.14,   2.29,   2.45,
                     2.63,   2.81,   3.01,   3.23,   3.45,
                     3.70,   3.96,   4.24,   4.53,   4.85,
                     5.19,   5.55,   5.94,   6.35,   6.79,   7.26  )

  risk_map40 <- setNames(risk_values40, score_range)



  ## risk map for 50-59y person
  risk_values50 <- c(0.17,   0.18,   0.20,   0.21,   0.23,
                     0.24,   0.26,   0.28,   0.30,   0.32,
                     0.34,   0.37,   0.39,   0.42,   0.45,
                     0.48,   0.52,   0.55,   0.59,   0.64,
                     0.68,   0.73,   0.78,   0.84,   0.90,
                     0.96,   1.03,   1.11,   1.19,   1.27,
                     1.36,   1.46,   1.56,   1.67,   1.79,
                     1.92,   2.06,   2.20,   2.36,   2.52,
                     2.70,   2.89,   3.10,   3.32,   3.55,
                     3.80,   4.07,   4.35,   4.66,   4.99,
                     5.33,   5.70,   6.10,   6.52,   6.98,
                     7.46,   7.97,   8.52,   9.10,   9.72,   10.38
  )

  risk_map50 <- setNames(risk_values50, score_range)



  ## risk map for 60-69y person
  risk_values60 <- c(0.29,   0.31,   0.33,   0.36,   0.38,
                     0.41,   0.44,   0.47,   0.51,   0.54,
                     0.58,   0.62,   0.67,   0.72,   0.77,
                     0.82,   0.88,   0.94,   1.01,   1.08,
                     1.16,   1.24,   1.33,   1.43,   1.53,
                     1.64,   1.75,   1.88,   2.01,   2.15,
                     2.31,   2.47,   2.64,   2.83,   3.03,
                     3.25,   3.48,   3.72,   3.98,   4.26,
                     4.56,   4.88,   5.22,   5.58,   5.97,
                     6.39,   6.83,   7.30,   7.80,   8.34,
                     8.91,   9.52,  10.17,  10.86,  11.59,
                     12.37,  13.19,  14.07,  15.00,  15.99, 17.03 )

  risk_map60 <- setNames(risk_values60, score_range)



  ## risk map for 70-79y person
  risk_values70 <- c(0.57,   0.61,   0.66,   0.71,   0.76,
                     0.81,   0.87,   0.93,   1.00,   1.07,
                     1.14,   1.23,   1.31,   1.41,   1.51,
                     1.61,   1.73,   1.85,   1.98,   2.13,
                     2.28,   2.44,   2.61,   2.79,   2.99,
                     3.20,   3.43,   3.67,   3.93,   4.21,
                     4.50,   4.82,   5.15,   5.51,   5.89,
                     6.30,   6.74,   7.21,   7.70,   8.23,
                     8.80,   9.40,  10.04,  10.72,  11.44,
                     12.21,  13.03,  13.90,  14.81,  15.79,
                     16.82,  17.91,  19.07,  20.29,  21.58,
                     22.93,  24.36,  25.86,  27.43,  29.09,  30.81  )
  risk_map70 <- setNames(risk_values70, score_range)



  ## risk map for 80-89y person
  risk_values80 <- c(0.94,   1.01,   1.08,   1.16,   1.24,
                     1.33,   1.43,   1.53,   1.64,   1.75,
                     1.88,   2.01,   2.15,   2.31,   2.47,
                     2.64,   2.83,   3.03,   3.25,   3.47,
                     3.72,   3.98,   4.26,   4.56,   4.88,
                     5.22,   5.58,   5.97,   6.39,   6.83,
                     7.30,   7.80,   8.34,   8.91,   9.52,
                     10.17,  10.86,  11.59,  12.37,  13.19,
                     14.07,  15.00,  15.98,  17.03,  18.13,
                     19.30,  20.53,  21.83,  23.20,  24.64,
                     26.16,  27.75,  29.41,  31.16,  32.98,
                     34.87,  36.85,  38.90,  41.02,  43.21,   45.47)
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
