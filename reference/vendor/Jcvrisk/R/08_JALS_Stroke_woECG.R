jals_stroke_woECG <- function(data) {
  # Check if all required columns are present in the data frame
  required_cols <- c("age", "male", "sbp", "dbp","t2dm", "smoking", "ht_medication")
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
  # Index for NA in medication
  idx_na <- is.na(data$ht_medication)

  # bp_medication == 0
  idx0 <- data$ht_medication == 0 & !idx_na
  bp_score[idx0]  <- with(data[idx0,],
                          ifelse(sbp <  120 & dbp < 80, 0,
                                 ifelse(sbp >= 120 & sbp <= 129 & dbp <   80, 8,
                                        ifelse(sbp >= 130 & sbp <= 139 | dbp >=  80 & dbp <=  89, 7,
                                               ifelse(sbp >= 140 & sbp <= 159 | dbp >=  90 & dbp <=  99, 14,
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
  data$male <- ifelse(data$male == 1, 5, 0)
  data$smoking <- ifelse(data$smoking == 1, 7, 0)


  # Calculate total score (under 0 = 0, over60 = 60)
  data$totalscore <- as.numeric(data$bp) + as.numeric(data$t2dm) + as.numeric(data$male) + as.numeric(data$smoking)

  data$totalscore <- ifelse(data$totalscore < 0, 0,
                            ifelse(data$totalscore > 60, 60,data$totalscore))

  # Create a character vector of total scores ranging from 0 to 60.
  score_range <- as.character(0:60)


  # Define the 5y absolute risk values corresponding to each total score across age group.

  ## risk map for 40-49y person
  risk_values40 <- c(0.11,   0.12,   0.13,   0.13,   0.14,
                     0.15,   0.17,   0.18,   0.19,   0.20,
                     0.22,   0.23,   0.25,   0.27,   0.29,
                     0.31,   0.33,   0.36,   0.38,   0.41,
                     0.44,   0.47,   0.50,   0.54,   0.58,
                     0.62,   0.66,   0.71,   0.76,   0.81,
                     0.87,   0.94,   1.00,   1.07,   1.15,
                     1.23,   1.32,   1.41,   1.52,   1.62,
                     1.74,   1.86,   1.99,   2.14,   2.29,
                     2.45,   2.62,   2.81,   3.01,   3.22,
                     3.45,   3.69,   3.95,   4.23,   4.52,
                     4.84,   5.18,   5.54,   5.92,   6.34, 6.77)

  risk_map40 <- setNames(risk_values40, score_range)




  ## risk map for 50-59y person
  risk_values50 <- c(0.15,   0.16,     0.17,   0.18,     0.19,
                     0.21,   0.22,     0.24,   0.25,     0.27,
                     0.29,   0.31,     0.33,   0.36,     0.38,
                     0.41,   0.44,     0.47,   0.50,     0.54,
                     0.58,   0.62,     0.66,   0.71,     0.76,
                     0.82,   0.88,     0.94,   1.01,     1.08,
                     1.15,   1.24,     1.33,   1.42,     1.52,
                     1.63,   1.74,     1.87,   2.00,     2.14,
                     2.30,   2.46,     2.63,   2.82,     3.02,
                     3.23,   3.46,     3.70,   3.96,     4.24,
                     4.54,   4.86,     5.20,   5.56,     5.94,
                     6.36,   6.80,     7.27,   7.77,     8.30, 8.87 )

  risk_map50 <- setNames(risk_values50, score_range)



  ## risk map for 60-69y person
  risk_values60 <- c(0.27,   0.29,   0.31,   0.33,   0.35,
                     0.38,   0.41,   0.44,   0.47,   0.50,
                     0.54,   0.57,   0.62,   0.66,   0.71,
                     0.76,   0.81,   0.87,   0.93,   1.00,
                     1.07,   1.15,   1.23,   1.32,   1.41,
                     1.51,   1.62,   1.73,   1.85,   1.99,
                     2.13,   2.28,   2.44,   2.61,   2.80,
                     3.00,   3.21,   3.43,   3.68,   3.93,
                     4.21,   4.51,   4.82,   5.16,   5.52,
                     5.90,   6.31,   6.75,   7.22,   7.71,
                     8.24,   8.81,   9.41,   10.05,   10.73,
                     11.46, 12.23,  13.04,   13.91,   14.83, 15.81)

  risk_map60 <- setNames(risk_values60, score_range)



  ## risk map for 70-79y person
  risk_values70 <- c(0.56,   0.60,   0.64,   0.69,   0.73,
                     0.79,   0.84,   0.90,   0.97,   1.04,
                     1.11,   1.19,   1.28,   1.37,   1.46,
                     1.57,   1.68,   1.80,   1.93,   2.06,
                     2.21,   2.37,   2.54,   2.71,   2.91,
                     3.11,   3.33,   3.57,   3.82,   4.09,
                     4.37,   4.68,   5.01,   5.36,   5.73,
                     6.13,   6.55,   7.01,   7.49,   8.00,
                     8.55,   9.14,   9.76,   10.42,  11.13,
                     11.88, 12.67,  13.52,  14.42,   15.37,
                     16.38, 17.44,  18.57,  19.76,   21.02,
                     22.35, 23.74,  25.21,  26.75,   28.37, 30.07  )
  risk_map70 <- setNames(risk_values70, score_range)


  ## risk map for 80-89y person
  risk_values80 <- c(0.95,   1.01,   1.09,   1.16,   1.25,
                     1.34,   1.43,   1.53,   1.64,   1.76,
                     1.88,   2.02,   2.16,   2.31,   2.48,
                     2.65,   2.84,   3.04,   3.26,   3.49,
                     3.73,   3.99,   4.27,   4.57,   4.89,
                     5.24,   5.60,   5.99,   6.41,   6.85,
                     7.32,   7.83,   8.37,   8.94,   9.55,
                     10.20,   10.89,   11.62,   12.40,   13.23,
                     14.11,   15.04,   16.03,   17.08,   18.19,
                     19.36,   20.59,   21.90,   23.27,   24.71,
                     26.23,   27.83,   29.49,   31.24,   33.07,
                     34.97,   36.94,   39.00,   41.12,   43.32,   45.58)

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
