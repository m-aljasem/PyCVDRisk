jals_cvd_wECG <- function(data) {
  # Check if all required columns are present in the data frame
  required_cols <- c("age", "male", "sbp", "dbp","t2dm", "hdl", "egfr", "smoking", "ht_medication","bmi","af")
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
  data$bmi <- as.numeric(data$bmi)
  data$hdl <- as.numeric(data$hdl)
  data$sbp <- as.numeric(data$sbp)
  data$dbp <- as.numeric(data$dbp)
  data$ht_medication <- as.numeric(data$ht_medication)
  data$age <- as.numeric(data$age)
  data$egfr <- as.numeric(data$egfr)
  data$t2dm <- as.numeric(data$t2dm)
  data$male <- as.numeric(data$male)
  data$smoking <- as.numeric(data$smoking)

  # dummy variable for bp_score (To specify variable class, numerical NA values were applied)
  bp_score <- rep(NA_real_, nrow(data))

  # Calculate scores
  data$af <- ifelse(data$af == 1, 18, 0)
  data$bmi <- as.character(cut(data$bmi, breaks = c(-Inf, 18.5, 25.0, Inf), labels = c(7,0,-2), right = FALSE))
  data$hdl <- as.character(cut(data$hdl, breaks = c(-Inf, 40, 60, Inf), labels = c(3,-1,0), right = FALSE))


  # Index for NA in medication
  idx_na <- is.na(data$ht_medication)

  # bp_medication == 0
  idx0 <- data$ht_medication == 0 & !idx_na
  bp_score[idx0]  <- with(data[idx0,],
                          ifelse(sbp < 120 & dbp < 80, 0,
                                 ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 0,
                                        ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 3,
                                               ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 4,
                                                      ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 11,
                                                             ifelse(sbp >= 180 | dbp >= 110, 13, NA)))))))

  # bp_medication == 1
  idx1 <- data$ht_medication == 1 & !idx_na
  bp_score[idx1] <- with(data[idx1, ], ifelse(sbp < 120 & dbp < 80, 6,
                                              ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 7,
                                                     ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 5,
                                                            ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 6,
                                                                   ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 9,
                                                                          ifelse(sbp >= 180 | dbp >= 110, 5, NA)))))))
  data$bp <- bp_score


  data$egfr <- as.character(cut(data$egfr, breaks = c(-Inf, 45, 60, 90, Inf), labels = c(11,2,0,1), right = FALSE))
  data$t2dm <- ifelse(data$t2dm == 1, 4, 0)
  data$male <- ifelse(data$male == 1, 6, 0)
  data$smoking <- ifelse(data$smoking == 1, 9, 0)


  # Calculate total score
  data$totalscore <- as.numeric(data$af) + as.numeric(data$bmi) + as.numeric(data$hdl) + as.numeric(data$bp) + as.numeric(data$egfr) + as.numeric(data$t2dm) + as.numeric(data$male) + as.numeric(data$smoking)
  data$totalscore <- ifelse(data$totalscore < 0, 0,
                            ifelse(data$totalscore > 60, 60,data$totalscore))

  # Create a character vector of total scores ranging from 0 to 60.
  score_range <- as.character(0:60)


  # Define the 5y absolute risk values corresponding to each total score across age group.

  ## risk map for 40-49y person
  risk_values40 <- c(0.07,   0.07,   0.08,   0.08,   0.09,
                     0.09,   0.10,   0.11,   0.12,   0.12,
                     0.13,   0.14,   0.15,   0.16,   0.18,
                     0.19,   0.20,   0.22,   0.23,   0.25,
                     0.27,   0.28,   0.31,   0.33,   0.35,
                     0.38,   0.40,   0.43,   0.46,   0.50,
                     0.53,   0.57,   0.61,   0.65,   0.70,
                     0.75,   0.80,   0.86,   0.92,   0.99,
                     1.06,   1.13,   1.22,   1.30,   1.40,
                     1.49,   1.60,   1.71,   1.84,   1.97,
                     2.11,   2.26,   2.42,   2.59,   2.77,
                     2.97,   3.18,   3.40,   3.64,   3.90,   4.17)

  risk_map40 <- setNames(risk_values40, score_range)



  ## risk map for 50-59y person
  risk_values50 <- c(0.10,   0.11,   0.12,   0.12,   0.13,
                     0.14,   0.15,   0.16,   0.17,   0.19,
                     0.20,   0.22,   0.23,   0.25,   0.26,
                     0.28,   0.30,   0.33,   0.35,   0.37,
                     0.40,   0.43,   0.46,   0.49,   0.53,
                     0.57,   0.61,   0.65,   0.70,   0.75,
                     0.80,   0.86,   0.92,   0.99,   1.06,
                     1.13,   1.21,   1.30,   1.39,   1.49,
                     1.60,   1.71,   1.83,   1.96,   2.10,
                     2.25,   2.41,   2.58,   2.76,   2.96,
                     3.17,   3.39,   3.63,   3.88,   4.16,
                     4.45,   4.76,   5.09,   5.45,   5.83,   6.23   )
  risk_map50 <- setNames(risk_values50, score_range)



  ## risk map for 60-69y person
  risk_values60 <- c(0.20,   0.22,   0.23,   0.25,   0.27,
                     0.29,   0.31,   0.33,   0.35,   0.38,
                     0.40,   0.43,   0.46,   0.50,   0.53,
                     0.57,   0.61,   0.65,   0.70,   0.75,
                     0.80,   0.86,   0.92,   0.99,   1.06,
                     1.14,   1.22,   1.30,   1.40,   1.50,
                     1.60,   1.72,   1.84,   1.97,   2.11,
                     2.26,   2.42,   2.59,   2.77,   2.97,
                     3.18,   3.40,   3.64,   3.90,   4.17,
                     4.47,   4.78,   5.11,   5.47,   5.85,
                     6.26,   6.69,   7.16,   7.65,   8.18,
                     8.74,   9.33,   9.97,   10.64,   11.36,   12.13   )

  risk_map60 <- setNames(risk_values60, score_range)



  ## risk map for 70-79y person
  risk_values70 <- c(0.74,   0.79,   0.85,   0.91,   0.97,
                     1.04,   1.11,   1.19,   1.28,   1.37,
                     1.47,   1.57,   1.68,   1.80,   1.93,
                     2.07,   2.22,   2.37,   2.54,   2.72,
                     2.91,   3.12,   3.34,   3.58,   3.83,
                     4.10,   4.38,   4.69,   5.02,   5.37,
                     5.74,   6.14,   6.57,   7.02,   7.51,
                     8.03,   8.58,   9.16,   9.79,  10.45,
                     11.16,  11.91,  12.71,  13.55,  14.45,
                     15.41,  16.42,  17.49,  18.62,  19.81,
                     21.07,  22.40,  23.80,  25.27,  26.82,
                     28.44,  30.14,  31.91,  33.77,  35.70,   37.70  )

  risk_map70 <- setNames(risk_values70, score_range)



  ## risk map for 80-89y person
  risk_values80 <- c(3.05,   3.27,   3.50,   3.75,   4.01,
                     4.29,   4.59,   4.91,   5.26,   5.62,
                     6.02,   6.43,   6.88,   7.35,   7.86,
                     8.40,   8.98,   9.59,  10.24,  10.93,
                     11.67,  12.45,  13.28,  14.17,  15.10,
                     16.10,  17.15,  18.26,  19.43,  20.67,
                     21.98,  23.36,  24.80,  26.33,  27.93,
                     29.60,  31.35,  33.18,  35.09,  37.07,
                     39.13,  41.26,  43.46,  45.72,  48.05,
                     50.44,  52.87,  55.35,  57.86,  60.40,
                     62.94,  65.49,  68.03,  70.54,  73.01,
                     75.44,  77.79,  80.06,  82.24,  84.31,   86.27  )

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
