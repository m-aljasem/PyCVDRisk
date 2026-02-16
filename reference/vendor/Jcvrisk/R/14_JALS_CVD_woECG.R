jals_cvd_woECG <- function(data) {
  # Check if all required columns are present in the data frame
  required_cols <- c("age", "male", "sbp", "dbp","t2dm", "hdl", "egfr", "smoking", "ht_medication","bmi")
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
  data$bmi <- as.character(cut(data$bmi, breaks = c(-Inf, 18.5, 25.0, Inf), labels = c(6,0,-1), right = FALSE))
  data$hdl <- as.character(cut(data$hdl, breaks = c(-Inf, 40, 60, Inf), labels = c(3,-1,0), right = FALSE))


  # Index for NA in medication
  idx_na <- is.na(data$ht_medication)

  # bp_medication == 0
  idx0 <- data$ht_medication == 0 & !idx_na
  bp_score[idx0]  <- with(data[idx0,],
                          ifelse(sbp <  120 & dbp < 80, 0,
                                 ifelse(sbp >= 120 & sbp <= 129 & dbp <   80, 0,
                                        ifelse(sbp >= 130 & sbp <= 139 | dbp >=  80 & dbp <=  89, 3,
                                               ifelse(sbp >= 140 & sbp <= 159 | dbp >=  90 & dbp <=  99, 4,
                                                      ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 10,
                                                             ifelse(sbp >= 180 | dbp >= 110, 13, NA)))))))

  # bp_medication == 1
  idx1 <- data$ht_medication == 1 & !idx_na
  bp_score[idx1] <- with(data[idx1, ], ifelse(sbp < 120 & dbp < 80, 7,
                                              ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 7,
                                                     ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 5,
                                                            ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 6,
                                                                   ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 8,
                                                                          ifelse(sbp >= 180 | dbp >= 110, 5, NA)))))))

  data$bp <- bp_score


  data$egfr <- as.character(cut(data$egfr, breaks = c(-Inf, 45, 60, 90, Inf), labels = c(11,2,0,1), right = FALSE))
  data$t2dm <- ifelse(data$t2dm == 1, 4, 0)
  data$male <- ifelse(data$male == 1, 6, 0)
  data$smoking <- ifelse(data$smoking == 1, 9, 0)


  # Calculate total score
  data$totalscore <- as.numeric(data$bmi) + as.numeric(data$hdl) + as.numeric(data$bp) + as.numeric(data$egfr) + as.numeric(data$t2dm) + as.numeric(data$male) + as.numeric(data$smoking)

  data$totalscore <- ifelse(data$totalscore < 0, 0,
                            ifelse(data$totalscore > 60, 60,data$totalscore))

  # Create a character vector of total scores ranging from 0 to 60.
  score_range <- as.character(0:60)


  # Define the 5y absolute risk values corresponding to each total score across age group.

  ## risk map for 40-49y person
  risk_values40 <- c(0.07,   0.07,   0.08,   0.08,   0.09,
                     0.09,   0.10,   0.11,   0.11,   0.12,
                     0.13,   0.14,   0.15,   0.16,   0.17,
                     0.19,   0.20,   0.21,   0.23,   0.25,
                     0.26,   0.28,   0.30,   0.32,   0.35,
                     0.37,   0.40,   0.43,   0.46,   0.49,
                     0.52,   0.56,   0.60,   0.65,   0.69,
                     0.74,   0.79,   0.85,   0.91,   0.98,
                     1.05,   1.12,   1.20,   1.29,   1.38,
                     1.48,   1.58,   1.70,   1.82,   1.94,
                     2.08,   2.23,   2.39,   2.56,   2.74,
                     2.93,   3.14,   3.36,   3.60,   3.85, 4.12  )

  risk_map40 <- setNames(risk_values40, score_range)



  ## risk map for 50-59y person
  risk_values50 <- c(0.10,   0.11,   0.11,   0.12,   0.13,
                     0.14,   0.15,   0.16,   0.17,   0.19,
                     0.20,   0.21,   0.23,   0.25,   0.26,
                     0.28,   0.30,   0.32,   0.35,   0.37,
                     0.40,   0.43,   0.46,   0.49,   0.53,
                     0.56,   0.60,   0.65,   0.69,   0.74,
                     0.80,   0.85,   0.91,   0.98,   1.05,
                     1.12,   1.21,   1.29,   1.38,   1.48,
                     1.59,   1.70,   1.82,   1.95,   2.09,
                     2.24,   2.40,   2.57,   2.75,   2.94,
                     3.15,   3.37,   3.61,   3.86,   4.13,
                     4.42,   4.73,   5.07,   5.42,   5.80,   6.20  )

  risk_map50 <- setNames(risk_values50, score_range)



  ## risk map for 60-69y person
  risk_values60 <- c(0.20,   0.22,   0.23,   0.25,   0.27,
                     0.29,   0.31,   0.33,   0.35,   0.38,
                     0.40,   0.43,   0.46,   0.50,   0.53,
                     0.57,   0.61,   0.66,   0.70,   0.75,
                     0.81,   0.86,   0.93,   0.99,   1.06,
                     1.14,   1.22,   1.31,   1.40,   1.50,
                     1.61,   1.72,   1.84,   1.98,   2.12,
                     2.27,   2.43,   2.60,   2.78,   2.98,
                     3.19,   3.41,   3.66,   3.91,   4.19,
                     4.48,   4.79,   5.13,   5.49,   5.87,
                     6.28,   6.71,   7.18,   7.67,   8.20,
                     8.76,   9.36,  10.00,  10.67,  11.40,   12.16   )

  risk_map60 <- setNames(risk_values60, score_range)



  ## risk map for 70-79y person
  risk_values70 <- c(0.75,   0.80,   0.86,   0.92,   0.99,
                     1.06,   1.13,   1.21,   1.30,   1.39,
                     1.49,   1.60,   1.71,   1.83,   1.96,
                     2.10,   2.25,   2.41,   2.58,   2.77,
                     2.96,   3.17,   3.39,   3.63,   3.89,
                     4.16,   4.45,   4.77,   5.10,   5.45,
                     5.83,   6.24,   6.67,   7.13,   7.63,
                     8.15,   8.71,   9.30,   9.94,  10.61,
                     11.33,  12.09,  12.90,  13.76,  14.67,
                     15.64,  16.66,  17.74,  18.89,  20.10,
                     21.37,  22.72,  24.14,  25.63,  27.19,
                     28.83,  30.55,  32.34,  34.21,  36.16,  38.18  )

  risk_map70 <- setNames(risk_values70, score_range)



  ## risk map for 80-89y person
  risk_values80 <- c(3.13,   3.36,   3.59,   3.84,   4.11,
                     4.40,   4.71,   5.04,   5.39,   5.77,
                     6.17,   6.60,   7.05,   7.54,   8.06,
                     8.61,   9.20,   9.83,  10.50,  11.21,
                     11.96,  12.76,  13.61,  14.51,  15.47,
                     16.49,  17.56,  18.69,  19.89,  21.16,
                     22.49,  23.89,  25.37,  26.92,  28.55,
                     30.25,  32.03,  33.89,  35.83,  37.84,
                     39.92,  42.08,  44.31,  46.60,  48.95,
                     51.35,  53.80,  56.30,  58.82,  61.36,
                     63.91,  66.45,  68.98,  71.48,  73.94,
                     76.33,  78.66,  80.90,  83.04,  85.07,   86.97  )

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
