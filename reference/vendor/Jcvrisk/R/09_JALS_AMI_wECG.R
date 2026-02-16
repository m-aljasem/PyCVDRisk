jals_ami_wECG <- function(data) {
  # Check if all required columns are present in the data frame
  required_cols <- c("age", "male", "sbp", "dbp", "t2dm", "hdl", "tc", "egfr", "smoking", "ht_medication", "af")
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
  data$tc <- as.numeric(data$tc)
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
  data$af <- ifelse(data$af == 1, 12, 0)

  data$non_hdl <- as.character(cut(data$tc - data$hdl, breaks = c(-Inf, 130, 150, 170, Inf), labels = c(0, 10, 13, 17), right = FALSE))

  data$hdl <- as.character(cut(data$hdl, breaks = c(-Inf, 40, 60, Inf), labels = c(11, 7, 0), right = FALSE))


  # Index for NA in medication
  idx_na <- is.na(data$ht_medication)

  # bp_medication == 0
  idx0 <- data$ht_medication == 0 & !idx_na
  bp_score[idx0]  <- with(data[idx0,],
                          ifelse(sbp <  120 & dbp < 80, 0,
                                 ifelse(sbp >= 120 & sbp <= 129 & dbp <   80, 4,
                                        ifelse(sbp >= 130 & sbp <= 139 | dbp >=  80 & dbp <=  89, 4,
                                               ifelse(sbp >= 140 & sbp <= 159 | dbp >=  90 & dbp <=  99, 7,
                                                      ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 9,
                                                             ifelse(sbp >= 180 | dbp >= 110, 17, NA)))))))

  # bp_medication == 1
  idx1 <- data$ht_medication == 1 & !idx_na
  bp_score[idx1] <- with(data[idx1, ], ifelse(sbp < 120 & dbp < 80, 2,
                                              ifelse(sbp >= 120 & sbp <= 129 & dbp < 80, 15,
                                                     ifelse(sbp >= 130 & sbp <= 139 | dbp >= 80 & dbp <= 89, 16,
                                                            ifelse(sbp >= 140 & sbp <= 159 | dbp >= 90 & dbp <= 99, 11,
                                                                   ifelse(sbp >= 160 & sbp <= 179 | dbp >= 100 & dbp <= 109, 6,
                                                                          ifelse(sbp >= 180 | dbp >= 110, 9, NA)))))))

  data$bp <- bp_score


  data$egfr <- as.character(cut(data$egfr, breaks = c(-Inf, 45, 60, 90, Inf), labels = c(3, 3, 0, 0), right = FALSE))
  data$t2dm <- ifelse(data$t2dm == 1, 9, 0)
  data$male <- ifelse(data$male == 1, 17, 0)
  data$smoking <- ifelse(data$smoking == 1, 11, 0)


  data$totalscore <- as.numeric(data$af) +as.numeric(data$non_hdl) + as.numeric(data$hdl) + as.numeric(data$bp) + as.numeric(data$egfr) + as.numeric(data$t2dm) + as.numeric(data$male) + as.numeric(data$smoking)

  data$totalscore <- ifelse(data$totalscore < 0, 0,
                            ifelse(data$totalscore > 60, 60,data$totalscore))

  # Create a character vector of total scores ranging from 0 to 60.
  score_range <- as.character(0:60)



  # Define the 5y absolute risk values corresponding to each total score across age group.

  ## risk map for 40-49y person
  risk_values40 <- c(0.01,   0.01,   0.01,   0.01,   0.01,
                     0.01,   0.01,   0.01,   0.01,   0.01,
                     0.01,   0.01,   0.01,   0.02,   0.02,
                     0.02,   0.02,   0.02,   0.02,   0.02,
                     0.03,   0.03,   0.03,   0.03,   0.03,
                     0.04,   0.04,   0.04,   0.05,   0.05,
                     0.05,   0.06,   0.06,   0.06,   0.07,
                     0.07,   0.08,   0.08,   0.09,   0.10,
                     0.10,   0.11,   0.12,   0.13,   0.14,
                     0.15,   0.16,   0.17,   0.18,   0.19,
                     0.21,   0.22,   0.24,   0.26,   0.28,
                     0.29,   0.32,   0.34,   0.36,   0.39,  0.42  )

  risk_map40 <- setNames(risk_values40, score_range)



  ## risk map for 50-59y person
  risk_values50 <- c(0.01,   0.02,   0.02,   0.02,   0.02,
                     0.02,   0.02,   0.02,   0.02,   0.03,
                     0.03,   0.03,   0.03,   0.03,   0.04,
                     0.04,   0.04,   0.05,   0.05,   0.05,
                     0.06,   0.06,   0.07,   0.07,   0.07,
                     0.08,   0.09,   0.09,   0.10,   0.11,
                     0.11,   0.12,   0.13,   0.14,   0.15,
                     0.16,   0.17,   0.18,   0.20,   0.21,
                     0.23,   0.24,   0.26,   0.28,   0.30,
                     0.32,   0.34,   0.37,   0.39,   0.42,
                     0.45,   0.49,   0.52,   0.56,   0.60,
                     0.64,   0.69,   0.73,   0.79,   0.84, 0.90  )

  risk_map50 <- setNames(risk_values50, score_range)



  ## risk map for 60-69y person
  risk_values60 <- c(0.02,   0.02,   0.02,   0.02,   0.02,
                     0.02,   0.02,   0.03,   0.03,   0.03,
                     0.03,   0.03,   0.04,   0.04,   0.04,
                     0.05,   0.05,   0.05,   0.06,   0.06,
                     0.06,   0.07,   0.07,   0.08,   0.09,
                     0.09,   0.10,   0.11,   0.11,   0.12,
                     0.13,   0.14,   0.15,   0.16,   0.17,
                     0.18,   0.20,   0.21,   0.23,   0.24,
                     0.26,   0.28,   0.30,   0.32,   0.34,
                     0.37,   0.39,   0.42,   0.45,   0.48,
                     0.52,   0.55,   0.59,   0.64,   0.68,
                     0.73,   0.78,   0.84,   0.90,   0.96, 1.03   )
  risk_map60 <- setNames(risk_values60, score_range)



  ## risk map for 70-79y person
  risk_values70 <- c(0.03,   0.03,   0.03,   0.03,   0.04,
                     0.04,   0.04,   0.04,   0.05,   0.05,
                     0.05,   0.06,   0.06,   0.07,   0.07,
                     0.08,   0.08,   0.09,   0.10,   0.10,
                     0.11,   0.12,   0.13,   0.13,   0.14,
                     0.15,   0.17,   0.18,   0.19,   0.20,
                     0.22,   0.23,   0.25,   0.27,   0.29,
                     0.31,   0.33,   0.35,   0.38,   0.41,
                     0.44,   0.47,   0.50,   0.54,   0.57,
                     0.62,   0.66,   0.71,   0.76,   0.81,
                     0.87,   0.93,   1.00,   1.07,   1.15,
                     1.23,   1.32,   1.41,   1.51,   1.62, 1.73 )
  risk_map70 <- setNames(risk_values70, score_range)



  ## risk map for 80-89y person
  risk_values80 <- c(0.04,   0.05,   0.05,   0.05,   0.06,
                     0.06,   0.07,   0.07,   0.08,   0.08,
                     0.09,   0.09,   0.10,   0.11,   0.11,
                     0.12,   0.13,   0.14,   0.15,   0.16,
                     0.17,   0.19,   0.20,   0.21,   0.23,
                     0.25,   0.26,   0.28,   0.30,   0.32,
                     0.35,   0.37,   0.40,   0.43,   0.46,
                     0.49,   0.52,   0.56,   0.60,   0.65,
                     0.69,   0.74,   0.79,   0.85,   0.91,
                     0.98,   1.05,   1.12,   1.20,   1.29,
                     1.38,   1.48,   1.58,   1.69,   1.81,
                     1.94,   2.08,   2.23,   2.39,   2.56, 2.74 )

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
