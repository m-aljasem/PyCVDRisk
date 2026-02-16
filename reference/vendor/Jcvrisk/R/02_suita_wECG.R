suita_wECG <- function(data) {
  required_cols <- c("age", "male", "sbp", "dbp","ht_medication", "t2dm", "tc", "hdl", "ldl", "urineprotein", "smoking", "af", "lvh")

  # Check for required columns
  if (!all(required_cols %in% names(data))) {
    stop("Data frame must contain the following columns: ", paste(required_cols, collapse = ", "))
  }

  # Check for NA values
  if (any(is.na(data[required_cols]))) {
    warning("Data frame contains NA values in required columns.")
  }

  # Check for ages over 80
  if (any(data$age >= 80)) {
    warning("Some individuals are aged 80 or older. Risk prediction may not correct")
  }

  # Check for ages under 30
  if (any(data$age < 30)) {
    warning("Some individuals are aged 30 or younger. Risk prediction may not correct")
  }

  # Convert columns to numeric
  data[required_cols] <- lapply(data[required_cols], as.numeric)

  nonhdl <- data$tc - data$hdl

  # Calculate scores
  data$male_score <- ifelse(data$male == 1, 4, 0)
  data$bp_score <- ifelse(data$sbp >= 160 | data$dbp >= 100 | data$ht_medication == 1 , 6,
                          ifelse((data$sbp >= 140 & data$sbp < 160) | (data$dbp >= 90 & data$dbp < 100), 3,
                                 ifelse((data$sbp >= 120 & data$sbp < 139) | (data$dbp >= 80 & data$dbp < 89), 0, -4)))
  data$hdl_score <- ifelse(data$hdl < 40, 0,
                           ifelse(data$hdl >= 40 & data$hdl < 60, -2, -4))
  data$cho_score <- ifelse((nonhdl < 170 & data$ldl < 140), 0, 2)
  data$urineprotein_score <- ifelse(data$urineprotein == 1, 2, 0)
  data$t2dm_score <- ifelse(data$t2dm == 1, 6, 0)
  data$smoking_score <- ifelse(data$smoking == 1, 4, 0)
  data$af_score <- ifelse(data$af == 1, 8, 0)
  data$lvh_score <- ifelse(data$lvh == 1, 5, 0)

  data$age_rank <- ifelse(data$age >= 30 & data$age < 40, 0,
                          ifelse(data$age >= 40 & data$age < 50, 8,
                                 ifelse(data$age >= 50 & data$age < 60, 14,
                                        ifelse(data$age >= 60 & data$age < 65, 18,
                                               ifelse(data$age >= 65 & data$age < 70, 22,
                                                      ifelse(data$age >= 70 & data$age < 75, 25,
                                                             ifelse(data$age >= 75 & data$age < 80, 28, 28)))))))

  # Total score
  data$totalscore <- data$male_score + data$bp_score + data$hdl_score + data$cho_score +
    data$urineprotein_score + data$t2dm_score + data$smoking_score +
    data$age_rank + data$af_score + data$lvh_score

  # Calculate 10-year risk
  data$ten_year_risk <- ifelse(data$totalscore <= 0, 1,
                               ifelse(data$totalscore >= 1 & data$totalscore <= 20, 2,
                                      ifelse(data$totalscore >= 21 & data$totalscore <= 25, 6,
                                             ifelse(data$totalscore >= 26 & data$totalscore <= 30, 9,
                                                    ifelse(data$totalscore >= 31 & data$totalscore <= 35, 15,
                                                           ifelse(data$totalscore >= 36, 26, NA))))))

  return(data$ten_year_risk)
}
