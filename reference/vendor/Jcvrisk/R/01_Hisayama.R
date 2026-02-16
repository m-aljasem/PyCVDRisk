hisayama <- function(data) {
  # Check if all required columns are present in the data frame
  required_cols <- c("age", "male", "sbp", "t2dm", "hdl", "ldl", "urineprotein", "smoking", "exercise")
  if (!all(required_cols %in% names(data))) {
    stop("Data frame must contain the following columns: ", paste(required_cols, collapse = ", "))
  }

  # Check for NA values in required columns
  if (any(sapply(data[required_cols], is.na))) {
    warning("Data frame contains NA values in required columns.")
  }

  # Check for ages over 85
  if (any(data$age >= 85)) {
    warning("Some individuals are aged 85 or older. results may not accurate")
  }

  # Check for ages under 40
  if (any(data$age < 40)) {
    warning("Some individuals are younger than 40. results may not accurate")
  }

  # Convert columns to numeric
  data$age <- as.numeric(data$age)
  data$male <- as.numeric(data$male)
  data$sbp <- as.numeric(data$sbp)
  data$t2dm <- as.numeric(data$t2dm)
  data$hdl <- as.numeric(data$hdl)
  data$ldl <- as.numeric(data$ldl)
  data$urineprotein <- as.numeric(data$urineprotein)
  data$smoking <- as.numeric(data$smoking)
  data$exercise <- as.numeric(data$exercise)

  # Calculate scores
  data$male <- ifelse(data$male == 1, 7, 0)
  data$sbp <- as.character(cut(data$sbp, breaks = c(-Inf, 120, 130, 140, 160, Inf), labels = 0:4, right = FALSE))
  data$hdl <- as.character(cut(data$hdl, breaks = c(-Inf, 40, 60, Inf), labels = 2:0, right = FALSE))
  data$ldl <- as.character(cut(data$ldl, breaks = c(-Inf, 120, 140, 160, Inf), labels = 0:3, right = FALSE))
  data$urineprotein <- ifelse(data$urineprotein == 1, 4, 0)
  data$t2dm <- ifelse(data$t2dm == 1, 3, 0)
  data$smoking <- ifelse(data$smoking == 1, 2, 0)
  data$exercise <- ifelse(data$exercise == 1, 0, 2)

  data$age_rank <- as.numeric(as.character(cut(data$age, breaks = c(-Inf, 50, 60, 70, 80, 85, Inf), labels = c(0, 5, 11, 16, 20, 20), right = FALSE)))

  # Calculate total score
  data$totalscore <- as.numeric(data$male) + as.numeric(data$sbp) + as.numeric(data$hdl) + as.numeric(data$ldl) + as.numeric(data$urineprotein) + as.numeric(data$t2dm) + as.numeric(data$smoking) + as.numeric(data$exercise)

  # Calculate risk
  risk <- 1 - 0.9696^exp((data$totalscore + data$age_rank) * 0.144 - 2.4767)

  return(risk)
}
