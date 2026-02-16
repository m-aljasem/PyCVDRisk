epoch_stroke <- function(data) {
  required_cols <- c("age", "urineprotein", "sbp", "t2dm", "smoking")

  # Check for required columns
  if (!all(required_cols %in% names(data))) {
    stop("Data frame must contain the following columns: ", paste(required_cols, collapse = ", "))
  }

  # Check for NA values
  if (any(is.na(data[required_cols]))) {
    warning("Data frame contains NA values in required columns.")
  }

  # Check for age conditions
  if (any(data$age >= 80)) {
    warning("Some individuals are aged 80 or older. results may not accurate.")
  }

  if (any(data$age < 40)) {
    warning("Some individuals are younger than 40. results may not accurate.")
  }

  # Convert columns to numeric
  data[required_cols] <- lapply(data[required_cols], as.numeric)

  # Define variables
  base <- 0.9961
  data$urineprotein <- ifelse(data$urineprotein == 1, 1, 0)
  data$smoking <- ifelse(data$smoking == 1, 1, 0)

  # Calculate derived variables
  meanage <- 57.0
  meansbp <- 131.1
  propdm <- 0.046
  propurineprotein <- 0.028
  propsmoking <- 0.267

  betamean <- (37.40606 * log(meanage) +
                 8.24292 * propsmoking +
                 26.39953 * log(meansbp) +
                 0.45679 * propdm +
                 0.63621 * propurineprotein -
                 5.92290 * (log(meanage) * log(meansbp)) -
                 1.84505 * (log(meanage) * propsmoking))

  # Calculate risk for each individual
  data$risk <- 1 - base^exp((37.40606 * log(data$age) +
                               8.24292 * data$smoking +
                               26.39953 * log(data$sbp) +
                               0.45679 * data$t2dm +
                               0.63621 * data$urineprotein -
                               5.92290 * (log(data$age) * log(data$sbp)) -
                               1.84505 * (log(data$age) * data$smoking) -
                               betamean))

  return(data$risk)
}
