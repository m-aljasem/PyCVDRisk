epoch_cvd <- function(data) {
  required_cols <- c("age", "male", "urineprotein", "sbp", "t2dm", "smoking")

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
  base <- 0.9942
  data$male <- ifelse(data$male == 1, 1, 0)
  data$urineprotein <- ifelse(data$urineprotein == 1, 1, 0)
  data$smoking <- ifelse(data$smoking == 1, 1, 0)

  # Calculate derived variables
  meanage <- 57.0
  meansbp <- 131.1
  propmale <- 0.480
  propdm <- 0.046
  propurineprotein <- 0.028
  propsmoking <- 0.267

  betamean <- (45.54988 * log(meanage) +
                 10.64932 * propsmoking +
                 0.30093 * propmale +
                 33.30729 * log(meansbp) +
                 0.49413 * propdm +
                 0.62120 * propurineprotein -
                 7.54461 * (log(meanage) * log(meansbp)) -
                 2.41998 * (log(meanage) * propsmoking))

  # Calculate risk for each individual
  data$risk <- 1 - base^exp((45.54988 * log(data$age) +
                               10.64932 * data$smoking +
                               0.30093 * data$male +
                               33.30729 * log(data$sbp) +
                               0.49413 * data$t2dm +
                               0.62120 * data$urineprotein -
                               7.54461 * (log(data$age) * log(data$sbp)) -
                               2.41998 * (log(data$age) * data$smoking)) - betamean)

  return(data$risk)
}
