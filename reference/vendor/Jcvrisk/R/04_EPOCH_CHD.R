epoch_chd <- function(data) {
  required_cols <- c("age", "male", "urineprotein", "sbp", "t2dm", "tc", "hdl", "smoking")

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
    warning("Some individuals are aged 80 or older. results may not accurate")
  }

  if (any(data$age < 40)) {
    warning("Some individuals are younger than 40. results may not accurate")
  }

  # Convert columns to numeric
  data[required_cols] <- lapply(data[required_cols], as.numeric)

  # Define variables
  base <- 0.9981
  data$male <- ifelse(data$male == 1, 1, 0)
  data$urineprotein <- ifelse(data$urineprotein == 1, 1, 0)
  data$smoking <- ifelse(data$smoking == 1, 1, 0)

  # Calculate derived variables
  data$tchdl <- data$tc / data$hdl
  meanage <- 57.0
  meansbp <- 131.1
  meancho <- 4.1
  propmale <- 0.480
  propdm <- 0.046
  propurineprotein <- 0.028
  propsmoking <- 0.267

  betamean <- (61.19918 * log(meanage) +
                 15.36389 * propsmoking +
                 0.65869 * propmale +
                 46.36999 * log(meansbp) +
                 0.56252 * propdm +
                 0.35931 * log(meancho) +
                 0.58243 * propurineprotein -
                 10.61448 * (log(meanage) * log(meansbp)) -
                 3.51974 * (log(meanage) * propsmoking))

  # Calculate risk for each individual
  data$risk <- 1 - base^exp((61.19918 * log(data$age) +
                               15.36389 * data$smoking +
                               0.65869 * data$male +
                               46.36999 * log(data$sbp) +
                               0.56252 * data$t2dm +
                               0.58243 * data$urineprotein +
                               0.35931 * log(data$tc) -
                               10.61448 * (log(data$age) * log(data$sbp)) -
                               3.51974 * (log(data$age) * data$smoking) -
                               betamean))

  return(data$risk)
}
