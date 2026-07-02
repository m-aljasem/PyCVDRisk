  diaphoresis <- as.numeric(diaphoresis)
  radiation <- as.numeric(radiation)
  pain_inspiration <- as.numeric(pain_inspiration)
  pain_palpation <- as.numeric(pain_palpation)
  ekg_ischemia_present <- as.numeric(ekg_ischemia_present)
  troponin_normal <- as.numeric(troponin_normal)

  if (any(is.na(c(age, known_cad_or_3_risk_factors, diaphoresis, radiation, pain_inspiration, pain_palpation)))) {
    stop("Error: All clinical inputs must be numeric.")
  }

  if (!sex %in% c("male", "female")) {
    stop("Error: Sex must be 'male' or 'female'.")
  }

  if (age < 18) {
    return(list(Error = "EDACS is validated for patients >= 18 years old."))
  }

  score <- 0

  if (age >= 18 && age <= 45) {
    score <- score + 2
  } else if (age >= 46 && age <= 50) {
    score <- score + 4
  } else if (age >= 51 && age <= 55) {
    score <- score + 6
  } else if (age >= 56 && age <= 60) {
    score <- score + 8
  } else if (age >= 61 && age <= 65) {
    score <- score + 10
  } else if (age >= 66 && age <= 70) {
    score <- score + 12
  } else if (age >= 71 && age <= 75) {
    score <- score + 14
  } else if (age >= 76 && age <= 80) {
    score <- score + 16
  } else if (age >= 81 && age <= 85) {
    score <- score + 18
  } else {
    score <- score + 20
  }

  if (sex == "male") {
    score <- score + 6
  }

  if (known_cad_or_3_risk_factors == 1) {
    score <- score + 4
  }

  if (diaphoresis == 1) {
    score <- score + 3
  }

  if (radiation == 1) {
    score <- score + 5
  }

  if (pain_inspiration == 1) {
    score <- score - 4
  }

  if (pain_palpation == 1) {
    score <- score - 6
  }

  protocol_result <- ""
  action <- ""

  if (score < 16 && ekg_ischemia_present == 0 && troponin_normal == 1) {
    protocol_result <- "Low Risk"
    action <- "Safe for early discharge to outpatient follow-up (Risk of Major Adverse Cardiac Event < 1%)."
  } else {
    protocol_result <- "Not Low Risk"
    action <- "Proceed with usual care (observation, serial troponins, further testing)."
  }

  return(list(
    EDACS_Score = score,
    Risk_Category = protocol_result,
    Recommendation = action,
    Inputs = list(
      Age = age,
      Sex = sex,
      History_Risk = known_cad_or_3_risk_factors == 1,
      Diaphoresis = diaphoresis == 1,
      Radiation = radiation == 1,
      Inspiration_Pain = pain_inspiration == 1,
      Palpation_Pain = pain_palpation == 1,
      EKG_Ischemia = ekg_ischemia_present == 1,
      Troponin_Normal = troponin_normal == 1
    )
  ))
}
