# Cut unmodified from cliot 1.0.0 (CRAN), R/neelpackage.R lines 9652-9725, 10052-10149, 20577-20671.
# Tarball https://cran.r-project.org/src/contrib/cliot_1.0.0.tar.gz, sha256 83794e0c8a3f15434a61678558bcc19e30bc42139122f797183727ac019aa1bd, downloaded 2026-11-23. GPL-3.
chads2_score <- function(chf, hypertension, age, diabetes, stroke_tia) {

  chf <- as.numeric(chf)
  hypertension <- as.numeric(hypertension)
  age <- as.numeric(age)
  diabetes <- as.numeric(diabetes)
  stroke_tia <- as.numeric(stroke_tia)

  if (any(is.na(c(chf, hypertension, age, diabetes, stroke_tia)))) {
    stop("Error: All inputs must be numeric.")
  }

  score <- 0

  if (chf == 1) {
    score <- score + 1
  }

  if (hypertension == 1) {
    score <- score + 1
  }

  if (age >= 75) {
    score <- score + 1
  }

  if (diabetes == 1) {
    score <- score + 1
  }

  if (stroke_tia == 1) {
    score <- score + 2
  }

  risk_level <- ""
  stroke_risk <- ""

  if (score == 0) {
    risk_level <- "Low Risk"
    stroke_risk <- "1.9%"
  } else if (score == 1) {
    risk_level <- "Moderate Risk"
    stroke_risk <- "2.8%"
  } else if (score == 2) {
    risk_level <- "Moderate Risk"
    stroke_risk <- "4.0%"
  } else if (score == 3) {
    risk_level <- "High Risk"
    stroke_risk <- "5.9%"
  } else if (score == 4) {
    risk_level <- "High Risk"
    stroke_risk <- "8.5%"
  } else if (score == 5) {
    risk_level <- "High Risk"
    stroke_risk <- "12.5%"
  } else {
    risk_level <- "High Risk"
    stroke_risk <- "18.2%"
  }

  return(list(
    CHADS2_Score = score,
    Risk_Category = risk_level,
    Annual_Stroke_Risk = stroke_risk,
    Inputs = list(
      CHF = chf == 1,
      Hypertension = hypertension == 1,
      Age_ge_75 = age >= 75,
      Diabetes = diabetes == 1,
      Stroke_TIA = stroke_tia == 1
    )
  ))
}

cha2ds2_vasc_score <- function(age, sex, chf_history, hypertension_history, stroke_tia_history, vascular_disease_history, diabetes_history) {

  age <- as.numeric(age)
  sex <- tolower(as.character(sex))
  chf_history <- as.numeric(chf_history)
  hypertension_history <- as.numeric(hypertension_history)
  stroke_tia_history <- as.numeric(stroke_tia_history)
  vascular_disease_history <- as.numeric(vascular_disease_history)
  diabetes_history <- as.numeric(diabetes_history)

  if (any(is.na(c(age, chf_history, hypertension_history, stroke_tia_history, vascular_disease_history, diabetes_history)))) {
    stop("Error: All numeric/binary inputs must be valid.")
  }

  if (!sex %in% c("male", "female")) {
    stop("Error: Sex must be 'male' or 'female'.")
  }

  score <- 0

  if (chf_history == 1) {
    score <- score + 1
  }

  if (hypertension_history == 1) {
    score <- score + 1
  }

  if (age >= 75) {
    score <- score + 2
  } else if (age >= 65) {
    score <- score + 1
  }

  if (diabetes_history == 1) {
    score <- score + 1
  }

  if (stroke_tia_history == 1) {
    score <- score + 2
  }

  if (vascular_disease_history == 1) {
    score <- score + 1
  }

  if (sex == "female") {
    score <- score + 1
  }

  stroke_risk <- ""
  recommendation <- ""

  if (score == 0) stroke_risk <- "0%"
  else if (score == 1) stroke_risk <- "1.3%"
  else if (score == 2) stroke_risk <- "2.2%"
  else if (score == 3) stroke_risk <- "3.2%"
  else if (score == 4) stroke_risk <- "4.0%"
  else if (score == 5) stroke_risk <- "6.7%"
  else if (score == 6) stroke_risk <- "9.8%"
  else if (score == 7) stroke_risk <- "9.6%"
  else if (score == 8) stroke_risk <- "6.7%"
  else stroke_risk <- "15.2%"

  if (sex == "male") {
    if (score == 0) {
      recommendation <- "Low risk. No anticoagulation recommended."
    } else if (score == 1) {
      recommendation <- "Moderate risk. Consider oral anticoagulation."
    } else {
      recommendation <- "High risk. Oral anticoagulation recommended."
    }
  } else {
    if (score == 1) {
      recommendation <- "Low risk. No anticoagulation recommended."
    } else if (score == 2) {
      recommendation <- "Moderate risk. Consider oral anticoagulation."
    } else {
      recommendation <- "High risk. Oral anticoagulation recommended."
    }
  }

  return(list(
    CHA2DS2_VASc_Score = score,
    Annual_Stroke_Risk_No_Anticoagulation = stroke_risk,
    Recommendation = recommendation,
    Inputs = list(
      Age = age,
      Sex = sex,
      CHF = chf_history == 1,
      Hypertension = hypertension_history == 1,
      Stroke_TIA = stroke_tia_history == 1,
      Vascular_Disease = vascular_disease_history == 1,
      Diabetes = diabetes_history == 1
    )
  ))
}

has_bled_score <- function(hypertension, abnormal_renal_function, abnormal_liver_function, stroke_history, bleeding_history, labile_inr, elderly_gt_65, drug_usage, alcohol_usage) {

  hypertension <- as.numeric(hypertension)
  abnormal_renal_function <- as.numeric(abnormal_renal_function)
  abnormal_liver_function <- as.numeric(abnormal_liver_function)
  stroke_history <- as.numeric(stroke_history)
  bleeding_history <- as.numeric(bleeding_history)
  labile_inr <- as.numeric(labile_inr)
  elderly_gt_65 <- as.numeric(elderly_gt_65)
  drug_usage <- as.numeric(drug_usage)
  alcohol_usage <- as.numeric(alcohol_usage)

  if (any(is.na(c(hypertension, abnormal_renal_function, abnormal_liver_function, stroke_history, bleeding_history, labile_inr, elderly_gt_65, drug_usage, alcohol_usage)))) {
    stop("Error: All inputs must be numeric (0 or 1).")
  }

  score <- 0

  if (hypertension == 1) {
    score <- score + 1
  }

  if (abnormal_renal_function == 1) {
    score <- score + 1
  }

  if (abnormal_liver_function == 1) {
    score <- score + 1
  }

  if (stroke_history == 1) {
    score <- score + 1
  }

  if (bleeding_history == 1) {
    score <- score + 1
  }

  if (labile_inr == 1) {
    score <- score + 1
  }

  if (elderly_gt_65 == 1) {
    score <- score + 1
  }

  if (drug_usage == 1) {
    score <- score + 1
  }

  if (alcohol_usage == 1) {
    score <- score + 1
  }

  bleed_risk <- ""
  recommendation <- ""

  if (score == 0) {
    bleed_risk <- "1.13 bleeds per 100 patient-years (Low Risk)"
    recommendation <- "Anticoagulation generally safe."
  } else if (score == 1) {
    bleed_risk <- "1.02 bleeds per 100 patient-years (Low Risk)"
    recommendation <- "Anticoagulation generally safe."
  } else if (score == 2) {
    bleed_risk <- "1.88 bleeds per 100 patient-years (Moderate Risk)"
    recommendation <- "Anticoagulation can be considered; address reversible risk factors."
  } else if (score == 3) {
    bleed_risk <- "3.74 bleeds per 100 patient-years (High Risk)"
    recommendation <- "Caution advised; frequent monitoring and correction of risk factors required."
  } else if (score == 4) {
    bleed_risk <- "8.70 bleeds per 100 patient-years (High Risk)"
    recommendation <- "High risk of bleeding; carefully weigh benefits vs. risks."
  } else {
    bleed_risk <- ">12.5 bleeds per 100 patient-years (Very High Risk)"
    recommendation <- "Very high risk; consider alternatives or extremely close monitoring."
  }

  return(list(
    HAS_BLED_Score = score,
    Annual_Bleeding_Risk = bleed_risk,
    Recommendation = recommendation,
    Inputs = list(
      Hypertension = hypertension == 1,
      Renal_Disease = abnormal_renal_function == 1,
      Liver_Disease = abnormal_liver_function == 1,
      Stroke_History = stroke_history == 1,
      Prior_Bleeding = bleeding_history == 1,
      Labile_INR = labile_inr == 1,
      Elderly_gt_65 = elderly_gt_65 == 1,
      Drugs_NSAIDs_Antiplatelets = drug_usage == 1,
      Alcohol_Use = alcohol_usage == 1
    )
  ))
}

