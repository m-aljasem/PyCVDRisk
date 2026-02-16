\name{suita_wECG}
\alias{suita_wECG}
\title{10-year risk calculation for CVD events based on the Suita study with electrocardiogram model}
\usage{
suita_wECG(data)
}
\arguments{
    \item{data}{A data frame containing the variables to estimate risk.}
}

\description{
 suita_ecg() calculates the 10-year probability of developing the first ASCVD event.
 Mandatory variables: "age", "male", "sbp", "dbp", "ht_medication", "t2dm", "tc", "hdl", "ldl", "urineprotein", "smoking", "af", "lvh".
 These variables must be applied with the above spelling.

 The details of this prediction model should be referred to the original article.
 "Development of Cardiovascular Disease Risk Prediction Model Using the Suita Study, a Population-Based Prospective Cohort Study in Japan"

 DOI:10.5551/jat.48843

 PMID:32023562

 PMID:PMC7803836
}
\value{
Based on the mandatory variables, a risk probability will be calculated for each line.
The primary cautions are as follows,

1."Data frame must contain the following columns.":

If you contact this warning message, the data frame did not contain required variables. Please check the data frame contain the mandatory variables in the designated spelling.

2."Data frame contains NA values in required columns.":

suita_wECG() returns the NA if the required variables were missing, or NAs were generated in the calculation process.

3."Some individuals are aged 80 or older. results may not be accurate.":

4."Some individuals are aged 30 or younger. results may not be accurate.":

This risk prediction model was developed based on the Suita study participants(older than 30 and younger than 80). In the calculation process, when the participants' age is outside of the range, the  result is returned as follows.

1. Younger than 30: same value for 30-40.

2. Older than 80: same value for 75-79.
}

\examples{

data <- data.frame(
  age = c(45, 55, 90),
  male = c(1, 0, 1),
  sbp = c(120, 130, 145),
  dbp = c(80, 90, 110),
  ht_medication = c(0, 1, 0),
  t2dm = c(0, 1, 0),
  tc = c(180, 160, 210),
  hdl = c(55, 45, 35),
  ldl = c(110, 100, 160),
  urineprotein = c(1, 1, 0),
  smoking = c(0, 1, 0),
  af = c(1, 0, 1),
  lvh = c(1, 0, 1)
)

# age           :age(continuous)
# male          :male(category: 0 = female, 1 = male)
# sbp           :systolic blood pressure(continuous)
# dbp           :diastolic blood pressure(continuous)
# ht_medication :hypertensive medication user(category:0 = non-user,1 = user)
# t2dm          :diabetes mellitus(category: 0 = non-prevalent DM, 1 = prevalent DM)
# tc            :total cholesterol(continuous)
# hdl           :high-density lipoprotein cholesterol(continuous)
# ldl           :low-density lipoprotein cholesterol(continuous)
# urineprotein  :proteinuria(category: 0 = negative, 1 = positive)
# smoking       :smoking status(category: 0 = never/past smoker, 1 = current smoker)
# af            :atrial fibrillation(category: 0 = negative, 1 = positive)
# lvh           :left ventricular hypertrophy(category: 0 = negative, 1 = positive)

suita_wECG(data)
}
