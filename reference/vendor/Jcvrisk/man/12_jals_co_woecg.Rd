\name{jals_composite_woECG}
\alias{jals_composite_woECG}
\title{5-year risk calculation for composite outcome (stroke + acute myocardial infarction) based on the JALS study without electrocardiogram model}
\usage{
jals_composite_woECG(data)
}
\arguments{
    \item{data}{A data frame containing the variables to estimate risk.}
}

\description{
 jals_composite_woECG() calculates the 5-year probability of developing the composite outcome.
 Mandatory variables: "age", "male", "sbp", "dbp", "t2dm", "hdl", "smoking", "ht_medication".
 These variables must be applied with the above spelling.

 The details of this prediction model should be referred to the original article.
 "Absolute risk score for stroke, myocardial infarction, and all cardiovascular disease: Japan Arteriosclerosis Longitudinal Study"

 DOI:10.1038/s41440-019-0220-z

 PMID:30760890
}
\value{
Based on the mandatory variables, a risk probability will be calculated for each line.
The primary cautions are as follows,

1."Data frame must contain the following columns.":

If you contact this warning message, the data frame did not contain required variables. Please check the data  frame contain the mandatory variables in the designated spelling.

2."Data frame contains NA values in required columns.":

jals_composite_woECG() returns the NA if the required variables were missing, or NAs were generated in the calculation process.

3."Some individuals are aged 90 or older. results may not be accurate.":

4."Some individuals are aged 40 or younger. results may not be accurate.":
}


\examples{

data <- data.frame(
  hdl = c(30, 50, 70, 40, 59, 60),
  sbp = c(110, 125, 135, 150, 170, 180),
  dbp = c(75, 75, 85, 95, 105, 110),
  ht_medication = c(0, 1, 0, 1, 0, 1),
  age = c(39, 49, 59, 69, 79, 80),
  t2dm = c(0, 0, 1, 1, 1, 0),
  male = c(1, 1, 0, 0, 0, 1),
  smoking = c(1, 0, 1, 0, 0, 1)
)

# hdl           :high-density lipoprotein cholesterol(continuous)
# sbp           :systolic blood pressure(continuous)
# dbp           :diastolic blood pressure(continuous)
# ht_medication :hypertensive medication user(category:0 = non-user,1 = user)
# age           :age(continuous)
# t2dm          :diabetes mellitus(category:0 = non-prevalent DM,1 = prevalent DM)
# male          :male(category:0 = female,1 = male)
# smoking       :smoking status(category:0 = never/past smoker,1 = current smoker)


jals_composite_woECG(data)
}
