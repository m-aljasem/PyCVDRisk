\name{epoch_cvd}
\alias{epoch_cvd}
\title{10-year risk of death from CVD from the EPOCH Japan study participants}
\usage{
epoch_cvd(data)
}
\arguments{
    \item{data}{A data frame containing the variables to estimate risk.}
}

\description{
 epoch_cvd() calculates the absolute 10-year risk of death from cardiovascular disease (CVD).

 Mandatory variables: "age", "male", "urineprotein", "sbp", "t2dm", "smoking".
 These variables must be applied with the above spelling.

 The details of this prediction model should be referred to the original article.
 "Estimation of 10-Year Risk of Death from Coronary Heart Disease, Stroke, and Cardiovascular Disease in a Pooled Analysis of Japanese Cohorts: EPOCH-JAPAN"

 DOI:10.5551/jat.58958

 PMID:33041313

 PMCID:PMC8326173
}
\value{
Based on the mandatory variables, a risk probability will be calculated for each line.
The primary cautions are as follows,

1."Data frame must contain the following columns.":

If you contact this warning message, the data frame did not contain required variables. Please check the data frame contain the mandatory variables in the designated spelling.

2."Data frame contains NA values in required columns.":

epoch_cvd() returns the NA if the required variables were missing, or NAs were generated in the calculation process.

3."Some individuals are aged 80 or older. results may not be accurate.":

4."Some individuals are aged 40 or younger. results may not be accurate.":
}

\examples{
data <- data.frame(
  age = c(39, 85, 70),
  male = c(1, 0, 1),
  sbp = c(120, 130, 140),
  t2dm = c(NA, 1, 0),
  urineprotein = c(1, 1, 0),
  smoking = c(0, 1, 1)
)

# age          :age(continuous)
# male         :male(category:0 = female, 1 = male)
# sbp          :systolic blood pressure(continuous)
# t2dm         :diabetes mellitus(category: 0 = non-prevalent DM, 1 = prevalent DM)
# urineprotein :proteinuria(category: 0 = negative, 1 = positive)
# smoking      :smoking status(category: 0 = never/past smoker, 1 = current smoker)


epoch_cvd(data)
}
