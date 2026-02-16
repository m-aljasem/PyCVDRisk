\name{hisayama}
\alias{hisayama}
\title{10-year risk calculation for ASCVD events based on the Hisayama study}
\usage{
hisayama(data)
}
\arguments{
    \item{data}{A data frame containing the variables to estimate risk.}
}

\description{
 hisayama() calculates the 10-year probability of developing the first atherosclerotic cardiovascular disease (ASCVD) event.
 Mandatory variables: "age", "male", "sbp", "t2dm", "hdl", "ldl", "urineprotein", "smoking", "exercise".
 These variables must be applied with the above spelling.

 The details of this prediction model should be referred to the original article.
 "Development and Validation of a Risk Prediction Model for Atherosclerotic Cardiovascular Disease in Japanese Adults: The Hisayama Study"

 DOI:10.5551/jat.61960

 PMCID:PMC8894117

 PMID:33487620
}


\value{
Based on the mandatory variables, a risk probability will be calculated for each line.
The primary cautions are as follows,

1."Data frame must contain the following columns.":

If you contact this warning message, the data frame did not contain required variables. Please check the data  frame contain the mandatory variables in the designated spelling.

2."Data frame contains NA values in required columns.":

hisayama() returns the NA if the required variables were missing, or NAs were generated in the calculation process.

3."Some individuals are aged 80 or older. results may not be accurate.":

4."Some individuals are aged 40 or younger. results may not be accurate.":

This risk prediction model was developed based on the Hisayama study participants(older than 40 and younger than 80). In the calculation process, when the participants' age is outside of the range, the  result is returned as follows.

1. Younger than 30: same value for 30-40.

2. Older than 80: same value for 75-79.
}


\examples{
data <- data.frame(
  age = c(45, 55, 90),
  male = c(1, 0, 1),
  sbp = c(120, 130, 145),
  t2dm = c(0, 1, 0),
  hdl = c(55, 45, 35),
  ldl = c(110, 130, 160),
  urineprotein = c(NA, 1, 0),
  smoking = c(0, 1, 0),
  exercise = c(1, 0, 1)
)

# age          :age(continuous)
# male         :male(category:0 = female,1 = male)
# sbp          :systolic blood pressure(continuous)
# t2dm         :diabetes mellitus(category:0 = non-prevalent DM,1 = prevalent DM)
# hdl          :high-density lipoprotein cholesterol(continuous)
# ldl          :low-density lipoprotein cholesterol(continuous)
# urineprotein :proteinuria(category:0 = negative,1 = positive)
# smoking      :smoking status(category:0 = never/past smoker,1 = current smoker)
# exercise     :weekly exercise(category:0 = less than 3 times,1 = 3 times or more)


hisayama(data)

}
