#' Calculating WHO/ISH Risk Scores
#'
#' This function calculates WHO/ISH risk scores for any epidemiolgical subregion of the world.
#'
#' @param age age (years)
#' @param gdr gender (1 = male, 0 = female)
#' @param smk smoking status (1 = current smoker, 0 = non-smoker)
#' @param sbp systolic blood pressure (mmHg)
#' @param dm diabetes status (1 = diabetic, 0 = non-diabetic)
#' @param chl total cholesterol (mmol/L)
#' @param subregion epidemiolgical subregion (character value, one of: “AFR_D”, “AFR_E”, “AMR_A”, “AMR_B”, “AMR_D”,“EMR_B”,“EMR_D”, “EUR_A”,“EUR_B”, EUR_C”, “SEAR_B”, “SEAR_D”, “WPR_A”, “WPR_B”)
#' @return A vector of WHO/ISH risk scores.
#' @author Dylan R.J. Collins, Joseph Lee, Niklas Bobrovtiz, Constantinos Koshiaris, Alison Ward, Carl Heneghan
#' @details
#' This function calculates WHO/ISH risk scores for any epidemiological subregion based on the values of six paramters passed to it (age, gender, smoking status, systolic blood pressure, diabetes status, and total cholesterol).
#' @export

WHO_ISH_Risk<- function (age,gdr,smk,sbp,dm,chl, subregion) {

  #Load required data
  df<-data.frame(age,gdr,smk,sbp,dm,chl)
  ref<-read.csv(file=system.file("extdata", "WHO_ISH_Scores.csv", package = "whoishRisk"))

  #Warning Messages
  if(any(df$age < 19)) warning("At least one age is 18 or younger")
  if(any(df$age > 100)) warning("At least one age is greater than 100")
  if(any(df$gdr > 1)) warning("Gender must be equal to 0 or 1")
  if(any(df$smk > 1)) warning("Smoking must be equal to 0 or 1")
  if(any(df$sbp < 90)) warning("At least one systolic blood pressure is below 90 mmHg")
  if(any(df$sbp > 250)) warning("At least one systolic blood pressure is over 250 mmHg")
  if(any(df$dm > 1)) warning("Diabetes status must be equal to 0 or 1")
  if(any(df$chl >10)) warning("At least one total cholesterol is greater than 10 mmol/L. Ensure all values are in units of mmol/L")


  #Age
  df$age<-ifelse(df$age > 17 & df$age < 50, 40, df$age)
  df$age<-ifelse(df$age >= 50 & df$age < 60, 50, df$age)
  df$age<-ifelse(df$age >= 60 & df$age < 70, 60, df$age)
  df$age<-ifelse(df$age >= 70, 70, df$age)

  #Cholesterol
  df$chl<-ifelse(df$chl > 0 & df$chl < 4.5, 4, df$chl)
  df$chl<-ifelse(df$chl >= 4.5 & df$chl < 5.5, 5, df$chl)
  df$chl<-ifelse(df$chl >= 5.5 & df$chl < 6.5, 6, df$chl)
  df$chl<-ifelse(df$chl >= 6.5 & df$chl < 7.5, 7, df$chl)
  df$chl<-ifelse(df$chl >= 7.5, 8, df$chl)

  #Systolic Blood Pressure
  df$sbp<-ifelse(df$sbp > 0 & df$sbp < 140, 120, df$sbp)
  df$sbp<-ifelse(df$sbp >= 140 & df$sbp < 160, 140, df$sbp)
  df$sbp<-ifelse(df$sbp >= 160 & df$sbp < 180, 160, df$sbp)
  df$sbp<-ifelse(df$sbp >= 180, 180, df$sbp)


  #Create a new variable called luv (look up value)
  df$luv<-paste(df$age,df$gdr,df$dm,df$smk,df$sbp,df$chl,sep="")

  #Create a new variable called refv (reference value)
  ref$refv<-paste(ref$age,ref$gdr,ref$dm,ref$smk,ref$sbp,ref$chl,sep="")

  #Match the look up value with the reference value
  if (subregion=="AFR_D"){
    df$AFR_D<-ref$AFR_D[match(df$luv,ref$refv)]
    return(as.character(df$AFR_D))}

  if (subregion=="AFR_E"){
    df$AFR_E<-ref$AFR_E[match(df$luv,ref$refv)]
    return(as.character(df$AFR_E))}

  if (subregion=="AMR_A"){
    df$AMR_A<-ref$AMR_A[match(df$luv,ref$refv)]
    return(as.character(df$AMR_A))}

  if (subregion=="AMR_B"){
    df$AMR_B<-ref$AMR_B[match(df$luv,ref$refv)]
    return(as.character(df$AMR_B))}

  if (subregion=="AMR_D"){
    df$AMR_D<-ref$AMR_D[match(df$luv,ref$refv)]
    return(as.character(df$AMR_D))}

  if (subregion=="EMR_B"){
    df$EMR_B<-ref$EMR_B[match(df$luv,ref$refv)]
    return(as.character(df$EMR_B))}

  if (subregion=="EMR_D"){
    df$EMR_D<-ref$EMR_D[match(df$luv,ref$refv)]
    return(as.character(df$EMR_D))}

  if (subregion=="EUR_A"){
    df$EUR_A<-ref$EUR_A[match(df$luv,ref$refv)]
    return(as.character(df$EUR_A))}

  if (subregion=="EUR_B"){
    df$EUR_B<-ref$EUR_B[match(df$luv,ref$refv)]
    return(as.character(df$EUR_B))}

  if (subregion=="EUR_C"){
    df$EUR_C<-ref$EUR_C[match(df$luv,ref$refv)]
    return(as.character(df$EUR_C))}

  if (subregion=="SEAR_B"){
    df$SEAR_B<-ref$SEAR_B[match(df$luv,ref$refv)]
    return(as.character(df$SEAR_B))}

  if (subregion=="SEAR_D"){
    df$SEAR_D<-ref$SEAR_D[match(df$luv,ref$refv)]
    return(as.character(df$SEAR_D))}

  if (subregion=="WPR_A"){
    df$WPR_A<-ref$WPR_A[match(df$luv,ref$refv)]
    return(as.character(df$WPR_A))}

  if (subregion=="WPR_B"){
    df$WPR_B<-ref$WPR_B[match(df$luv,ref$refv)]
    return(as.character(df$WPR_B))}

}
