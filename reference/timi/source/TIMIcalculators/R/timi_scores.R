# Original risk function used for every score based on cox model
# This way of calculating requires the original data
.risk <- function(model, newdata, time) {
  as.numeric(1-summary(survival::survfit(model, newdata = newdata), times = time)$surv)
}
# COXPH Predictions made off the model, no data needed but model=TRUE
.predictRisk <- function(model, newdata, time) {
  timeVar <- as.character(model$formula[[2]])[2]
  statusVar <- as.character(model$formula[[2]])[3]
  newdata[ , timeVar] <- time
  newdata[ , statusVar] <- 1
  # The underlying function is under :::, need to use the generic
  # 1 - survival:::predict.coxph(object = model , newdata = newdata , type = "survival" , se.fit=FALSE)
  1 - predict(object = model , newdata = newdata , type = "survival" , se.fit=FALSE)
}
# CPH Predictions that really needs no data
.cphPredictRisk <- function(model, newdata, time) {
  1-rms::survest(model, newdata,
                 times=time , conf.int=FALSE)$surv
}

# Standard check for data.frame input
.checkDF <- function(df , type){
    # Check if dataframe
    if(!is.data.frame(df)){
      stop("df is not a dataframe")
    }
    samplefile <- timi_examples[[type]]
    colsIneed <- colnames(samplefile)
    colsIneedChar <- colnames(samplefile)[ sapply(samplefile , class) == "character" ]
    # Check for missing variables
    if(!all(colsIneed %in% colnames(df))){
      mex <- paste0("Missing Variables:" , paste(setdiff(colsIneed , colnames(df)) , collapse = ", "))
      stop(mex)
    }
    # Check that each columns is the same type
    mex2 <- NULL
    for( i in colsIneed){
      if(class(samplefile[,i])!=class(df[ , i])){
        mex2 <- paste(mex2 , paste( paste0("Column " , i , " is not the right class.")
                                    , paste0("User provided:" , class(df[ , i]))
                                    , paste0("Required:" , class(samplefile[ , i]))
                                    , sep ="\n") , sep = "\n")
      }
    }
    if(!is.null(mex2)){
      stop(mex2)
    }
    # Check that each character column contains only recognized fields
    mex3 <- NULL
    for( i in colsIneedChar){
      elementDiff <- setdiff( unique(df[ , i]) , samplefile[ , i])
      if(length(elementDiff)>0){
        mex3 <- paste(mex3
                      , paste0("Column " , i , "contains unrecognized elements:"
                               , paste(elementDiff , collapse = ", ")) , sep = "\n")
      }
    }
    if(!is.null(mex3)){
      stop(mex3)
    }
}

#' TIMI score for SEE
#'
#' Description to be made
#'
#' @param df data.frame in the same format as timi_examples$trsmace. See details
#' @param BL_CRCL Creatinine clearance (Cockroft Gault formula; mL/min)
#' @param afib_var Paroxysmal (Non-paroxysmal reference)
#' @param vka_naive VKA naïve (VKA experience reference)
#' @param SEX Female (Male reference)
#' @param hxstroketia History of Stroke/TIA (no history reference)
#' @param hxmi History of MI (no history reference)
#' @param hxpad History of PAD (no history reference)
#' @param CURRSMOKE Current smoker (non-current as reference)
#' @examples
#' # One patient
#' computeSEE(BL_CRCL = 54,afib_var=1, vka_naive=1
#' , SEX = 0, hxstroketia = 0
#' , hxmi = 1 , hxpad = 0 , CURRSMOKE = 1 )
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_see <- computeSEE(timi_examples$see)
#' @return if the arguments have length 1, returns the score alone,
#'   otherwise returns the original dataframe with "see_score" as extra column
#' @importFrom survival Surv coxph
#' @importFrom stats predict model.frame
#' @importFrom plyr mapvalues
#' @importFrom rms cph survest
#' @export
computeSEE <- function(df=NULL , BL_CRCL,
                     afib_var, vka_naive, SEX, hxstroketia,
                     hxmi, hxpad,CURRSMOKE){

  dfmode <- FALSE
  # DATA FRAME MODE
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "see")
    dfOriginal <- df
  } else {
    df<-data.frame(
      BL_CRCL = BL_CRCL
      ,afib_var = afib_var
      ,vka_naive = vka_naive
      ,SEX = SEX
      ,hxstroketia = hxstroketia
      ,hxmi = hxmi
      ,hxpad = hxpad
      ,CURRSMOKE = CURRSMOKE
      , stringsAsFactors = FALSE)
  }
  df$prediction <- .cphPredictRisk(modelSEE,newdata=df,time=974)
  if(!dfmode){
    return(paste0(round(df$prediction*100,4), "%"))
  } else {
    return(cbind(dfOriginal , see_score=df$prediction))
  }
}

#' ApoB prediction
#'
#' This calculator can predict ApoB values from a standard lipid panel
#'
#' @param df data.frame in the same format as timi_examples$trsmace. See details
#' @param age Age in years
#' @param sex 0 for Female, 1 for Male
#' @param hdl HDL-C in mg/dL
#' @param totchol Total Cholesterol in mg/dL
#' @param tg Triglycerides in mg/dL
#' @param lpa Lp(a) in mg/dL
#' @param method "linear" for linear regression or "xgboost" for ML prediction. default xgboost
#' @examples
#' # One patient
#' computeAPOB(age = 60 , sex = 1 , hdl = 50 , totchol = 180
#'   , tg = 100 , lpa = 20 , method = "linear")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_apob <- computeAPOB(timi_examples$apob )
#' @return if the arguments have length 1, returns the score alone,
#'   otherwise returns the original dataframe with "apob_pred" as extra column
#' @importFrom stats predict
#' @import xgboost
#' @export
computeAPOB <- function(df=NULL , age , sex , hdl , totchol , tg , lpa , method = "xgboost"){

  dfmode <- FALSE
  # DATA FRAME MODE
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "apob")
    dfOriginal <- df
    df$lpa <- log(df$lpa+0.01)
    df$tg <- log(df$tg+0.01)

  } else {
    df<-data.frame(
      age = age
      ,sex = sex
      ,hdl = hdl
      ,totchol = totchol
      ,tg = log(tg+0.01)
      ,lpa= log(lpa+0.01)
      , stringsAsFactors = FALSE)
  }

  if(method == "xgboost"){
    df$prediction <- predict(modelAPOBxgboost , data.matrix(df))
  }
  if( method == "linear"){
    df$prediction <- predict(modelAPOBlinear , df)
  }

  if(!dfmode){
    return(paste0(round(df$prediction,1)))
  } else {
    return(cbind(dfOriginal , apob_pred=df$prediction))
  }
}

#' TIMI score for Kidney Disease Progression
#'
#' This score estimates the 2-year risk of kidney disease progression (KDP)
#' in patients with type 2 diabetes (T2D).
#' This score was developed as a clinical risk score for the composite
#' kidney endpoint of sustained >=40% eGFR reduction, end-stage
#' kidney disease (ESKD), or kidney death, in a pooled cohort of 41,195 patients
#' with T2DM from 4 TIMI clinical trial cohorts.
#'
#' @param df data.frame in the same format as timi_examples$trsmace. See details
#' @param hxchf History of Congestive Heart Failure. Values: "Yes", "No"
#' @param ascvd History of Atherosclerotic cardiovascular disease. Values: "Yes", "No"
#' @param dmdur_cat Diabetes duration (yrs). Values: "<5", "5-10", ">10"
#' @param sbp_cat Systolic Blood Pressure. Values: "<140", "140-160", ">=160"
#' @param egfrcat Estimated glomerular filtration rate (ml/min/1.73 m^2). Values: "<45", "[45-60)", ">=60"
#' @param uacr Urine albumin-to-creatinine ratio. Continuous value: mg/g or mg/nmol
#' @param hemoglobln Hemoglobin in g/dL.
#' @param hba1c Hemoglobin A1c. Continuous value: % or mmol/mol
#' @param uacr_unit Either "mgg" for mg/g or "mgnmol" for mg/nmol
#' @param hba1c_unit Either "perc" for % or "mmol" for mmol/mol
#' @examples
#' # One patient
#' computeKDP(hxchf = "No",ascvd="Yes", dmdur_cat="<5"
#' , sbp_cat = ">=160", egfrcat = "[45-60)"
#' , uacr = 6 , hemoglobln = 10 , hba1c = 7 ,uacr_unit = "mgg" , hba1c_unit = "perc")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_kdp <- computeKDP(timi_examples$kdp ,uacr_unit = "mgg" , hba1c_unit = "perc")
#' @return if the arguments have length 1, returns the score alone,
#'   otherwise returns the original dataframe with "kdp_score" as extra column
#' @importFrom survival Surv coxph
#' @importFrom stats predict model.frame
#' @importFrom plyr mapvalues
#' @importFrom rms cph survest
#' @export
computeKDP<-function(df=NULL , hxchf,
                     ascvd, dmdur_cat, sbp_cat, egfrcat,
                     uacr, hemoglobln,hba1c
                     ,uacr_unit=c("mgg","mgnmol")
                     ,hba1c_unit=c("perc" , "mmol")){

  uacr_unit <- uacr_unit[1]
  uacr_unit <- c(mgg=1,mgnmol=8.849558)[uacr_unit]

  hba1c_unit <- hba1c_unit[1]


  dfmode <- FALSE
  # DATA FRAME MODE
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "kdp")
    dfOriginal <- df
    df$hba1c <- if(hba1c_unit=="perc") df$hba1c else (df$hba1c + 23.5)/10.93

  } else {
    hba1c <- ifelse(hba1c_unit=="perc", hba1c , (hba1c + 23.5)/10.93)
    df<-data.frame(
      hxchf = hxchf
      ,ascvd = ascvd
      ,dmdur_cat = dmdur_cat
      ,sbp_cat = sbp_cat
      ,egfrcat = egfrcat
      ,uacr = uacr*uacr_unit
      ,hba1c = hba1c
      ,hemoglobln = hemoglobln
      , stringsAsFactors = FALSE)
  }

  # Check categorical variables
  testLevels <- list(hxchf = c("No" , "Yes")
                     , ascvd = c("No" , "Yes")
                     , dmdur_cat = c("<5" , "5-10" , ">10")
                     , sbp_cat = c("<140" , "140-160" , ">=160" )
                     , egfrcat = c(">=60" , "<45" , "[45-60)" ))
  for(i in names(testLevels)){
    # we do accept NAs but no value is calculated
    # if(!all(c(df[ ,i] %in% c(testLevels[[i]],NA)))){
    #   unrecogn <- paste( setdiff(unique(df[ ,i]) , testLevels[[i]]) , collapse = ", ")
    #   stop(paste0("Variable " , i , " contains unrecognized values: " , unrecogn))
    # } else {
      # df[ , i] <- as.numeric(plyr::mapvalues( df[ , i]
      #                                  , from = testLevels[[i]]
      #                                  , to = seq(0 , length(testLevels[[i]])-1)
      #                                  , warn_missing = FALSE))
      df[ , i] <- factor(df[ , i], levels = testLevels[[i]])

    # }
  }
  levels(df$ascvd) <- c("0" , "1")
  # df$prediction<-1-pec::predictSurvProb(modelKDP,newdata=df,times=730)
  # df$prediction <- .risk(modelKDP,newdata=df,time=730)
  # df$prediction2 <- .predictRisk(modelKDP2,newdata=df,time=730)
  df$prediction <- .cphPredictRisk(modelKDP,newdata=df,time=730)
  if(!dfmode){
    return(paste0(round(df$prediction*100,4), "%"))
  } else {
    return(cbind(dfOriginal , kdp_score=df$prediction))
  }
}

#' TIMI Risk Score for Atherothrombosis in Diabetes
#'
#' This score estimates the 3-year risk of atherothrombotic events in patients with diabetes.
#' This score was developed as a clinical risk score for fatal
#' or non-fatal myocardial infarction (MI) or ischemic stroke (IS)
#' in a pooled cohort of 42,181 patients with T2DM from 4 TIMI clinical trial cohorts.
#' Candidate variables were assessed with multivariable Cox regression,
#' and independent variables (p<0.05) retained in the final model.
#' Sixteen variables were independent predictors of MI or IS
#'
#' @param df data.frame in the same format as timi_examples$trsmace. See details
#' @param BLinsulinmed Insulin use. Values: "Yes", "No"
#' @param Male Sex Male. Values: "Male" , "Female"
#' @param SBP_gr Systolic Blood Pressure Group. Values: "<140", "140-159", ">=160"
#' @param UACR_gr Urine albumin-to-creatinine ratio. Values in mg/g:
#' "<30","30-300",">300" or mg/nmol "<3.39","3.39-33.9",">33.9"
#' @param age_gr Age groups: "<65", "65-74", ">=75"
#' @param curr_smoke Current smoker. Values: "Yes", "No"
#' @param eGFR60 Estimated glomerular filtration rate (ml/min/1.73 m^2). Values ">=60","<60"
#' @param hxcabg History of CABG. Values: "Yes", "No"
#' @param hxcad History of CAD Values: "Yes", "No"
#' @param hxistrk History of Ischemic Stroke. Values: "Yes", "No"
#' @param hxmi History of MI. Values: "Yes", "No"
#' @param hxpad History of PAD. Values: "Yes", "No"
#' @param hxpci History of PCI. Values: "Yes", "No"
#' @param BLHBA1C Hba1C. Continuos value in % or mmol/mol
#' @param BLwaist Waist circumference in cm or inches or BMI.
#' @param BLldlc LDL-cholesterol in mg/dL or mmol/L
#' @param BLHBA1C_unit "perc" for % or "mmol" for mmol/mol. Default "perc"
#' @param BLwaist_unit "cm" for centimeter or "inches" for inches.
#' Use "bmi" for BMI based model. Default "cm"
#' @param BLldlc_unit "mgdl" for mg/dL or "mmol" for mmol/L. Default "mgdl"
#' @param time 3 or 10. Projections at 3-year, default, or 10-year extrapolation
#' @examples
#' # One patient
#' computeTRSMACE(BLinsulinmed = "Yes",
#'    Male="Female", SBP_gr="140-159", UACR_gr="<30", age_gr=">=75",
#'    curr_smoke="No", eGFR60="<60",hxcabg="No",
#'    hxcad="No",hxistrk="Yes",hxmi="No",hxpad="Yes",hxpci="No",
#'    BLHBA1C=7,BLwaist=90,BLldlc=2.5
#'    ,BLHBA1C_unit="perc",BLwaist_unit="cm",BLldlc_unit="mmol",time=3)
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_mace <- computeTRSMACE(df = timi_examples$trsmace,
#'    BLHBA1C_unit="perc",BLwaist_unit="cm",BLldlc_unit="mgdl")
#' @return
#' if df is not used, returns the score alone,
#'   otherwise returns the original
#'   dataframe with "trsmace_score" as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$trsmace.
#' See data(timi_examples) to see all example files.
#'
#' @importFrom survival Surv coxph
#' @importFrom stats predict model.frame
#' @importFrom plyr mapvalues
#' @importFrom rms cph survest
#' @importFrom fastDummies dummy_cols
#' @export
computeTRSMACE<-function(df = NULL , BLinsulinmed,
                         Male, SBP_gr, UACR_gr, age_gr,
                         curr_smoke, eGFR60,hxcabg,
                         hxcad,hxistrk,hxmi,hxpad,hxpci,
                         BLHBA1C,BLwaist,BLldlc
                         ,BLHBA1C_unit=c("perc","mmol")
                         ,BLwaist_unit=c("cm","inches","bmi")
                         ,BLldlc_unit=c("mgdl","mmol")
                         ,time=c(3,10)){
  time <- time[1]
  BLHBA1C_unit <- BLHBA1C_unit[1]
  BLwaist_unit <- BLwaist_unit[1]
  BLldlc_unit <- BLldlc_unit[1]
  # Retrieve conversion factors
  BLwaist_unit2 <- c(cm=1,inches=2.54,bmi=1)[BLwaist_unit]
  BLldlc_unit2 <- c(mgdl=1,mmol=38.66976)[BLldlc_unit]

  dfmode <- FALSE
  # DATA FRAME MODE
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "trsmace")
    dfOriginal <- df
    df$BLHBA1C <- if(BLHBA1C_unit=="perc") df$BLHBA1C else (df$BLHBA1C + 23.5)/10.93
    df$BLwaist <- df$BLwaist*BLwaist_unit2
    df$BLldlc <- df$BLldlc*BLldlc_unit2

  } else {
    BLHBA1C <- if(BLHBA1C_unit=="perc") BLHBA1C else (BLHBA1C + 23.5)/10.93
    df <- data.frame(
      BLinsulinmed=BLinsulinmed
      ,Male=Male
      ,SBP_gr=SBP_gr
      ,UACR_gr=UACR_gr
      ,age_gr=age_gr
      ,curr_smoke=curr_smoke
      ,eGFR60=eGFR60
      ,hxcabg=hxcabg
      ,hxcad=hxcad
      ,hxistrk=hxistrk
      ,hxmi=hxmi
      ,hxpad=hxpad
      ,hxpci=hxpci
      ,BLHBA1C=BLHBA1C
      ,BLwaist=BLwaist*BLwaist_unit2
      ,BLldlc=BLldlc*BLldlc_unit2)

  }
  # Use BMI version
  if(BLwaist_unit=="bmi"){
    colnames(df)[colnames(df)=="BLwaist"] <- "BLBMI"
  }

  df$BLinsulinmed <- plyr::mapvalues(df$BLinsulinmed , from = c("No","Yes") , to = c(0,1) , warn_missing = FALSE)
  df$Male <- plyr::mapvalues(df$Male, from = c("Female" , "Male") , to = c(0,1), warn_missing = FALSE)
  df$SBP_gr <- plyr::mapvalues(df$SBP_gr, from = c("<140" , "140-159" , ">=160") , to = c(0,1,2), warn_missing = FALSE)

  df$UACR_gr <- ifelse( df$UACR_gr %in% c("<30","<3.39") , 0
                            , ifelse( df$UACR_gr %in% c("30-300","3.39-33.9") , 1
                                     , ifelse( df$UACR_gr %in% c(">=300",">=33.9") , 2 , NA)))
  df$age_gr <- plyr::mapvalues(df$age_gr, from = c("<65" , "65-74" , ">=75") , to = c(0,1,2), warn_missing = FALSE)
  df$curr_smoke <- plyr::mapvalues(df$curr_smoke,  from = c("No","Yes") , to = c(0,1), warn_missing = FALSE)
  df$eGFR60 <- plyr::mapvalues(df$eGFR60,  from = c(">=60","<60") , to = c(0,1), warn_missing = FALSE)
  df$hxcabg <- plyr::mapvalues(df$hxcabg,  from = c("No","Yes") , to = c(0,1), warn_missing = FALSE)
  df$hxcad <- plyr::mapvalues(df$hxcad, from = c("No","Yes") , to = c(0,1), warn_missing = FALSE)
  df$hxistrk <- plyr::mapvalues(df$hxistrk, from = c("No","Yes") , to = c(0,1), warn_missing = FALSE)
  df$hxmi <- plyr::mapvalues(df$hxmi, from = c("No","Yes") , to = c(0,1), warn_missing = FALSE)
  df$hxpad <- plyr::mapvalues(df$hxpad, from = c("No","Yes") , to = c(0,1), warn_missing = FALSE)
  df$hxpci <- plyr::mapvalues(df$hxpci, from = c("No","Yes") , to = c(0,1), warn_missing = FALSE)
  # Generate factors in case the DF doesn't contain all possible values
  notToFactor <- c("BLHBA1C" , "BLwaist" , "BLldlc" , "BLBMI",'SBP_gr',"UACR_gr","age_gr")
  for(i in setdiff(colnames(df) , notToFactor)){
    df[ , i] <- factor(df[ , i] , levels = c(0,1))
  }
  # Factors with 3 groups
  for(i in c('SBP_gr',"UACR_gr","age_gr")){
    df[ ,i] <- factor(df[ , i] , levels = c(0,1,2))
  }
  # Create dummies for 3 level factors
  df <- fastDummies::dummy_cols(df, select_columns = c('SBP_gr',"UACR_gr","age_gr") )
  # Remove unused columns
  for(i in c('SBP_gr',"UACR_gr","age_gr")){
    df[ ,i] <- NULL
  }
  # Re-factorize dummied variables (0,1 factors)
  for( i in grep('SBP_gr|UACR_gr|age_gr' , colnames(df) , value=TRUE)){
    df[ , i] <- factor(df[ , i] , levels = c(0,1))
  }
  if(BLwaist_unit=="bmi"){
    if(time==3){
      df$prediction <- .cphPredictRisk(modelTRSMACEbmi,newdata=df,time=1095.75)
      # df$prediction2 <- .predictRisk(modelTRSMACEbmi2,newdata=df,time=1095.75)
    } else {
      df$prediction <- 1-(1-(.cphPredictRisk(modelTRSMACEbmi,newdata=df,time=1095.75)))^(10/3)
      # df$prediction2 <- 1-(1-(.predictRisk(modelTRSMACEbmi2,newdata=df,time=1095.75)))^(10/3)
    }

  }else{
    if(time==3){
      df$prediction <- .cphPredictRisk(modelTRSMACE,newdata=df,time=1095.75)
      # df$prediction2 <- .predictRisk(modelTRSMACE2,newdata=df,time=1095.75)
    } else {
      df$prediction <- 1-(1-(.cphPredictRisk(modelTRSMACE,newdata=df,time=1095.75)))^(10/3)
      # df$prediction2 <- 1-(1-(.predictRisk(modelTRSMACE2,newdata=df,time=1095.75)))^(10/3)
    }
  }
  if(!dfmode){
    return(paste0(round(df$prediction[1]*100,4), "%"))
  } else {
    return(cbind(dfOriginal , trsmace_score=df$prediction))
  }

}


#' TIMI Risk Score for New Onset Diabetes (NODM)
#'
#' This score estimates the 2.5-year risk of New Onset Diabetes
#'
#' @param df data.frame in the same format as timi_examples$trsmace. See details
#' @param age Age in years
#' @param BLhba1c HbA1C value as percentage or mmol/mol
#' @param tghdlratio Tryglicerides to HDL ratio
#' @param bmi Body Mass Index
#' @param glucose Glucose value in mg/dL
#' @param BLhba1c_unit "perc" for % or "mmol" for mmol/mol. Default "perc"
#' @param splines Use spline model estimates? Default FALSE
#' @examples
#' # One patient
#' computeTRSNODM(age = 50, BLhba1c = 7 , tghdlratio = 4 , bmi = 30
#'    , glucose=50 ,BLhba1c_unit="perc")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_nodm <- computeTRSNODM(df = timi_examples$trsnodm,
#'    BLhba1c_unit="perc", splines = FALSE)
#' # Add splines estimates
#' timi_example_nodm <- computeTRSNODM(df = timi_example_nodm,
#'    BLhba1c_unit="perc", splines = TRUE)
#' @return if df is not used, returns the score alone,
#'   otherwise returns the original
#'   dataframe with "trsnodm_score" as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$trsmace.
#' See data(timi_examples) to see all example files.
#' @references Filipe A. Moura et al,
#' [Risk of new-onset diabetes and efficacy of pharmacological weight loss
#' therapy](https://dom-pubs.onlinelibrary.wiley.com/doi/abs/10.1111/dom.15798)
#' Diabetes, Obesity and Metabolism, 26 July 2024 https://doi.org/10.1111/dom.15798
#' @importFrom survival Surv coxph
#' @importFrom stats predict model.frame
#' @importFrom plyr mapvalues
#' @importFrom rms rcs cph survest
#' @export
computeTRSNODM<-function(df = NULL , age ,
                         BLhba1c , tghdlratio , bmi, glucose
                         ,BLhba1c_unit=c("perc" , "mmol"),splines=FALSE){
  BLhba1c_unit <- BLhba1c_unit[1]
  dfmode <- FALSE
  # DATA FRAME MODE
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "trsnodm")
    dfOriginal <- df
    df$BLhba1c<- if(BLhba1c_unit=="perc") df$BLhba1c else (df$BLhba1c + 23.5)/10.93
  } else {
    BLhba1c <- if(BLhba1c_unit=="perc") BLhba1c else (BLhba1c + 23.5)/10.93
    df <- data.frame(
      age=age
      ,BLhba1c=BLhba1c
      ,tghdlratio=tghdlratio
      ,bmi=bmi
      ,glucose=glucose)
  }

  df$bmicat<-1
  df$bmicat[df$bmi>=30] <- 2 # fix

  if(splines){
    # df$prediction<-1-predictSurvProb(modelTRSNODMsp,newdata=df,times=900)
    # df$prediction <- .risk(modelTRSNODMsp,newdata=df,time=900)
    df$prediction <- .cphPredictRisk(modelTRSNODMsp,newdata=df,time=900)
    # df$prediction2 <- .predictRisk(modelTRSNODMsp2,newdata=df,time=900)
  }else{
    # df$prediction<-1-predictSurvProb(modelTRSNODM,newdata=df,times=900)
    # df$prediction <- .risk(modelTRSNODM,newdata=df,time=900)
    df$prediction <- .cphPredictRisk(modelTRSNODM,newdata=df,time=900)
    # df$prediction2 <- .predictRisk(modelTRSNODM2,newdata=df,time=900)
  }
  if(!dfmode){
    return(paste0(round(df$prediction*100,4), "%"))
  } else {
    return(cbind(dfOriginal , trsnodm_score=df$prediction))
  }

}

#' TIMI Risk Score for UA/NSTEMI
#'
#' This score estimates the 14-day risk of all-cause mortality,
#' new or recurrent myocardial infarction, or severe recurrent ischemia
#' requiring urgent revascularization in patients with unstable angina or NSTEMI.
#'
#' @param df data.frame in the same format as timi_examples$uanstemi See details
#' @param age Either ">=65" or "<65"
#' @param cadrisk 3 or more Risk Factors for CAD, including family history of CAD,
#' hypertension, hypercholesterolemia, diabetes, or being a current smoker.
#' "Yes" or "No"
#' @param knowncad Known Coronary Artery Disease (stenosis >= 50%). "Yes" or "No"
#' @param asa Aspirin Use in Past 7d. "Yes" or "No"
#' @param angina Severe angina (>= 2 episodes w/in 24 hrs). "Yes" or "No"
#' @param stchange ST changes >= 0.5mm (0.05 mV). "Yes" or "No"
#' @param cardiac Elevated cardiac troponin or CK-MB. "Yes" or "No"
#' @examples
#' # One patient
#' computeUANSTEMI(age = "<65", cadrisk="Yes",knowncad="No"
#' ,asa="Yes",angina="No",stchange="No",cardiac="Yes")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_uanstemi <- computeUANSTEMI(df = timi_examples$uanstemi)
#' @return if df is not used, returns the score alone,
#'   otherwise returns the original
#'   dataframe with "uanstemi_Points" and "uanstemi_Risk" as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$uanstemi
#' and both points and risk in % are returned.
#' See data(timi_examples) to see all example files.
#' @references Antman EM et al. [The TIMI Risk Score for Unstable Angina/Non-ST
#' Elevation MI: A Method for Prognostication and Therapeutic
#' Decision Making](https://jamanetwork.com/journals/jama/fullarticle/192996).
#' JAMA. 2000;284:835-842.
#' @export
computeUANSTEMI <- function(df=NULL,age,cadrisk,knowncad,asa,angina,stchange,cardiac){
  consequence <- c("0 Points - < 4.7% risk"
                   ,"1 Point - 4.7% risk"
                   ,"2 Points - 8.3% risk"
                   ,"3 Points - 13.2% risk"
                   ,"4 Points - 19.9% risk"
                   ,"5 Points - 26.2% risk"
                   ,"6 Points - 40.9% risk"
                   ,"7 Points - 40.9% risk")
  myrisk <- c("<4.7%"
              ,"4.7%"
              ,"8.3%"
              ,"13.2%"
              ,"19.9%"
              ,"26.2%"
              ,"40.9%"
              ,"40.9%")
  mycols <- c("age","cadrisk","knowncad"
              ,"asa","angina","stchange"
              ,"cardiac")
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "uanstemi")
    dfOriginal <- df
    for(i in mycols){
      df[ , i] <- ifelse(df[ ,i]%in%c("Yes" , ">=65") , 1 , 0)
    }
    dfOriginal$uanstemi_Points <- rowSums(df[ , mycols])
    dfOriginal$uanstemi_Risk <- myrisk[ dfOriginal$uanstemi_Points+1 ]
    return(dfOriginal)
  } else {
    age <- ifelse(age %in% ">=65" , 1 , 0)
    cadrisk <- ifelse( cadrisk %in% "Yes" , 1 , 0)
    knowncad <- ifelse( knowncad %in% "Yes" , 1 , 0)
    asa <- ifelse( asa %in% "Yes" , 1 , 0)
    angina <- ifelse( angina %in% "Yes" , 1 , 0)
    stchange <- ifelse( stchange %in% "Yes" , 1 , 0)
    cardiac <- ifelse( cardiac %in% "Yes" , 1 , 0)
    o <- sum(age,cadrisk,knowncad,asa,angina,stchange,cardiac)
    return(consequence[o+1])
  }
}

#' TIMI Risk Score for STEMI
#'
#' This score estimates the 30-day risk of all-cause mortality in patients with
#' ST Elevation Myocardial Infarction (STEMI). A patient with an ST-Elevation
#' Myocardial Infarction (STEMI) who is post-cath may have bleeding risks and
#' the primary team would like to help objectify risk-benefit
#' of post-PCI medications.
#'
#' @param df data.frame in the same format as timi_examples$stemi See details
#' @param age Either ">=65" or "<65"
#' @param sysb Systolic BP < 100 mm/Hg. "Yes" or "No"
#' @param heartrate Heart rate > 100 bpm. "Yes" or "No"
#' @param killclass Killip Class II-IV (JVD, S3, rales, cardiogenic shock). "Yes" or "No"
#' @param stelev Anterior ST Elevation or LBBB. "Yes" or "No"
#' @param diabhypang Diabetes, Hypertension or Angina. "Yes" or "No"
#' @param weight Weight < 67kg (147.7 lbs). "Yes" or "No"
#' @param timetotrt Time to treatment > 4 hours. "Yes" or "No"
#' @examples
#' # One patient
#' computeSTEMI(age = "<65", sysb = "No"
#' , heartrate = "Yes" , killclass = "No" , stelev = "Yes"
#' , diabhypang = "No" , weight = "Yes" , timetotrt = "No")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_stemi <- computeSTEMI(df = timi_examples$stemi)
#' @return if df is not used, returns the score alone,
#'   otherwise returns the original
#'   dataframe with "stemi_Points" and "stemi_Risk" as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$stemi
#' and both points and risk in % are returned.
#' See data(timi_examples) to see all example files.
#' @references Morrow DA et al. [TIMI risk score for ST-elevation myocardial
#' infarction: A convenient, bedside, clinical score for risk assessment at
#' presentation: An intravenous nPA for treatment of infarcting myocardium early
#' II trial substudy](https://www.ahajournals.org/doi/full/10.1161/01.cir.102.17.2031).
#' Circulation. 2000;102:2031-2037.
#' @export
computeSTEMI <- function(df=NULL,age,sysb,heartrate,killclass,stelev,diabhypang,weight,timetotrt){
  consequence <- c("0 Points - 0.8% risk"
                   ,"1 Point - 1.6% risk"
                   ,"2 Points - 2.2% risk"
                   ,"3 Points - 4.4% risk"
                   ,"4 Points - 7.3% risk"
                   ,"5 Points - 12.4% risk"
                   ,"6 Points - 16.1% risk"
                   ,"7 Points - 23.4% risk"
                   ,"8 Points - 26.8% risk"
                   ,"9 Points - 35.9% risk"
                   ,"10 Points - 35.9% risk"
                   ,"11 Points - 35.9% risk"
                   ,"12 Points - 35.9% risk"
                   ,"13 Points - 35.9% risk"
                   ,"14 Points - 35.9% risk"
                   ,"15 Points - 35.9% risk"
                   ,"16 Points - 35.9% risk")
  myrisk <- c("0.8%"
              ,"1.6%"
              ,"2.2%"
              ,"4.4%"
              ,"7.3%"
              ,"12.4%"
              ,"16.1%"
              ,"23.4%"
              ,"26.8%"
              ,rep("35.9%",8))
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "stemi")
    dfOriginal <- df
    for(i in c("diabhypang","weight","stelev","timetotrt")){
      df[ , i] <- ifelse(df[ ,i]%in%"No" , 0 , 1)
    }
    for(i in c("heartrate","killclass")){
      df[ , i] <- ifelse(df[ ,i]%in%"No" , 0 , 2)
    }
    df$age <- ifelse(df$age=="65-74" , 2 , ifelse( df$age==">=75" , 3 , 0))
    df$sysb <- ifelse( df$sysb %in% "No" , 0 , 3)
    dfOriginal$stemi_Points <- rowSums(df[ , c("age","diabhypang"
                                               ,"sysb","heartrate","killclass"
                                               ,"weight","stelev","timetotrt")])
    dfOriginal$stemi_Risk <- myrisk[ dfOriginal$stemi_Points+1 ]
    return(dfOriginal)
  } else {
    age <- ifelse(age=="65-74" , 2 , ifelse( age==">=75" , 3 , 0))
    diabhypang <- ifelse( diabhypang %in% "No" , 0 , 1)
    sysb <- ifelse( sysb %in% "No" , 0 , 3)
    heartrate <- ifelse( heartrate %in% "No" , 0 , 2)
    killclass <- ifelse( killclass %in% "No" , 0 , 2)
    weight <- ifelse( weight %in% "No" , 0 , 1)
    stelev <- ifelse( stelev %in% "No" , 0 , 1)
    timetotrt <- ifelse( timetotrt %in% "No" , 0 , 1)
    o <- sum(age,diabhypang,sysb,heartrate,killclass,weight,stelev,timetotrt)
    return(consequence[o+1])
  }
}

#' TIMI Risk Score for Heart Failure in Diabetes (TRS-HF(DM))
#'
#' This score estimates the incidence rates of hospitalization
#' for heart failure in patients with diabetes.
#'
#' @param df data.frame in the same format as timi_examples$hfdiab See details
#' @param hf Prior heart failure. "Yes" or "No"
#' @param af History of atrial fibrillation. "Yes" or "No"
#' @param cad History of Coronary artery disease. "Yes" or "No"
#' @param egfr eGFR (ml/min/1.73 m^2). "<60" or ">=60"
#' @param albcrea Urine Albumine to Creatine ratio in mg/g
#' (accepted values: "<30" , "30-300" , ">=300") or
#' mg/nmol (accepted values: "<3.39" , "3.39-33.9" , ">33.9")
#' @examples
#' # One patient
#' computeHFDIAB(hf = "Yes" ,af = "No" ,cad = "No",egfr=">=60",albcrea="3-300")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_hfdiab <- computeHFDIAB(df = timi_examples$hfdiab)
#' @return if df is not used, returns the score alone in 1000 patient-years,
#'   otherwise returns the original
#'   dataframe with "hfdiab_Points" and "hfdiab_Risk_per1k_patientyear"
#'   as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$hfdiab.
#' See data(timi_examples) to see all example files.
#' @references Berg DD, et al. [Heart Failure Risk Stratification and Efficacy
#' of Sodium-Glucose Cotransporter-2 Inhibitors in Patients With Type 2
#' Diabetes Mellitus](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.119.042685?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed).
#' Circulation. 2019;140:1569-1577.
#' @export
computeHFDIAB <- function(df=NULL,hf,af,cad,egfr,albcrea){
  consequence <- c("0 Points - 2.4 (per 1000 patient-years)"
                   ,"1 Point - 4.4 (per 1000 patient-years)"
                   ,"2 Points - 13.1 (per 1000 patient-years)"
                   ,"3 Points - 28.7 (per 1000 patient-years)"
                   ,"4 Points - 56.1 (per 1000 patient-years)"
                   ,"5 Points - 56.1 (per 1000 patient-years)"
                   ,"6 Points - 56.1 (per 1000 patient-years)"
                   ,"7 Points - 56.1 (per 1000 patient-years)"
                   ,"8 Points - 56.1 (per 1000 patient-years)")
  myrisk <- c("2.4"
              ,"4.4"
              ,"13.1"
              ,"28.7"
              ,"56.1"
              ,"56.1"
              ,"56.1"
              ,"56.1"
              ,"56.1")
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "hfdiab")
    dfOriginal <- df
    for(i in c("af","cad","egfr")){
      df[ , i] <- ifelse(df[ ,i]%in%c("Yes","<60") , 1 , 0)
    }
    df$hf <- ifelse(df$hf%in%"Yes" , 2 , 0)
    df$albcrea <- ifelse( df$albcrea %in% c("<30","<3.39") , 0
                       , ifelse(df$albcrea %in% c("30-300","3.39-33.9") , 1
                                , ifelse(df$albcrea %in% c(">300",">33.9") , 2 , 0)))
    dfOriginal$hfdiab_Points <- rowSums(df[ , c("hf","af","cad","egfr","albcrea")])
    dfOriginal$hfdiab_Risk_per1k_patientyear <- myrisk[ dfOriginal$hfdiab_Points+1 ]
    return(dfOriginal)
  } else {
    hf <- ifelse( hf%in%"Yes" , 2 , 0)
    af <- ifelse( af%in%"Yes" , 1 , 0)
    cad <- ifelse( cad%in%"Yes" , 1 , 0)
    egfr <- ifelse( egfr%in%"<60" , 1 , 0)
    albcrea <- ifelse( albcrea %in% c("<30","<3.39") , 0
                       , ifelse(albcrea %in% c("30-300","3.39-33.9") , 1
                                , ifelse(albcrea %in% c(">300",">33.9") , 2 , 0)))
    o <- sum(hf,af,cad,egfr,albcrea)
    return(consequence[o+1])
  }
}

#' TIMI Biomarker Score for Heart Failure in Diabetes (TBS-HF(DM))
#'
#' This score estimates the incidence rates of hospitalization for heart failure
#'  in patients with diabetes using clinically available biomarkers.
#'
#' @param df data.frame in the same format as timi_examples$biomhfdiab See details
#' @param hf Prior heart failure. "Yes" or "No"
#' @param hstnt High Sensitive Troponin T in ng/L
#' (accepted values: "<6 ng/L" , "6 to <10 ng/L" , "10 to <14 ng/L" , ">=14 ng/L")
#' or ug/L (accepted values: "<.006 ug/L" , ".006 to <.01 ug/L" , ".01 to <.014 ug/L" , ">=.014 ug/L")
#' @param ntprobnp NT-proBNP group in ng/L (accepted values: "<50 ng/L" , "50 to <125 ng/L" , "125 to <450 ng/L" , ">=450 ng/L")
#' or pmol/L (accepted values: "<5.9 pmol/L" , "5.9 to <14.8 pmol/L" , "14.8 to <53.2 pmol/L" , ">=53.2 pmol/L")
#' @param hstnt_unit Either "ngl" (default) for ng/L or "ugl" for ug/L
#' @param ntprobnp_unit Either "ngl" (default) for ng/L or "pmol" pmol/L
#' @examples
#' # One patient
#' computeBIOMHFDIAB(hf = "Yes" ,hstnt = "<6 ng/L" ,ntprobnp = "5.9 to <14.8 pmol/L"
#' ,hstnt_unit="ngl",ntprobnp_unit="pmol")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_biomhfdiab <- computeBIOMHFDIAB(df = timi_examples$biomhfdiab)
#' @return if df is not used, returns the score alone in 1000 patient-years,
#'   otherwise returns the original
#'   dataframe with "biomhfdiab_Points", "biomhfdiab_Risk_per1k_patientyear" and
#'   "biomhfdiab_Risk_level" as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$biomhfdiab.
#' See data(timi_examples) to see all example files.
#' @references Berg DD, et al. [A Biomarker-Based Score for Risk of
#' Hospitalization for Heart Failure in Patients With
#' Diabetes](https://diabetesjournals.org/care/article/44/11/2573/138510/A-Biomarker-Based-Score-for-Risk-of).
#' Diabetes Care. 2021 Nov;44(11):2573-2581.
#' @importFrom plyr mapvalues
#' @export
computeBIOMHFDIAB <- function(df=NULL,hf,hstnt,ntprobnp
                              ,hstnt_unit=c("ngl","ugl")
                              ,ntprobnp_unit=c("ngl","pmol")){
  consequence <- c("0 points - Low risk - 1 (per 1000 patient-years)"
                   ,"1 point - Low risk - 1 (per 1000 patient-years)"
                   ,"2 points - Low risk - 1 (per 1000 patient-years)"
                   ,"3 points - Intermediate risk - 2.5 (per 1000 patient-years)"
                   ,"4 points - Intermediate risk - 2.5 (per 1000 patient-years)"
                   ,"5 points - Intermediate risk - 2.5 (per 1000 patient-years)"
                   ,"6 points - High risk - 13.4 (per 1000 patient-years)"
                   ,"7 points - High risk - 13.4 (per 1000 patient-years)"
                   ,"8 points - High risk - 13.4 (per 1000 patient-years)"
                   ,"9 points - Very high risk - 62.7 (per 1000 patient-years)"
                   ,"10 points - Very high risk - 62.7 (per 1000 patient-years)"
                   ,"11 points - Very high risk - 62.7 (per 1000 patient-years)")
  hstnt_unit <- hstnt_unit[1]
  ntprobnp_unit <- ntprobnp_unit[1]
  myrisk <- rep(c(1,2.5,13.4,62.7) , c(3,3,3,3))
  myrisk_char <- rep(c("Low Risk","Intermediate Risk","High Risk","Very High Risk") , c(3,3,3,3))
  hstntVec <- c(0,1,2,3)
  ntprobnpVec <- c(0,2,4,6)
  if(!is.null(df)){
    dfmode <- TRUE
    dfOriginal <- df
    # change unit to standard for us
    # hstnt in ng/L
    # ntprobnp in ng/L
    if(hstnt_unit=="ugl"){
      df$hstnt <- plyr::mapvalues(df$hstnt
        , from = c("<.006 ug/L" , ".006 to <.01 ug/L" , ".01 to <.014 ug/L" , ">=.014 ug/L")
        , to = c("<6 ng/L" , "6 to <10 ng/L" , "10 to <14 ng/L" , ">=14 ng/L"))
      hstnt_unit <- "ngl"
    }
    if(ntprobnp_unit=="pmol"){
      df$ntprobnp <- plyr::mapvalues(df$ntprobnp
                                  , from = c("<5.9 pmol/L" , "5.9 to <14.8 pmol/L" , "14.8 to <53.2 pmol/L" , ">=53.2 pmol/L")
                                  , to = c("<50 ng/L" , "50 to <125 ng/L" , "125 to <450 ng/L" , ">=450 ng/L"))
      ntprobnp_unit <- "ngl"
    }
    .checkDF(df , type = "biomhfdiab")

    df$hf <- ifelse(df$hf%in%"Yes" , 2 , 0)
    if(hstnt_unit=="ngl"){
      names(hstntVec) <- c("<6 ng/L" , "6 to <10 ng/L" , "10 to <14 ng/L" , ">=14 ng/L")
    } else {
      names(hstntVec) <- c("<.006 ug/L" , ".006 to <.01 ug/L" , ".01 to <.014 ug/L" , ">=.014 ug/L")
    }
    df$hstnt <- hstntVec[df$hstnt]
    if(ntprobnp_unit=="ngl"){
      names(ntprobnpVec) <- c("<50 ng/L" , "50 to <125 ng/L" , "125 to <450 ng/L" , ">=450 ng/L")
    } else {
      names(ntprobnpVec) <- c("<5.9 pmol/L" , "5.9 to <14.8 pmol/L" , "14.8 to <53.2 pmol/L" , ">=53.2 pmol/L")
    }
    df$ntprobnp <- ntprobnpVec[df$ntprobnp]
    dfOriginal$biomhfdiab_Points <- rowSums(df[ , c("hf" , "hstnt" , "ntprobnp")])
    dfOriginal$biomhfdiab_Risk_per1k_patientyear <- myrisk[ dfOriginal$biomhfdiab_Points+1 ]
    dfOriginal$biomhfdiab_Risk_level <- myrisk_char[ dfOriginal$biomhfdiab_Points+1 ]
    return(dfOriginal)
  } else {
    hf <- ifelse( hf%in%"Yes" , 2 , 0)
    if(hstnt_unit=="ngl"){
      names(hstntVec) <- c("<6 ng/L" , "6 to <10 ng/L" , "10 to <14 ng/L" , ">=14 ng/L")
    } else {
      names(hstntVec) <- c("<.006 ug/L" , ".006 to <.01 ug/L" , ".01 to <.014 ug/L" , ">=.014 ug/L")
    }
    hstnt <- hstntVec[hstnt]

    if(ntprobnp_unit=="ngl"){
      names(ntprobnpVec) <- c("<50 ng/L" , "50 to <125 ng/L" , "125 to <450 ng/L" , ">=450 ng/L")
    } else {
      names(ntprobnpVec) <- c("<5.9 pmol/L" , "5.9 to <14.8 pmol/L" , "14.8 to <53.2 pmol/L" , ">=53.2 pmol/L")
    }
    ntprobnp <- ntprobnpVec[ntprobnp]
    o <- sum(hf,hstnt,ntprobnp)
    return(consequence[o+1])
  }
}

#' TIMI Risk Score for Secondary Prevention in ASCVD (TRA2P)
#'
#' This score estimates the 3-year risk of cardiovascular death, myocardial
#' infarction, or ischemic stroke in patients with stable ischemic heart
#' disease and prior myocardial infarction.
#'
#' @param df data.frame in the same format as timi_examples$tra2p. See details
#' @param hf Prior heart failure. "Yes" or "No"
#' @param hypert History of hypertension. "Yes" or "No"
#' @param age Age >= 75 years. "Yes" or "No"
#' @param diabetes History of Diabetes. "Yes" or "No"
#' @param stroke Prior stroke. "Yes" or "No"
#' @param cabg Prior coronary artery bypass grafting. "Yes" or "No"
#' @param pad Peripheral arterial disease. "Yes" or "No"
#' @param egfr Renal dysfunction with eGFR < 60 ml/min/1.73 m^2. "Yes" or "No"
#' @param smoker Current smoking. "Yes" or "No"
#' @examples
#' # One patient
#' computeTRA2P(hf = "Yes",hypert="No",age="No",diabetes="No"
#' ,stroke="Yes",cabg="No",pad="No",egfr="Yes",smoker="No")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_tra2p <- computeTRA2P(df = timi_examples$tra2p)
#' @return if df is not used, returns the score alone in 1000 patient-years,
#'   otherwise returns the original
#'   dataframe with "tra2p_Risk_percentage" , "tra2p_Risk_Vorapaxar_VS_Placebo",
#'   "tra2p_Risk_Simvastatin_VS_Ezetimibe" and
#'   "tra2p_Risk_Aspirin_VS_Rivaroxaban" as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$tra2p.
#' See data(timi_examples) to see all example files.
#' @references Bohula EA, et al. [Atherothrombotic Risk Stratification and the
#' Efficacy and Safety of Vorapaxar in Patients With Stable Ischemic
#' Heart Disease and Previous Myocardial
#' Infarction](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.115.019861)
#' Circulation. 2016;134:304-313.
#' @export
computeTRA2P <- function(df=NULL,hf,hypert,age,diabetes,stroke,cabg,pad,egfr,smoker){
  consequence <- c("0 Points - 3.5% rate"
                   ,"1 Point - 6.8% rate"
                   ,"2 Points - 9.9% rate"
                   ,"3 Points - 14.5% rate"
                   ,"4 Points - 21.8% rate"
                   ,"5 Points - 28.8% rate"
                   ,"6 Points - 45.3% rate"
                   ,">=7 Points - 58.6% rate")
  consequence_num <- c(3.5 , 6.8 , 9.9 , 14.5 , 21.8 , 28.8 , 45.3 , 58.6)
  consequence_vor <- c("3.5% VS 3.6%"
                       , rep("6.0% VS 8.1%",2)
                       , rep("14.5% VS 17.7%",5))
  consequence_eze <- c(rep("13.1% VS 14%",2)
                       , "21.5% VS 19.3%"
                       , rep("40.2% VS 33.9%",5))
  consequence_com <- c(rep("2.98% VS 2.18%",2)
                       , "4.66% VS 3.69%"
                       , rep("8.18% VS 6.11%",5))
  mycols <- c("hf","hypert","age","diabetes","stroke","cabg","pad","egfr","smoker")
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "tra2p")
    dfOriginal <- df
    for(i in mycols){
      df[ ,i] <- ifelse(df[ , i] %in% "Yes" , 1 , 0)
    }
    dfOriginal$tra2p_Points <- rowSums(df[ , mycols])
    minTop <- ifelse( dfOriginal$tra2p_Points > 7 , 7 , dfOriginal$tra2p_Points)
    dfOriginal$tra2p_Risk_percentage <- consequence_num[ minTop+1 ]
    dfOriginal$tra2p_Risk_Vorapaxar_VS_Placebo <- consequence_vor[ minTop+1 ]
    dfOriginal$tra2p_Risk_Simvastatin_VS_Ezetimibe <- consequence_eze[ minTop+1 ]
    dfOriginal$tra2p_Risk_Aspirin_VS_Rivaroxaban <- consequence_com[ minTop+1 ]
    return(dfOriginal)
  } else {
    hf <- ifelse( hf%in%"Yes" , 1 , 0)
    hypert <- ifelse( hypert%in%"Yes" , 1 , 0)
    age <- ifelse( age%in%"Yes" , 1 , 0)
    diabetes <- ifelse( diabetes%in%"Yes" , 1 , 0)
    stroke <- ifelse( stroke%in%"Yes" , 1 , 0)
    cabg <- ifelse( cabg%in%"Yes" , 1 , 0)
    pad <- ifelse( pad%in%"Yes" , 1 , 0)
    egfr <- ifelse( egfr%in%"Yes" , 1 , 0)
    smoker <- ifelse( smoker%in%"Yes" , 1 , 0)
    o <- sum(hf,hypert,age,diabetes,stroke,cabg,pad,egfr,smoker)
    o <- ifelse(o > 7 , 7 , o)
    out <- c( consequence[o+1]
              ,consequence_vor[o+1]
              ,consequence_eze[o+1]
              ,consequence_com[o+1])
    names(out) <- c("tra2p_Risk_percentage" , "tra2p_Risk_Vorapaxar_VS_Placebo"
                    , "tra2p_Risk_Simvastatin_VS_Ezetimibe"
                    , "tra2p_Risk_Aspirin_VS_Rivaroxaban")
    return(out)
  }
}

#' TIMI Risk Index for Mortality (TRI)
#'
#' The TIMI Risk Index (TRI) is a simple calculator to predict 30-day mortality,
#'  which has been validated in a registry of 150,000 STEMI patients and
#'  a registry of over 300,000 NSTEMI patients. It has been shown to both
#'  predict in-hospital and long-term mortality.
#'  TIMI Risk Index = Heart Rate * (Age/10)^2 / Systolic BP
#'
#' @param df data.frame in the same format as timi_examples$tri. See details
#' @param hr Heart Rate. Numerical
#' @param age Age in years
#' @param sbp Systolic Blood Pressure
#' @param mitype One of "STEMI", "NSTEMI", "NSTEMI-RT" or "NSTEMI-noRT"
#' @examples
#' # One patient
#' computeTRI(hr = 80,age=65,sbp=100,mitype="NSTEMI")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_tri <- computeTRI(df = timi_examples$tri)
#' @return if df is not used, returns the score alone as %,
#'   otherwise returns the original
#'   dataframe with "tri_Risk_percentage" as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$tri.
#' See data(timi_examples) to see all example files.
#' @references Wiviott SD, et al [National Registry of Myocardial Infarction.
#' Application of the Thrombolysis in Myocardial Infarction risk index in
#' non-ST-segment elevation myocardial infarction: evaluation of patients
#' in the National Registry of Myocardial
#' Infarction](https://www.sciencedirect.com/science/article/pii/S0735109706003007?via%3Dihub).
#' J Am Coll Cardiol. 2006 Apr 18;47(8):1553-8. Epub 2006 Mar 29. PubMed PMID: 16630990.
#'
#' Morrow D, et al. [A simple risk index for rapid initial triage of patients
#' with ST-elevation myocardial infarction: an InTIME II
#' substudy](https://www.sciencedirect.com/science/article/pii/S0140673601066491).
#' Lancet. 2001 Volume 358, Issue 9293, 10 November 2001, Pages 1571-1575
#' @export
computeTRI <- function(df=NULL,hr,age,sbp,mitype){
  mycols <- c("hr","age","sbp")
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "tri")
    dfOriginal <- df
    df$tri <- apply( df[ , mycols] , 1 , function(k) k["hr"]*(k["age"]/10)^2/k["sbp"])
    tricut <- as.character(cut(df$tri , c(seq(0,80,10),Inf) , include.lowest = TRUE))
    tricut <- paste0("d" , gsub("\\(|\\[|\\)|\\]" , "" , sub("," , "_" , tricut)))
    dfOriginal$tri_Risk_percentage <- unlist(lapply( 1:length(tricut) , function(k) {
      unlist(triTab[ triTab$Pop == df$mitype[k] , tricut[k] , drop = TRUE])
    }))
    return(dfOriginal)
  } else {
    tri <- hr*(age/10)^2/sbp
    tricut <- as.character(cut(tri , c(seq(0,80,10),Inf) , include.lowest = TRUE))
    tricut <- paste0("d" , gsub("\\(|\\[|\\)|\\]" , "" , sub("," , "_" , tricut)))
    triOut <- unlist(triTab[ triTab$Pop == mitype , , drop = TRUE])
    return(unname(triOut[tricut]))
  }
}

#' TIMI AF score for severe net clinical outcome (TIMI-AF Score)
#'
#' The choice between initiating a non-vitamin K antagonist oral anticoagulant
#' (NOAC) and a vitamin K antagonist (VKA) in patients with atrial fibrillation
#' (AF) may be challenging. To assist in this decision, we developed a risk
#' score to assess patients' net clinical outcome by identifying the risk of
#' disabling stroke, life-threatening bleeding, and all-cause mortality
#' at 3 years and identify patients for whom a therapeutic benefit of
#' NOACs over VKA is predicted.
#'
#' @param df data.frame in the same format as timi_examples$timiaf. See details
#' @param age Age group. Available values: "<=65" , "66-74" , ">=75"
#' @param lvef Left Ventricular Ejection Fraction in %. Available values:
#' ">=50", "30-49" , "<30" , "unknown"
#' @param hemo Hemoglobin (g/dL). Available values: "<13" or ">=13"
#' @param race Ethnicity. Available values" "White", "Black", "Asian", "Other"
#' @param af Baseline Atrial Fibrillation. "Yes" or "No"
#' @param hxstroke Prior stroke. "Yes" or "No"
#' @param creat Creatinine (umol/L). Available values: "<110" or ">=110"
#' @param sex Sex. Available values: "Female" or "Male"
#' @param diabetes Baseline Diabetes. "Yes" or "No"
#' @param hxcd Kistory of Carotid disease. "Yes" or "No"
#' @param hxmi Prior myocardial infarction. "Yes" or "No"
#'
#' @examples
#' # One patient
#' computeAFIB(age="<=65",lvef="30-49",hemo=">=13"
#' ,race="Asian",af="Yes",hxstroke="Yes",creat="<110",sex="Female"
#' ,diabetes="No",hxcd="Yes",hxmi="No")
#' # Multiple subjects
#' data(timi_examples)
#' timi_example_timiaf <- computeAFIB(df = timi_examples$timiaf)
#' @return if df is not used, returns the score alone as % for Warfarin,
#' Low Dose Edoxoban and High Dose Edoxoban. Otherwise returns the original
#'   dataframe with "Points", Warfarin", "LDER" and "HDER" as extra column
#' @details
#' if df is NULL, the user needs to supply single values for each variables.
#' If df is used, df is a data.frame in the same format of timi_examples$timiaf.
#' See data(timi_examples) to see all example files.
#' @references Fanola CL, et al. [A novel risk prediction score in atrial
#' fibrillation for a net clinical outcome from the ENGAGE AF-TIMI 48
#' randomized clinical trial](https://pubmed.ncbi.nlm.nih.gov/28064150/).
#' Eur Heart J. 2017 Mar 21;38(12):888-896. doi:10.1093/eurheartj/ehw565.
#' PMID: 28064150.
#' @export
computeAFIB <- function(df=NULL,age,lvef,hemo,race,af,hxstroke,creat,sex,diabetes,hxcd,hxmi){
  mycols <- c("age","lvef","hemo","race","af","hxstroke","creat","sex","diabetes","hxcd","hxmi")
  if(!is.null(df)){
    dfmode <- TRUE
    .checkDF(df , type = "timiaf")
    dfOriginal <- df
    df$sampID <- 1:nrow(df)
    df$age2 <- ifelse(df$age %in% ">=75" , 3 , ifelse(df$age %in% "66-74" , 2 , 0))
    df$lvef2 <- ifelse(df$lvef %in% "<30" , 3
                       , ifelse(df$lvef %in% "30-49" , 2
                                , ifelse(df$lvef %in% ">=50" , 0 , 1)))
    df$hemo2 <- ifelse( df$hemo %in% "<13" , 2 , 0)
    df$race2 <- ifelse(df$race %in% "White" , 0 , 2)
    df$af2 <- ifelse(df$af %in% "Yes" , 1 , 0)
    df$hxstroke2 <- ifelse(df$hxstroke %in% "Yes" , 1 , 0)
    df$creat2 <- ifelse(df$creat %in% ">=110" , 1 , 0)
    df$sex2 <- ifelse( df$sex %in% "Male" , 1 , 0)
    df$diabetes2 <- ifelse(df$diabetes %in% "Yes" , 1 , 0)
    df$hxcd2 <- ifelse(df$hxcd %in% "Yes" , 1 , 0)
    df$hxmi2 <- ifelse(df$hxmi %in% "Yes" , 1 , 0)
    df$Points <- rowSums(df[ , paste0(mycols , "2")])
    df$PointsMax <- ifelse(df$Points>12 , 12 , df$Points)
    df <- merge(df , afTab , by.x = "PointsMax" , by.y = "Score" , all.x=TRUE)
    # Keep matrix in the same order as dfOriginal
    df <- df[ order( df$sampID) , ]
    return(cbind(dfOriginal , df[ , c("Points" , "Warfarin" , "LDER" , "HDER")]))
  } else {
    age2 <- ifelse(age %in% ">=75" , 3 , ifelse(age %in% "66-74" , 2 , 0))
    lvef2 <- ifelse(lvef %in% "<30" , 3 , ifelse(lvef %in% "30-49" , 2 , ifelse(lvef %in% ">=50" , 0 , 1)))
    hemo2 <- ifelse( hemo %in% "<13" , 2 , 0)
    race2 <- ifelse(race %in% "White" , 0 , 2)
    af2 <- ifelse(af %in% "Yes" , 1 , 0)
    hxstroke2 <- ifelse(hxstroke %in% "Yes" , 1 , 0)
    creat2 <- ifelse(creat %in% ">=110" , 1 , 0)
    sex2 <- ifelse( sex %in% "Male" , 1 , 0)
    diabetes2 <- ifelse(diabetes %in% "Yes" , 1 , 0)
    hxcd2 <- ifelse(hxcd %in% "Yes" , 1 , 0)
    hxmi2 <- ifelse(hxmi %in% "Yes" , 1 , 0)
    o <- min( sum(age2,lvef2,hemo2,race2,af2,hxstroke2,creat2,sex2,diabetes2,hxcd2,hxmi2) , 12)
    out <- c( afTab$Warfarin[o+1]
              ,afTab$LDER[o+1]
              ,afTab$HDER[o+1])
    out <- paste( paste(o , "points - ") , paste0(out , "%") )
    names(out) <- c("Warfarin" , "Low Dose Edoxoban" , "High Dose Edoxoban")
    return(out)
  }
}
