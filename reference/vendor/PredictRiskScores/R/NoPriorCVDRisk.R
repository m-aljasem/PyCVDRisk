#' PREDICT CVD (2017) Risk Score for People Without CVD
#'
#' \code{NoPriorCVDRisk} calculates the 5 year absolute risk of cardiovascular disease (CVD) for people without a history of atherosclerotic CVD.
#' In this version, BMI is not included as a predictor. The outcome of interest is the 5-year risk of a non-fatal or fatal CVD event, including hospitalisation
#' for coronary heart disease, stroke or other cerebrovascular disease (including transient ischaemic attack), peripheral vascular disease and heart failure,
#' or cardiovascular death.
#'
#' @usage NoPriorCVDRisk(dat, sex, age, eth, nzdep, exsmoker, smoker, diabetes,
#'                af, familyhx, sbp, tchdl, bpl, lld, athrombi, ...)
#'
#' @param dat     an optional data.frame or data.table containing input data (see details)
#' @param sex     binary sex or gender
#' @param age     age in years  (see details)
#' @param eth     ethnicity (see details)
#' @param nzdep     socio-economic deprivation (see details)
#' @param exsmoker ex-smoker or recently quit
#' @param smoker    currently smoking
#' @param diabetes  diabetes status
#' @param af        atrial fibrillation status
#' @param familyhx family history of premature CVD
#' @param sbp       measured systolic blood pressure in mmHg
#' @param tchdl     most recent value of total:HDL cholesterol
#' @param bpl       receiving at least one blood pressure lowering medication
#' @param lld       receiving lipid lowering medication
#' @param athrombi  receiving antiplatelet or anticoagulant medication
#' @param ...       further arguments (see values)
#'
#' @details  When a dataset is supplied, a risk score is produced for each row of data, resulting in a numeric vector of the same length.
#' Each argument requires the variable name from the dataset \code{dat} that corresponds with the parameter. If the parameter \code{dat} is not supplied, then each argument is assigned an individual's
#' actual parameter value. \cr
#'
#' The risk prediction equations were developed from a cohort of people aged 30 to 74 years. Additional analyses indicate that the sex-specific risk equations performed adequately in those aged 75-79 years.
#' People aged 18-29 years or 80 years and older, are outside the range used to derive the equation, and therefore risk will be even more of an approximation. The function
#' will calculate ages 18-29 as 30; and ages 80-110 as 79. All other age inputs are invalid and will return \code{NA}. \cr
#'
#' The co-efficients for ethnicity apply only to the following groups: European, Maori, Pacific, Indian, and (non-Indian) Asian. To obtain a risk estimate, ensure that the
#' ethnicity input is either labelled or encoded using one of the values listed below (see values). All other inputs are invalid and will return \code{NA}. \cr
#'
#' The scale for socioeconomic deprivation is derived from the New Zealand Index of Deprivation; a small area-based measure that combines census data relating to income,
#' home ownership, employment, qualifications, family structure, housing, access to transport and communications. The equations require NZDep to be categorised as quintiles,
#' with 1 being the least deprived and 5 being the most deprived.
#'
#' @return
#' returns either a single 5-year CVD risk estimate, or a numeric vector of risk estimates if \code{dat} is provided.
#' Input values for each parameter must conform to the following convention:
#'
#' \item{sex}{label or encode as one of the following:
#'            \itemize{
#'              \item M, Male, 1
#'              \item F, Female, 0
#'              }}
#' \item{age}{numeric value for years of age between 20 and 110}
#' \item{eth}{label or encode as one of the following:
#'            \itemize{
#'              \item NZ European, European, NZEO, Euro, E, 1, 10, 11, or 12
#'              \item Maori, NZMaori, NZ Maori, M, 2, or 21
#'              \item Pacific, Pacific Islander, PI, P, 3, 30, 31, 32, 33, 34, 35, 36, or 37
#'              \item Indian, Fijian Indian, South Asian, IN, I, or 43
#'              \item Asian, Other Asian, SE Asian, East Asian, Chinese, ASN, A, 4, 40, 41, 42, or 44
#'              \item note: Other Asian includes non-Indian South Asian
#'              }}
#' \item{nzdep}{numeric value between 1 and 5}
#' \item{exsmoker}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, Ex, Ex-smoker, Exsmoker, E, 1, T, TRUE
#'              \item N, No, Non-smoker, Non, 0, F, FALSE
#'              }}
#' \item{smoker}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, Smoker, Current, S, 1, T, TRUE
#'              \item N, No, Non-smoker, Non, 0, F, FALSE
#'              }}
#' \item{diabetes, \cr af, familyhx}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#' \item{bpl, lld,\cr athrombi}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#' \item{sbp, tchdl}{numeric value of measured result. Note:
#'            \itemize{
#'              \item SBP and total:HDL values must be available
#'              }}
#' \item{...}{further arguments:
#'            \itemize{
#'              \item \code{dp} numeric value to set decimal place; default is 4
#'              \item \code{allow.age} logical. Whether or not age range is extended outside of 30 - 74; default is TRUE. If set to FALSE, then \code{NA} is returned as risk estimate.
#'              \item \code{allow.na} logical. Whether or not missing values for binary variables and smoking status are treated as 0; default is TRUE. If set to FALSE, then \code{NA} is returned as risk estimate.
#'              }}
#'
#' @section See Also:
#' \code{\link{NoPriorCVDRisk}} \cr
#' \code{\link{NoPriorCVDRisk_BMI}} \cr
#' \code{\link{NoPriorCVDRisk_Policy}} \cr
#' \code{\link{NoPriorCVDBleedRisk}} \cr
#' \code{\link{NoPriorT2DRisk}} \cr
#' \code{\link{PostCVDRisk}} \cr
#' \code{\link{PostACSRisk}} \cr
#'
#' @author
#' Billy Wu (R Developer) and Romana Pylypchuk (Principal Investigator)
#'
#' @export
#' @references
#' Pylypchuk R, Wells S, Kerr A, Poppe K, Riddell T, Harwood M, et al. Cardiovascular disease risk prediction equations in 400 000 primary care patients in New Zealand: a derivation and validation study. Lancet 2018;391:1897-907.
#'
#' \href{https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(18)30664-0/fulltext}{Full Article}
#'
#' @export
#' @examples
#' # As Calculator (dataset not provided)
#' NoPriorCVDRisk(sex=F, age=30, eth=M, exsmoker=0, smoker=TRUE, nzdep=3, diabetes=Y,
#'                af=F, familyhx=1, lld=Y, athrombi=yes, bpl=T, sbp=150, tchdl=5)
#'
#' NoPriorCVDRisk(sex=1, age=82, eth="Asian", exsmoker=NA, smoker=Y, nzdep=4, diabetes=N,
#'                af=1, familyhx=1, lld=T, athrombi=F, bpl=0, sbp=140, tchdl=3.3,
#'                dp = 5, allow.age = FALSE, allow.na = FALSE)
#'
#' # As a vectoriser (dataset provided)
#' NoPriorCVDRisk(dat=TEST, sex=sex, age=age, eth=eth, nzdep=nzdep, exsmoker=exsmoker,
#'                smoker=smoker,  diabetes=diabetes, af=af, familyhx=familyhx, lld=lld,
#'                athrombi=athromb, bpl=bpl, sbp=sbp, tchdl=tchdl, allow.na =TRUE)
#'
# --- Code ---
NoPriorCVDRisk <- function(dat, sex, age, eth, nzdep, exsmoker, smoker, diabetes, af, familyhx, sbp, tchdl, bpl, lld, athrombi,...){

  # Params
  demo.vars   <- c("sex", "age", "eth", "nzdep")
  smk.vars    <- c("exsmoker", "smoker")
  bin.vars    <- c("diabetes", "af", "familyhx", "lld", "athrombi", "bpl")
  num.vars    <- c("sbp", "tchdl")

  # Calls
  call      <- gsub("()", "",  match.call()[1])
  is.table  <- deparse(substitute(dat))!=""
  input     <- as.list(match.call()[-1])

  if(length(list(...)) == 0){

    dp        <- 4
    allow.age <- TRUE
    allow.na  <- TRUE

  } else {

    default <- setdiff(c("dp", "allow.age", "allow.na"), names(list(...)))

    if(length(default) %in% 1:2){

      lapply(default,
             function(x){

               if(x == "dp"){
                 val <- 4
               } else if(x == "allow.na") {
                 val <- TRUE
               } else {
                 val <- TRUE
               }
               assign(x, val, envir = parent.frame(2))
             })
    }

    lapply(names(list(...)),
           function(x)
             assign(x, unlist(list(...)[x]),
                    envir = parent.frame(2)))

  }

  # ParamCheck
  vars <- c(demo.vars, bin.vars, smk.vars, num.vars)

  ParamCheck(input, vars, call, is.table, allow.age, allow.na)

  # Values
  f.ind <- which(tolower(input$sex) %in% ok.female)
  m.ind <- which(tolower(input$sex) %in% ok.male)

  demo.vals <- list(age      = input$age,
                    maori    = +(tolower(input$eth) %in% ok.maori),
                    pacific  = +(tolower(input$eth) %in% ok.pi),
                    indian   = +(tolower(input$eth) %in% ok.indian),
                    asian    = +(tolower(input$eth) %in% ok.asian),
                    exsmoker = +(tolower(input$exsmoker) %in% ok.exsmkr),
                    smoker   = +(tolower(input$smoker) %in% ok.smoker),
                    nzdep    = input$nzdep)

  bin.vals <- sapply(bin.vars,
                     function(x){
                       +(tolower(input[[x]]) %in% ok.true)
                     },
                     USE.NAMES = TRUE,
                     simplify = FALSE)

  num.vals <- sapply(num.vars,
                      function(x){
                        as.numeric(input[[x]])
                      },
                      USE.NAMES = TRUE,
                      simplify = FALSE)

  values <- c(demo.vals, bin.vals, num.vals) # Order sensitive!

  # Adjustments
  if(allow.age){
    values$age[which(values$age < 30)] <- 30
    values$age[which(values$age > 79)] <- 80
  }

  if(!allow.na){

    vars <- c(smk.vars, bin.vars)

    values[vars] <- sapply(vars,
                           function(x){

                             input[[x]] <- if(is.name(input[[x]])){
                               as.character(input[[x]])
                             }
                               replace(values[[x]],
                                       which(is.na(input[[x]])),
                                       NA)
                           },
                           USE.NAMES = TRUE,
                           simplify = FALSE)
  }

  values$exsmoker[which(values$smoker == 1)] <- 0

  # Recentering
  values$age[f.ind] <- values$age[f.ind] - 56.13665
  values$age[m.ind] <- values$age[m.ind] - 51.79953
# browser()
  values$nzdep[f.ind] <- values$nzdep[f.ind] - 2.990826
  values$nzdep[m.ind] <- values$nzdep[m.ind] - 2.972793

  values$sbp[f.ind] <- values$sbp[f.ind] - 129.0173
  values$sbp[m.ind] <- values$sbp[m.ind] - 129.1095

  values$tchdl[f.ind] <- values$tchdl[f.ind] - 3.726268
  values$tchdl[m.ind] <- values$tchdl[m.ind] - 4.38906

  # Interaction
  values$int_age_diab <- ifelse(values$diabetes == 0, 0, values$age)
  values$int_age_sbp  <- values$age * values$sbp
  values$int_sbp_bplt <- ifelse(values$bpl == 0, 0, values$sbp)

  # Coefficients
  fem.coeff <- list(age     = 0.0756412,
                    maori   = 0.3910183,
                    pacific = 0.2010224,
                    indian  = 0.1183427,
                    asian   = -0.28551,
                    ex_smoke = 0.087476,
                    cur_smoke = 0.6226384,
                    nzdep     = 0.1080795,
                    diabetes = 0.5447632,
                    af       = 0.8927126,
                    familyhx = 0.0445534,
                    lld = -0.0593798,
                    athrombi = 0.1172496,
                    bpl = 0.339925,
                    sbp = 0.0136606,
                    tchdl = 0.1226753,
                    int_age_diab = -0.0222549,
                    int_age_sbp = -0.0004425,
                    int_sbp_bplt = -0.004313)

  male.coeff <- list(age     = 0.0675532,
                     maori   = 0.2899054,
                     pacific = 0.1774195,
                     indian  = 0.2902049,
                     asian   = -0.3975687,
                     ex_smoke = 0.0753246,
                     cur_smoke = 0.5058041,
                     nzdep    = 0.0794903,
                     diabetes = 0.5597023,
                     af       = 0.5880131,
                     familyhx = 0.1326587,
                     lld = -0.0537314,
                     athrombi = 0.0934141,
                     bpl = 0.2947634,
                     sbp = 0.0163778,
                     tchdl = 0.1283758,
                     int_age_diab = -0.020235,
                     int_age_sbp = -0.0004184,
                     int_sbp_bplt = -0.0053077)

  # Turn into function
  value.score <- mapply(function(val, f.coeff, m.coeff){

    effect <- rep(0, length(input$sex))
    effect <- replace(effect, f.ind, val[f.ind] * f.coeff)
    effect <- replace(effect, m.ind, val[m.ind] * m.coeff)

    return(effect)
  },
  val = values,
  f.coeff = fem.coeff,
  m.coeff = male.coeff,
  SIMPLIFY = F)

  sum.score <- Reduce("+", value.score)

  estimate <- rep(0, length(sum.score))
  estimate <- replace(estimate, f.ind, 1 - 0.983169213058 ^ exp(sum.score[f.ind]))
  estimate <- replace(estimate, m.ind, 1 - 0.974755526232 ^ exp(sum.score[m.ind]))

  rounded.val <- as.numeric(formatC(round(estimate, dp),
                                    format = 'f',
                                    digits = dp))

  if(length(ls(pattern = "inval.")) >= 1){

    rounded.val <- replace(rounded.val,
                           unlist(mget(ls(pattern = "inval."))),
                           NA)
  }

  return(rounded.val)

}






