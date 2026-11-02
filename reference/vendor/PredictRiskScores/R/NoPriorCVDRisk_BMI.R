#' PREDICT CVD (2018.2) Risk Score for People Without Prior CVD
#'
#' \code{NoPriorCVDRisk_BMI} calculates the 5 year absolute risk of cardiovascular disease (CVD) for people without a history of atherosclerotic CVD.
#' In this version, BMI is used as a predictor. The outcome of interest is the 5-year risk of a non-fatal or fatal CVD event, including hospitalisation
#' for coronary heart disease, stroke or other cerebrovascular disease (including transient ischaemic attack), peripheral vascular disease and heart failure,
#' or cardiovascular death.
#'
#' @usage NoPriorCVDRisk_BMI(dat, sex, age, eth, nzdep, exsmoker, smoker, diabetes,
#'                    af, familyhx, sbp, tchdl, bmi, bpl, lld, athrombi, ...)
#'
#' @inheritParams NoPriorCVDRisk
#' @param bmi  body mass index in kg/m^2
#'
#' @inherit NoPriorCVDRisk details
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
#' \item{diabetes,\cr af, hf}{label or encode as one of the following:
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
#' \item{bmi}{numeric value of calculated BMI. If BMI is unknown, input as \code{NA}}
#' \item{...}{further arguments:
#'            \itemize{
#'              \item \code{dp} numeric value to set decimal place; default is 4
#'              \item \code{allow.age} logical. Whether or not age range is extended outside of 30 - 74; default is TRUE. If set to FALSE, then \code{NA} is returned as risk estimate.
#'              \item \code{allow.na} logical. Whether or not missing values for binary variables and smoking status are treated as 0; default is TRUE. If set to FALSE, then \code{NA} is returned as risk estimate.
#'              }}
#'
#' @inheritSection NoPriorCVDRisk See Also
#'
#' @author
#' Billy Wu (R Developer) and Romana Pylypchuk (Principal Investigator)
#'
#' @references
#' New Zealand Ministry of Health: HISO 10071:2019 Cardiovascular Disease Risk Assessment Data Standard
#'
#' \href{https://www.health.govt.nz/publication/hiso-100712019-cardiovascular-disease-risk-assessment-data-standard}{HISO Document}
#'
#' @export
#' @examples
#' # As calculator (dataset not provided)
#' NoPriorCVDRisk_BMI(sex="F", age=65, eth="Indian", smoker=0, nzdep=5,  diabetes=0,
#'                    af=0, familyhx=1, lld=1, athrombi=1, bpl=1, sbp=118, tchdl=3.3, bmi=32)
#'
#' NoPriorCVDRisk_BMI(sex=F, age=55, eth=IN, exsmoker=Y, smoker=0, nzdep=5,  diabetes=T,
#'                    af=Y, familyhx=T, lld=1, athrombi=Y, bpl=T, sbp=120, tchdl=3.2, bmi=42)
#'
#' # As a vectoriser (dataset provided)
#' NoPriorCVDRisk_BMI(dat=DF, sex=sex, age=age, eth=ethnic_labels, smoker=smoking_status, nzdep=nzdep_quintiles,
#'                    diabetes=diab_status, af=af, familyhx=fam_hx, lld=lipidlowering, athrombi=antithrombics,
#'                    bpl=bplowering, sbp=systolic_bp, tchdl=tchdl_ratio, bmi=bmi)
#'
# --- Code ---
NoPriorCVDRisk_BMI <- function(dat, sex, age, eth, nzdep, exsmoker, smoker, diabetes, af, familyhx, sbp, tchdl, bmi, bpl, lld, athrombi,...){

  # Params
  demo.vars   <- c("sex", "age", "eth", "nzdep")
  smk.vars    <- c("exsmoker", "smoker")
  bin.vars    <- c("diabetes", "af", "familyhx", "lld", "athrombi", "bpl")
  num.vars    <- c("sbp", "tchdl")
  numNA.vars  <- c("bmi")

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
  vars <- c(demo.vars, bin.vars, smk.vars, num.vars, numNA.vars)

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

  bmi     <- list(bmilt185   = +(input$bmi < 18.5 & !is.na(input$bmi)),
                  bmi25_30   = +(input$bmi %in% 25:29.9 & !is.na(input$bmi)),
                  bmi30_35   = +(input$bmi %in% 30:34.9 & !is.na(input$bmi)),
                  bmi35_40   = +(input$bmi %in% 35:39.9 & !is.na(input$bmi)),
                  bmige40    = +(input$bmi >= 40 & !is.na(input$bmi)),
                  bmimiss    = +(input$bmi == "" | is.na(input$bmi)))


  values <- c(demo.vals, bin.vals, num.vals, bmi) # Order sensitive!

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
  values$age[f.ind] <- values$age[f.ind] - 56.05801
  values$age[m.ind] <- values$age[m.ind] - 51.59444
  # browser()
  values$nzdep[f.ind] <- values$nzdep[f.ind] - 2.994877
  values$nzdep[m.ind] <- values$nzdep[m.ind] - 2.975732

  values$sbp[f.ind] <- values$sbp[f.ind] - 128.6736
  values$sbp[m.ind] <- values$sbp[m.ind] - 128.8637

  values$tchdl[f.ind] <- values$tchdl[f.ind] - 3.715383
  values$tchdl[m.ind] <- values$tchdl[m.ind] - 4.385853

  # Interaction
  values$int_age_diab <- ifelse(values$diabetes == 0, 0, values$age)
  values$int_age_sbp  <- values$age * values$sbp
  values$int_sbp_bplt <- ifelse(values$bpl == 0, 0, values$sbp)

  # Coefficients
  fem.coeff <- list(age       = 0.0734393,
                    maori     = 0.4164622,
                    pacific   = 0.2268597,
                    indian    = 0.2086713,
                    asian     = -0.2680559,
                    ex_smoke  = 0.1444243,
                    cur_smoke = 0.6768396,
                    nzdep     = 0.0957229,
                    diabetes  = 0.4967444,
                    af        = 0.9293084,
                    familyhx  = 0.0645588,
                    lld       = -0.0568366,
                    athrombi  = 0.1393368,
                    bpl       = 0.3487781,
                    sbp       = 0.0176523,
                    tchdl     = 0.1361335,
                    bmilt185      = 0.6277962,
                    bmi25_30      = 0.0018215,
                    bmi30_35      = -0.0169324,
                    bmi35_40      = 0.0343351,
                    bmige40       = 0.3196519,
                    bmimiss       = 0.0213595,
                    int_age_diab = -0.0189779,
                    int_age_sbp  = -0.000471,
                    int_sbp_bplt = -0.0054002)

  male.coeff <- list(age       = 0.0669484,
                     maori     = 0.3166164,
                     pacific   = 0.2217931,
                     indian    = 0.3666816,
                     asian     = -0.4131973,
                     ex_smoke  = 0.0748648,
                     cur_smoke = 0.5317607,
                     nzdep     = 0.0631146,
                     diabetes  = 0.4107586,
                     af        = 0.6250334,
                     familyhx  = 0.1275721,
                     lld       = -0.0256429,
                     athrombi  = 0.0701999,
                     bpl       = 0.2847596,
                     sbp       = 0.0179827,
                     tchdl     = 0.1296756,
                     bmilt185      = 0.5488212,
                     bmi25_30      = -0.033177,
                     bmi30_35      = -0.0025986,
                     bmi35_40      = 0.1202739,
                     bmige40       = 0.3799261,
                     bmimiss       = -0.073928,
                     int_age_diab = -0.0124356,
                     int_age_sbp  = -0.0004931,
                     int_sbp_bplt = -0.0049226)

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
  estimate <- replace(estimate, f.ind, 1 - 0.9845026 ^ exp(sum.score[f.ind]))
  estimate <- replace(estimate, m.ind, 1 - 0.9712501 ^ exp(sum.score[m.ind]))

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
