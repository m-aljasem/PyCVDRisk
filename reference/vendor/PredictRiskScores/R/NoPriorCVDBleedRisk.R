
#' PREDICT Major Bleed (2018) Risk Score for People Without Prior CVD
#'
#' \code{NoPriorCVDBleedRisk} calculates the 5 year absolute risk of major bleeding (gastrointestinal, intracranial, and other bleeds), for people without a history of atherosclerotic CVD.
#'
#' @usage NoPriorCVDBleedRisk(dat, sex, age, eth, exsmoker, smoker, nzdep, diabetes,
#'                  familyhx, lld, bpl, cancer, gibleed, puddiag, alcohol, liver,
#'                  puddrug, nsaid, steroids, ssri, sbp, tchdl, ...)
#'
#' @inheritParams NoPriorCVDRisk
#' @param cancer  prior primary malignancy excluding squamous and basal cell skin cancers
#' @param gibleed prior gastrointestinal bleeding
#' @param puddiag prior non-bleeding and non-perforated peptic ulcer disease
#' @param alcohol chronic high use of alcohol
#' @param liver   chronic liver disease or pancreatitis
#' @param puddrug   receiving at lease one peptic ulcer disease medication
#' @param nsaid     receiving at least one nonsteroidal anti-inflammatary drug
#' @param steroids  receiving at least one corticosteroid medication
#' @param ssri      receiving at least one selective serotonine reuptake inhibitor
#'
#' @inherit NoPriorCVDRisk details
#'
#' @return
#' returns either a single 5-year major bleed risk estimate, or a numeric vector of risk estimates if \code{dat} is provided.
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
#' \item{diabetes,\cr familyhx}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#' \item{bpl, lld}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#'\item{cancer, gibleed,\cr puddiag, alcohol, liver\cr}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#' \item{puddrug, nsaid,\cr steroids, ssri}{label or encode as one of the following:
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
#' @inheritSection NoPriorCVDRisk See Also
#'
#' @author
#' Billy Wu (R Developer) and Vanessa Selak (Principal Investigator)
#'
#' @references
#' Selak V, Jackson R, Poppe K, et al. Predicting Bleeding Risk to Guide Aspirin Use for the Primary Prevention of Cardiovascular Disease: A Cohort Study. Ann Intern Med. 2019;170:357-368. doi: https://doi.org/10.7326/M18-2808
#'
#' Using the bleeding risk equation developed by Selak et al (2019), I developed a web-based risk calculator that provides clinicians with an individualised estimate of the CVD benefit and bleeding harms of aspirin for their patients without established CVD.
#'
#' \href{https://www.annals.org/aim/fullarticle/doi/10.7326/M18-2808}{Full Article}
#' \href{https://aspirinbenefitharmcalculator.shinyapps.io/calculator/}{Online Calculator}
#'
#' @export
#' @examples
#' # As calculator (dataset not provided)
#' NoPriorCVDBleedRisk(sex=0, age=55, eth=21, exsmoker=0, smoker=0, nzdep=5, diabetes=1,
#'                     familyhx=1, lld=1, bpl=1, cancer=1, gibleed=1, puddiag=1, alcohol=0,
#'                     liver=0, puddrug=0, nsaid=1, steroids=1, ssri=0, sbp=130, tchdl=5)
#'
#' NoPriorCVDBleedRisk(sex=0, age=76, eth=21, exsmoker=0, smoker=N, nzdep=5, diabetes=1,
#'                     familyhx=1, lld=1, bpl=1, cancer=1, gibleed=1, puddiag=1, alcohol=0,
#'                     liver=0, puddrug=0, nsaid=1, steroids=NA, ssri=0, sbp=130, tchdl=5,
#'                     dp = 5, allow.age = F, allow.na = FALSE)
#'
#' # As a vectoriser (dataset provided)
#' NoPriorCVDBleedRisk(dat=DT, sex=sex, age=index_age, eth=eth_vars, exsmoker=exsmoke,
#'                     smoker=smoking_current, nzdep=nzdep, familyhx=family_hx, diabetes=dm,
#'                     sbp=index_sbp, tchdl=tchdl, lld=lld, bpl=bpl, cancer=hx_cancer,
#'                     gibleed=hx_gibleed, puddiag=hx_hud, alcohol=alc, liver=hx_liver,
#'                     puddrug=pudmx, nsaid=nsaid, steroids=steroidmx, ssri=ssri)
#'
# --- Code ---
NoPriorCVDBleedRisk <- function(dat, sex, age, eth, exsmoker, smoker, nzdep, diabetes, familyhx, lld, bpl, cancer, gibleed, puddiag,
                                alcohol, liver, puddrug, nsaid, steroids, ssri, sbp, tchdl, ...){

  # Params
  demo.vars   <- c("sex", "age", "eth", "nzdep")
  smk.vars    <- c("exsmoker", "smoker")
  bin.vars    <- c("diabetes", "familyhx", "lld", "bpl", "cancer", "gibleed", "puddiag", "alcohol", "liver", "puddrug", "nsaid", "steroids", "ssri")
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

  # Coefficients
  fem.coeff <- list(age      = 0.03502806,
                    maori    = 0.311582316,
                    pacific  = 0.291826502,
                    indian   = -0.170178670,
                    asian    = 0.044890076,
                    ex_smoke  = 0.144844011,
                    cur_smoke = 0.495240401,
                    nzdep    = 0.098736992,
                    diabetes = 0.182633821,
                    familyhx = 0.055185249,
                    lld     = 0.010545182,
                    bpl     = 0.140874933,
                    cancer  = 0.299418027,
                    gibleed = 1.157219678,
                    puddiag = 0.426358755,
                    alcohol = 0.95065995,
                    liver   = 0.979437007,
                    puddrug = 0.370528961,
                    nsaid    = 0.10655804,
                    steroids = 0.328347624,
                    ssri     = 0.16495507,
                    sbp      = 0.004991576,
                    tchdl    = 0.001878851)

  male.coeff <- list(age      = 0.03538036,
                     maori    = 0.40955001,
                     pacific  = 0.52687151,
                     indian   = -0.01815411,
                     asian    = 0.37798865,
                     exsmoke  = 0.15536803,
                     cursmoke = 0.38226181,
                     nzdep    = 0.09305327,
                     diabetes = 0.17500777,
                     familyhx = 0.05028066,
                     lld     = -0.04636764,
                     bpl     = 0.20741834,
                     cancer  = 0.56636099,
                     gibleed = 1.14121551,
                     puddiag = 0.22227113,
                     alcohol = 0.67405759,
                     liver   = 0.77588662,
                     puddrug = 0.36282612,
                     nsaid    = 0.17279428,
                     steroids = 0.35261644,
                     ssri     = 0.2928215,
                     sbp      = 0.00373758,
                     tchdl    = -0.05009861)

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
  estimate <- replace(estimate, f.ind, 1 - 0.98902929 ^ exp(sum.score[f.ind] - 3.262378))
  estimate <- replace(estimate, m.ind, 1 - 0.98861720 ^ exp(sum.score[m.ind] - 2.787439))

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


