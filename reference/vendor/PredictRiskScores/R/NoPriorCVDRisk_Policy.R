#' VARIANZ CVD (2018) Policy-level Risk Score for People Without CVD
#'
#' \code{NoPriorCVDRisk_Policy} calculates the 5 year absolute risk of cardiovascular disease (CVD) for people without a history of atherosclerotic CVD that
#' is intended for use at policy or general population level. The outcome of interest is the 5-year risk of a non-fatal or fatal CVD event, including hospitalisation
#' for coronary heart disease, stroke or other cerebrovascular disease (including transient ischaemic attack), peripheral vascular disease and heart failure,
#' or cardiovascular death.
#'
#' @usage NoPriorCVDRisk_Policy(dat, sex, age, eth, nzdep, diabetes, af,
#'                       bpl, lld, athrombi, ...)
#'
#' @inheritParams NoPriorCVDRisk
#'
#' @details  \code{NoPriorCVDRisk_Policy} is intended to be used at the policy or general population level. As such, a dataset containing a population should be provided, and outputs should be summarised.
#' Using this function as a calculator for an individual is not recommended. When a dataset is supplied, a risk score is produced for each row of data, resulting in a numeric vector of the same length.
#' Each argument requires the variable name from the dataset \code{dat} that corresponds with the parameter. \cr
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
#'              \item Indian, Fijian Indian, IN, I, or 43
#'              \item Other, Middle Eastern, African, Latin American, South American, Latin, Asian, Other Asian, SE Asian, East Asian,
#'                    Chinese, MELAA, ME, ASN, A, 4, 40, 41, 42, 44, 5, 51, 52, 53, 54, 61
#'              \item note: Other Asian includes non-Indian South Asian
#'              }}
#' \item{nzdep}{numeric value between 1 and 5}
#' \item{diabetes,\cr af}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#' \item{bpl, lld,\cr athrombi}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
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
#' Billy Wu (R Developer) and Suneela Mehta (Principal Investigator)
#'
#' @references
#' Mehta, S., Jackson, R., Pylypchuk, R., Poppe, K., Wells, S., & Kerr, A. J. (2018). Development and validation of alternative cardiovascular risk prediction equations for population health planning: a routine health data linkage study of 1.7 million New Zealanders. International journal of epidemiology, 47 (5), 1571-1584.
#'
#' \href{https://academic.oup.com/ije/article/47/5/1571/5053287}{Full Article}
#'
#' @export
#' @examples
#' # As a vectoriser (dataset provided)
#' NoPriorCVDRisk_Policy(dat=TEST, sex=sex, age=age, eth=eth, nzdep=nzdep, diabetes=diabetes, af=af, lld=lld,
#'                       athrombi=athromb, bpl=bpl, dp =6, allow.na = FALSE)
#'
# --- Code ---
NoPriorCVDRisk_Policy <- function(dat, sex, age, eth, nzdep, diabetes, af, bpl, lld, athrombi, ...){

  # Params
  demo.vars   <- c("sex", "age", "eth", "nzdep")
  bin.vars    <- c("diabetes", "af", "lld", "athrombi", "bpl")

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
  vars <- c(demo.vars, bin.vars)

  ParamCheck(input, vars, call, is.table, allow.age, allow.na)

  # Values
  f.ind <- which(tolower(input$sex) %in% ok.female)
  m.ind <- which(tolower(input$sex) %in% ok.male)

  demo.vals <- list(age      = input$age,
                    maori    = +(tolower(input$eth) %in% ok.maori),
                    pacific  = +(tolower(input$eth) %in% ok.pi),
                    indian   = +(tolower(input$eth) %in% ok.indian),
                    other    = +(tolower(input$eth) %in% c(ok.asian, ok.other)),
                    nzdep    = input$nzdep)

  bin.vals <- sapply(bin.vars,
                     function(x){
                       +(tolower(input[[x]]) %in% ok.true)
                     },
                     USE.NAMES = TRUE,
                     simplify = FALSE)

  values <- c(demo.vals, bin.vals) # Order sensitive!

  # Adjustments
  if(allow.age){
    values$age[which(values$age < 30)] <- 30
    values$age[which(values$age > 79)] <- 80
  }

  if(!allow.na){

    vars <- bin.vars

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

  # Recentering
  values$age[f.ind] <- values$age[f.ind] - 48.04908
  values$age[m.ind] <- values$age[m.ind] - 48.77995

  values$nzdep[f.ind] <- values$nzdep[f.ind] - 3.031210
  values$nzdep[m.ind] <- values$nzdep[m.ind] - 3.045908

  # Interaction
  values$age_x_bpl    <- values$age * values$bpl
  values$age_x_diab   <- values$age * values$diabetes
  values$age_x_af     <- values$age * values$af

  values$bpl_x_diab     <- values$bpl * values$diabetes
  values$athromb_x_diab <- values$athrombi * values$diabetes
  values$bpl_x_af       <- values$bpl * values$af

  # Coefficients
  fem.coeff <- list(age      = 0.0849801,
                    maori    = 0.6491723,
                    pacific  = 0.3048516,
                    indian   = 0.0387651,
                    other    = -0.5629498,
                    nzdep    = 0.1138991,
                    diabetes = 1.143758,
                    af       = 1.156071,
                    lld      = 0.0208865,
                    athrombi = 0.3933442,
                    bpl      = 0.7829573,
                    age_x_bpl      = -0.0219648,
                    age_x_diab     = -0.0203055,
                    age_x_af       = -0.0163526,
                    bpl_x_diab     = -0.3619771,
                    athromb_x_diab = -0.2727397,
                    bpl_x_af       = -0.2386059)

  male.coeff <- list(age       = 0.0785617,
                     maori     = 0.424878,
                     pacific   = 0.1835682,
                     indian    = 0.2163051,
                     other     = -0.9524949,
                     nzdep     = 0.0916424,
                     diabetes  = 0.6824352,
                     af        = 0.7176393,
                     lld       = 0.0670676,
                     athrombi  = 0.2934377,
                     bpl       = 0.6228715,
                     age_x_bpl = -0.023287,
                     age_x_diab     = -0.0159681,
                     age_x_af       = -0.0136306,
                     bpl_x_diab     = -0.1406977,
                     athromb_x_diab = -0.2456125,
                     bpl_x_af       = -0.1730356)

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
  estimate <- replace(estimate, f.ind, 1 - 0.9888268899721 ^ exp(sum.score[f.ind]))
  estimate <- replace(estimate, m.ind, 1 - 0.9715879364233 ^ exp(sum.score[m.ind]))

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
