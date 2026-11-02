
#' PREDICT CVD Type-II Diabetes (2018.1) Risk Score for People Without Prior CVD
#'
#' \code{PriorT2DRisk} calculates the 5 year risk of cardiovascular disease (CVD) (hospitalisation for acute coronary syndrome, heart failure, stroke or other cerebrovascular disease, peripheral vascular death, cardiovascular death),
#' for people with diabetes. This equation takes into account multiple diabetes-related variables. The outcome of interest is the 5-year risk of a non-fatal or fatal CVD event, including hospitalisation
#' for coronary heart disease, stroke or other cerebrovascular disease (including transient ischaemic attack), peripheral vascular disease and heart failure,
#' or cardiovascular death.
#'
#' @usage PriorT2DRisk(sex, age, eth, nzdep, smoker, af, familyhx,
#'              lld, athrombi, bpl, oral, insulin, sbp, tchdl, bmi,
#'              egfr, acr, hba1c, years, ...)
#'
#' @inheritParams NoPriorCVDRisk_BMI
#' @param years years since diagnosis of type 2 diabetes
#' @param egfr  most recent calculated value of eGFRvalue in mL/min/1.73m2
#' @param acr   most recent value of ACR value in mg/mmol
#' @param hba1c most recent value of HbA1c in mmol/mol
#' @param oral    receiving oral hypoglycaemic medication
#' @param insulin receiving insulin treatment
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
#' \item{smoker}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, Smoker, Current, S, 1, T, TRUE
#'              \item N, No, Non-smoker, 0, F, FALSE
#'              }}
#' \item{af, familyhx}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#' \item{bpl, lld,\cr athrombi}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#' \item{oral, insulin}{label or encode as one of the following:
#'            \itemize{
#'              \item Y, Yes, 1, T, TRUE
#'              \item N, No, 0, F, FALSE
#'              }}
#' \item{bmi, sbp, tchdl,\cr egfr, acr, hba1c}{numeric value of measured result. Note:
#'            \itemize{
#'              \item all values must be available
#'              }}
#' \item{years}{numeric value of number of years since T2D diagnosis}
#' \item{...}{further arguments:
#'            \itemize{
#'              \item \code{dp} numeric value to set decimal place; default is 4
#'              \item \code{allow.age} logical. Whether or not age range is extended outside of 30 - 74; default is TRUE. If set to FALSE, then \code{NA} is returned as risk estimate.
#'              \item \code{allow.na} logical. Whether or not missing values for binary variables and smoking status are treated as 0; default is TRUE. If set to FALSE, then \code{NA} is returned as risk estimate.
#'              }}
#'
#' @inheritSection PostACSRisk See Also
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
#' # As calculator (Dataset not provided)
#' PriorT2DRisk(sex="M", age=35, eth=2, nzdep=5, smoker=1, af=1, familyhx=1, lld=0,
#'              athrombi=0, bpl=0, oral=0, insulin=0, sbp=120, tchdl=3.3, bmi=27,
#'              egfr=78, acr=1, hba1c=48, years=1)
#'
#' PriorT2DRisk(sex=0, age=75, eth=PI, nzdep=3, smoker=0, af=F, familyhx=0, lld=Y,
#'              athrombi=0, bpl=F, oral=T, insulin=0, sbp=130, tchdl=4, bmi=31,
#'              egfr=92, acr=1.4, hba1c=56, years=3)
#'
#' # As a vectoriser (Dataset provided)
#' PriorT2DRisk(dat=DF, sex=sex, age=age, eth=eth, nzdep=nzdep, smoker=smoker,
#'              af=af, familyhx=familyhx, sbp=sbp, tchdl=tchdl, bmi=bmi, years=years,
#'              egfr=egfr, acr=acr, hba1c=hba1c, oral=oral, insulin=insulin, bpl=bpl,
#'              lld=lld, athrombi=athrombi)
#'
# --- Code ---
PriorT2DRisk <- function(dat, sex, age, eth, nzdep, smoker, af, familyhx, lld, athrombi, bpl, oral, insulin, sbp, tchdl, bmi, egfr, acr, hba1c, years, ...){

  # Params
  demo.vars   <- c("sex", "age", "eth", "nzdep")
  smk.vars    <- c("smoker")
  bin.vars    <- c("af", "familyhx", "lld", "athrombi", "bpl", "oral", "insulin")
  num.vars    <- c("sbp", "tchdl", "bmi", "egfr", "acr", "hba1c", "years")

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

  # Recentering
  values$age[f.ind] <- values$age[f.ind] - 53.598009
  values$age[m.ind] <- values$age[m.ind] - 53.738152

  values$nzdep[f.ind] <- values$nzdep[f.ind] - 3.657006
  values$nzdep[m.ind] <- values$nzdep[m.ind] - 3.410281

  values$sbp[f.ind] <- values$sbp[f.ind] - 131.380365
  values$sbp[m.ind] <- values$sbp[m.ind] - 131.662168

  values$tchdl[f.ind] <- values$tchdl[f.ind] - 3.970698
  values$tchdl[m.ind] <- values$tchdl[m.ind] - 4.330372

  values$bmi[f.ind] <- values$bmi[f.ind] - 33.515572
  values$bmi[m.ind] <- values$bmi[m.ind] - 31.338254

  values$egfr[f.ind] <- values$egfr[f.ind] - 89.558866
  values$egfr[m.ind] <- values$egfr[m.ind] - 88.788314

  values$acr[f.ind] <- log((acr + 0.0099999997764826) / 1000) + 4.314302355
  values$acr[m.ind] <- log((acr + 0.0099999997764826) / 1000) + 4.275179000

  values$hba1c[f.ind] <- values$hba1c[f.ind] - 63.618622
  values$hba1c[m.ind] <- values$hba1c[m.ind] - 63.889441

  values$years[f.ind] <- values$years[f.ind] - 5.406364
  values$years[m.ind] <- values$years[m.ind] - 5.183025

  # Coefficients
  fem.coeff <- list(age       = 0.0424465,
                    maori     = 0.0770441,
                    pacific   = -0.253300,
                    indian    = 0.138371,
                    asian     = -0.3611259,
                    smoker    = 0.4391752,
                    nzdep     = 0.0699105,
                    af        = 0.7864886,
                    familyhx  = 0.1063846,
                    lld       = -0.1595083,
                    athrombi  = 0.0605766,
                    bpl       = 0.0988141,
                    oral      = 0.1248604,
                    insulin   = 0.3535548,
                    sbp       = 0.0127053,
                    tchdl     = 0.1139678,
                    bmi       = 0.0073966,
                    egfr      = -0.0090784,
                    acr       = 0.1842885,
                    hba1c     = 0.0076733,
                    years     = 0.0163962
                    )

  male.coeff <- list(age       = 0.0472422,
                     maori     = -0.0553093,
                     pacific   = -0.210811,
                     indian    = 0.1522338,
                     asian     = -0.3852922,
                     smoker    = 0.3509447,
                     nzdep     = 0.0413719,
                     af        = 0.5284553,
                     familyhx  = 0.2093793,
                     lld       = -0.0344494,
                     athrombi  = 0.0474684,
                     bpl       = 0.1532122,
                     oral      = 0.0051476,
                     insulin   = 0.1846547,
                     sbp       = 0.0054797,
                     tchdl     = 0.0805627,
                     bmi       = 0.0117137,
                     egfr      = -0.0025889,
                     acr       = 0.1815067,
                     hba1c     = 0.0074805,
                     years     = 0.0162351
                     )

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
  estimate <- replace(estimate, f.ind, 1 - 0.9455710 ^ exp(sum.score[f.ind]))
  estimate <- replace(estimate, m.ind, 1 - 0.9121175 ^ exp(sum.score[m.ind]))

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



