#' General Parameter Checking for PredictRiskScores
#'
#' \code{ParamCheck} evaluates input parameters inherited from the parent.frame; to check for names, values, and validity and ensure that a risk calculation can be carried out.
#' The call is stopped completely when required arguments are not stated or when compulsary inputs are invalid.
#' Invalid values may still be calculated an \code{NA} returned with an error message indicating one or more problematic variables.
#'
#' @usage ParamCheck(input, vars, call, is.table, allow.age, allow.na)
#'
#' @param input list of all input values inherited from the function call
#' @param vars character vector of required parameters
#' @param call name of the function being called
#' @param is.table logical; whether a dataset is provided in the parent.frame
#' @param allow.age logical. Whether or not age range is extended outside of 30 - 74; default is TRUE. If set to FALSE, then \code{NA} is returned as risk estimate.
#' @param allow.na logical. Whether or not missing values for binary variables and smoking status are treated as 0; default is TRUE. If set to FALSE, then \code{NA} is returned as risk estimate.
#' @details
#' \code{ParamCheck} identifies the following errors and invalid inputs:
#' \enumerate{
#'  \item missing arguments - operations stops
#'  \item missing input data - operation stops
#'  \item invalid sex, age, ethnicity, nzdep, smoking inputs
#'  \item invalid binary inputs
#'  \item invalid numeric inputs
#'  \item invalid level inputs
#'  }
#' @return
#' \code{ParamCheck} returns warning messages if any invalid values are detected. In cases where input data is invalid, \code{ParamCheck} will
#' stop the function's operation and return an error message.
#' @author
#' Billy Wu (R Developer)
#'
ParamCheck <- function(input, vars, call, is.table, allow.age, allow.na){

  # Set vars by call type
  demo.vars <- get("demo.vars", parent.frame())

  if(call == "NoPriorCVDRisk_Policy"){
    num.vars   <- character(0)
    smk.vars   <- character(0)

    if(!is.table){
      warning(call. = FALSE, paste0("Using this function as a calculator for an individual is not recommended! See ?",  call))
    }

  } else {
    num.vars   <- get("num.vars", parent.frame())
    smk.vars   <- get("smk.vars", parent.frame())
  }

  numNA.vars <- if(call %in% c("NoPriorCVDRisk_Policy", "NoPriorCVDBleedRisk", "PriorT2DRisk", "NoPriorCVDRisk")){
    character(0)
  } else {
    get("numNA.vars", parent.frame())
  }

  lvl.vars <- if(call == "PostACSRisk"){
    get("lvl.vars", parent.frame())
  }

  # 1.  Missing argument check
  if(is.table){
    input <- get("input", parent.frame())
  }

  if(!all(vars %in% names(input))) {
    stop(call. = F,
         paste("Missing parameter(s):",
               paste(setdiff(vars, names(input)),
                     collapse = ", ")))
  }

  # 2. Missing input check
  if(is.table){  # Dataset provided (but missing correct input columns)

    dat       <- as.data.frame(get("dat", parent.frame()), row.names = NULL)
    input     <- input[!names(input) %in% c("dat", "dp", "allow.age", "allow.na")]
    colnames  <- as.vector(sapply(input, as.character))

    is.missing <- any(!colnames %in% names(dat))

    if(is.missing){
      to.check <- colnames[!input %in% names(dat)]
      stop(call. = F,
           paste("Check input(s) names:",
                 paste(sQuote(to.check),
                       collapse = ", ")))
    }

    input[]  <- dat[, colnames]

  } else { # One-off calculator (but missing correct input values)

    to.check <- names(input)[!sapply(input, class) %in% c("character", "numeric", "name", "logical")]

    if(length(to.check) > 0){
      stop(call. = F,
           paste("multiple input values detected for",
                 paste(sQuote(to.check),
                       collapse = ", ")))
    }

  }

  # 3. Compulsary input check
  vars <- c(demo.vars, num.vars)

  has.na <- sapply(vars,
                   function(x){
                     if(class(input[[x]]) == "name"){
                       input[[x]] <- as.character(input[[x]])
                     }
                     any(is.na(input[[x]]) | input[[x]] == "NA")
                   }, USE.NAMES = TRUE)

  if(any(has.na)){
    stop(call. = F,
         paste(paste(sQuote(names(which(has.na))),
                     collapse = ", "),
               "cannot contain NA! For acceptable values,", paste0("see ?", call)))
  }

  # If all clear, put data to parent.frame
  assign("input", input, parent.frame())

  # Accepted Values
  ok.ages <- if(allow.age){
    as.numeric(18:110)
  } else {
    as.numeric(30:74)
  }

  na <- if(allow.na){
    c("na", "NA", NA)
  } else {
    character(0)
  }

  ok.female <- tolower(c("f", "female", 0))
  ok.male   <- tolower(c("m", "male", 1))
  ok.nzdeps <- as.numeric(1:5)
  ok.nzeo   <- tolower(c("NZ European", "European", "NZEO", "Euro", "E", "1", "10", "11", "12"))
  ok.maori  <- tolower(c("Maori", "NZMaori", "NZ Maori", "M", "2", "21"))
  ok.pi     <- tolower(c("Pacific", "Pacific Islander", "PI", "P", "3", "30", "31", "32", "33", "34", "35", "36", "37"))
  ok.asian  <- tolower(c("Asian", "Other Asian", "SE Asian", "East Asian", "Chinese", "ASN", "A", "4", "40", "41", "42", "44"))
  ok.indian <- tolower(c("Indian", "Fijian Indian", "IN", "I", "43"))
  ok.other  <- tolower(c("Other", "Middle Eastern", "African", "Latin American", "South American", "Latin", "MELAA", "ME", "5", "51", "52", "53", "54", "61"))
  ok.smoker <- tolower(c("Y", "Yes", "Smoker", "Current", "S", 1, "T", TRUE))
  ok.exsmkr <- tolower(c("Y", "Yes", "Ex", "Ex-smoker", "Exsmoker", "E", 1, "T", TRUE))
  ok.nonsmk <- tolower(c("N", "No", "Non", "Non-smoker", "N", 0, "F", FALSE))
  ok.true   <- tolower(c("Y", "Yes", 1, "T", TRUE))
  ok.false  <- tolower(c("N", "No", 0, "F", FALSE))
  ok.stemi  <- tolower(c("STEMI", "ST-Elevation", "S", "2"))
  ok.nstemi <- tolower(c("NSTEMI", "NONSTEMI", "NON-STEMI", "N", "1"))

  # 4.  Input check
  lapply(ls(pattern = "ok."),
         function(x)
           assign(x, get(x), envir = parent.frame(3L))
  )

  # Demographic
  lapply(c(demo.vars, smk.vars),

         function(x){

           if(x == "sex"){
             invalid <- which(!tolower(input[[x]]) %in% c(ok.female, ok.male))
           }

           if(x == "age"){
             invalid <- which(!as.numeric(input[[x]]) %in% ok.ages | is.na(input$age))
           }

           if(x == "eth"){
             invalid <- if(call == "NoPriorCVDRisk_Policy"){
               which(!tolower(input[[x]]) %in% c(ok.nzeo, ok.maori, ok.pi, ok.indian, ok.asian, ok.other))
             } else {
               which(!tolower(input[[x]]) %in% c(ok.nzeo, ok.maori, ok.pi, ok.indian, ok.asian))
             }
           }

           if(x == "nzdep"){
             invalid <- which(!as.numeric(input[[x]]) %in% ok.nzdeps)
           }

           if(x == "exsmoker"){
             invalid <- which(!tolower(input[[x]]) %in% c(ok.exsmkr, ok.nonsmk, na))
           }

           if(x == "smoker"){
             invalid <- which(!tolower(input[[x]]) %in% c(ok.smoker, ok.nonsmk, na))
           }

           assign(paste("inval", x, sep = "."),
                  invalid,
                  envir = parent.frame(2L))

         })

  # Binary vars
  lapply(get("bin.vars", parent.frame()),
         function(x){

           invalid <- which(!tolower(input[[x]]) %in% c(ok.true, ok.false, na))

           assign(paste("inval", x, sep = "."),
                  invalid,
                  envir = parent.frame(2L))

         })

  if(call != "NoPriorCVDRisk_Policy"){

    lapply(c(num.vars, numNA.vars),
           function(x){

             invalid <- which(!class(input[[x]]) %in% c("numeric", "integer", "logical"))

             assign(paste("inval", x, sep = "."),
                    invalid,
                    envir = parent.frame(2L))

           })
  }

  if(call == "PostACSRisk"){
    lapply(lvl.vars,
           function(x){

             if(x == "acstype"){
               invalid <- which(!tolower(input[[x]]) %in% c(ok.stemi, ok.nstemi, c(na, "0")))
             }

             assign(paste("inval", x, sep = "."),
                    invalid,
                    envir = parent.frame(2L))

           })
  }

  # warning message
  inval.vars <- ls(pattern = "inval.")
  inval.vars <- inval.vars[sapply(inval.vars,
                                  function(x){
                                    length(get(x)) >= 1
                                  }, USE.NAMES = T)]

  if(length(inval.vars) >= 1){
    warning(call. = FALSE,
            paste0(paste(sQuote(gsub("inval.", "", inval.vars)),
                         collapse = ", "),
                   " contain(s) one or more un-calculatable value(s). See R documentation using ", "?", call))
  }

  # Invalid params return NA score
  lapply(inval.vars,
         function(x)
           assign(x, get(x), envir = parent.frame(3L))
  )

}



