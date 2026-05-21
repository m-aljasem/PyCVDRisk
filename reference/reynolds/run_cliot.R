# Compare the CRAN package cliot 1.0.0 (reynolds_risk_score_women) with this package on the seeded women of
# fixtures/cook_sas_seeded.csv. Run from repo root: Rscript reference/reynolds/run_cliot.R
# The function is cut out of source/third_party/cliot_1.0.0/cliot/R/neelpackage.R and run unchanged, and once
# with its ln(SBP) coefficient 3.193 replaced by the printed 3.137.
src <- readLines("reference/reynolds/source/third_party/cliot_1.0.0/cliot/R/neelpackage.R")
i <- grep("^reynolds_risk_score_women <- function", src)
j <- i + which(src[(i + 1):length(src)] == "}")[1]
body <- src[i:j]
eval(parse(text = body))
f_orig <- reynolds_risk_score_women
eval(parse(text = sub("3.193 \\* log\\(systolic_bp_mmhg\\)", "3.137 * log(systolic_bp_mmhg)", body)))
f_fix <- reynolds_risk_score_women
stopifnot(!identical(deparse(f_orig), deparse(f_fix)))
p <- read.csv("reference/reynolds/fixtures/cook_sas_seeded.csv")
w <- p[p$sex == "female", ]
a1c <- ifelse(w$dm == 1, w$a1c, 0)  # paper: HbA1c term only if diabetic
run <- function(f) mapply(function(...) f(...)$Reynolds_Risk_Score, w$age, w$sbp, w$tc, w$hdl, w$crp, a1c, w$smoker, w$fh)
o <- run(f_orig); x <- run(f_fix)
cat(sprintf("cliot as published: %d/%d equal to ours at 1 decimal; cliot/ours - 1 from %.3f to %.3f\n",
            sum(o == round(w$ours, 1)), nrow(w), min(o / w$ours - 1), max(o / w$ours - 1)))
cat(sprintf("cliot with 3.137: %d/%d equal to ours at 1 decimal; max |diff| %.3f (cliot rounds to 1 decimal)\n",
            sum(x == round(w$ours, 1)), nrow(w), max(abs(x - w$ours))))
t6 <- sapply(c(0.1, 0.5, 1, 3, 5, 8, 10, 20), function(c) sapply(0:1, function(h) f_orig(50, 155, 240, 35, c, 0, 1, h)$Reynolds_Risk_Score))
cat("cliot Table 6 (as published):", t6, "\n")
