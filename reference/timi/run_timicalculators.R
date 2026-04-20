# TIMI UA/NSTEMI reference answers from the TIMI Study Group's own R package TIMIcalculators (companion to the
# calculator app; github.com/gmelloni/TIMIcalculators @ ae85afb, archived in source/TIMIcalculators).
# Only computeUANSTEMI() is cut out of R/timi_scores.R and evaluated; nothing else in the file is run.
# Run from the repo root: Rscript reference/timi/run_timicalculators.R
src <- parse("reference/timi/source/TIMIcalculators/R/timi_scores.R", keep.source = FALSE)
fn <- Filter(function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
               identical(e[[2]], as.name("computeUANSTEMI")), as.list(src))
stopifnot(length(fn) == 1)
eval(fn[[1]])
yn <- function(v) ifelse(v == 1, "Yes", "No")
g <- expand.grid(age65 = 0:1, cadrisk = 0:1, knowncad = 0:1, asa = 0:1, angina = 0:1, stchange = 0:1, cardiac = 0:1)
out <- mapply(function(a, r, k, s, an, st, c)
  computeUANSTEMI(age = ifelse(a == 1, ">=65", "<65"), cadrisk = yn(r), knowncad = yn(k), asa = yn(s),
                  angina = yn(an), stchange = yn(st), cardiac = yn(c)),
  g$age65, g$cadrisk, g$knowncad, g$asa, g$angina, g$stchange, g$cardiac)
write.csv(data.frame(combo = seq_len(nrow(g)) - 1, g, displayed = out),
          "reference/timi/fixtures/package_reference.csv", row.names = FALSE)
cat(nrow(g), "rows\n")
