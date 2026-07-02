# EDACS from cliot 1.0.0 (CRAN), edacs_chest_pain_score() cut out unmodified to source/cliot_edacs.R.
# Exhaustive grid: integer ages 18-110 x sex x (known CAD or >= 3 risk factors) x the four symptoms.
# Run from the repo root: Rscript reference/edacs/run_cliot.R
source("reference/edacs/source/cliot_edacs.R")
g <- expand.grid(age = 18:110, sex = c("female", "male"), cad_or_3rf = 0:1, diaphoresis = 0:1,
                 radiates = 0:1, pleuritic = 0:1, palpation = 0:1, stringsAsFactors = FALSE)
g$score <- mapply(function(a, s, c, d, r, i, p) edacs_chest_pain_score(a, s, c, d, r, i, p)$EDACS_Score,
                  g$age, g$sex, g$cad_or_3rf, g$diaphoresis, g$radiates, g$pleuritic, g$palpation)
write.csv(g, "reference/edacs/fixtures/cliot.csv", row.names = FALSE, quote = FALSE)
cat(nrow(g), "rows\n")
