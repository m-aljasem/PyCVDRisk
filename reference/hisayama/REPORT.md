# Hisayama ASCVD risk model (2022): verification report

Evidence level C + B: the paper's equation, point derivation and constants; the point score also matches the R
package Jcvrisk on every patient, and a web calculator of the JAS 2022 chart on 79/80 cells.

## Sources

- **Paper**: Honda T, Chen S, Hata J, Yoshida D, Hirakawa Y, Furuta Y, Shibata M, Sakata S, Kitazono T, Ninomiya T.
  Development and validation of a risk prediction model for atherosclerotic cardiovascular disease in Japanese
  adults: the Hisayama Study. J Atheroscler Thromb 2022;29(3):345-61. doi:10.5551/jat.61960 (`source/honda2022.pdf`,
  with supplement; sha256 4211287b...8344). Table 2: coefficients (3 decimals); the equation with S10 = 0.9696
  (competing-risk adjusted, Supplemental Table 1) and constant 6.7963; Fig. 1: point chart and risk table;
  Supplemental Table 3: how the points were derived; Supplemental Table 4: the non-HDL-C model (constant 6.9895).
- **R package Jcvrisk 0.1.3** (CRAN 2025-10-29, Okumiyama H, Fujii R; MIT): `hisayama()` implements the point score
  and its formula. Vendored unmodified in `reference/vendor/Jcvrisk` (matches the package's MD5 list).
- **Web calculator**: ecg.watson.jp/medical/ASCVD ("ASCVD Risk Calculator for Japanese patients (Hisayama score)"; `source/ecg_watson_ascvd.html`, sha256 d4246e41...; printed copy `source/ecg_watson_calculator_page.pdf`).
  It implements the JAS 2022 guideline chart: no proteinuria or exercise items, impaired glucose tolerance 1 point
  in place of diabetes 3, ages 40-79.

## Model

`hisayama`: p = 1 - 0.9696^exp(sum(beta x) - 6.7963), age, sex, SBP, diabetes, HDL-C, LDL-C (or non-HDL-C, the
alternative model), proteinuria, current smoking, no regular exercise. `hisayama_points` + `hisayama_points_risk`:
Fig. 1 points (0-27 without age) and p = 1 - 0.9696^exp((score + age points) x 0.144 - 2.4767), age points 0, 5,
11, 16, 20 by decade (80-84).

## Checks (`tests/hisayama/test_hisayama.py`)

- Points = weight / 0.144, rounded, for all 18 rows of Supplemental Table 3.
- 2.4767 = 6.7963 - 4.3196, and 4.3196 is sum(beta x reference value) from that table's weights (within 0.002).
- Both equation constants are within 0.05 of sum(beta x mean) from Table 1 (3-decimal betas).
- Jcvrisk `hisayama()` on 2,702 seeded patients aged 40-84: equal to 1e-12.
- ecg.watson.jp: its 20 x 4 table equals the formula at score + 2 (the page has no exercise item, so every patient
  carries the 2 points for no regular exercise), shown to 1 decimal with a floor of 1.0, in 79/80 cells.

## Findings

1. **Fig. 1's risk table does not follow the paper's own formula.** Its values form one sequence in score + age
   points, as the formula implies, but they rise faster (on the log(-log) scale about 0.155 per point, not 0.144):
   15.7% against 11.9% at score 27, ages 40-49; 49.7% against 37.0% at score 16, ages 80-84. The package follows
   the stated formula, as do Jcvrisk and the JAS 2022 chart (via the web calculator); Fig. 1's table is not used.
2. **Web calculator, one cell**: score 19, ages 70-79 shows 41.1%; the formula gives 41.4%.
3. **The JAS 2022 chart is a different score.** It drops proteinuria and exercise and scores glucose intolerance;
   its primary source (the JAS 2022 guideline) is not in hand, so it is not implemented.
