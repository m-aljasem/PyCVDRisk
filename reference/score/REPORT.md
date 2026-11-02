# SCORE (2003) — verification report

Generated 2026-11-01 by `reference/score/report.py`.

## 1. Source

Conroy et al., Eur Heart J 2003, Appendix A (Tables A and B), saved in `source/`.
`check_coefficients.py` parses both tables from the PDF: all 22 values match.

## 2. The official SCORE charts (evidence level A)

The two total-cholesterol charts (Fig 1 high-risk, Fig 2 low-risk regions; 800 cells)
were transcribed by eye from the PDF at 400-800 dpi (`fixtures/charts_transcribed.txt`).
The paper prints alpha to one decimal and the betas to two or three, so the published
function cannot reproduce the charts exactly: a change of 0.05 in alpha alone moves risk by
about 5%. Each cell is therefore checked against the interval of risks the printed
coefficients allow (each moved by up to half a unit of its last digit).

- Printed coefficients, rounded to whole %: 75.6% of cells exact,
  98.6% within 1 point. The misses are all high-risk cells (men,
  smokers, cholesterol 7-8) where the chart is lower, as expected from coefficient rounding.
- **798/800 cells lie within the printed-precision interval** (median width
  0.50 points). The two outside it are chart anomalies, not transcription errors:
  - High risk, man, non-smoker, 65, SBP 160, cholesterol 8: printed **16**, shaded as >=15%,
    in a row reading 9 11 13 15 16; the function gives 17.4-21.0. Almost certainly a typo for 18/19.
  - High risk, woman, non-smoker, 50, SBP 120, cholesterol 6: printed 1; the function gives
    0.40-0.46 for every admissible coefficient value.
- Sensitivity: moving any one of the 22 printed coefficients by one unit of its last digit
  pushes at least one other chart cell out of its interval in 14/44 cases.

## 3. Reference implementations

No working third-party implementation was found. The only one on CRAN, `calculateScoreEur`
in PBImisc 1.0, returns NA for every input (it uses `coefs["w"]`, which does not exist,
instead of the weighted sum `w`), and always adds the smoking coefficient regardless of
`currentSmoker`. v1 of this package used coefficients that appear nowhere in the paper.

## 4. Scope

- Total-cholesterol model only. The cholesterol/HDL-ratio charts (Figs 3-4) use a function
  whose coefficients are not published.
- Ages 40-65, the range of the charts; the paper gives no other validity range.
- The formula follows Appendix A literally: S0(t) = exp(-exp(alpha)(t - 20)^p), risk for each
  cause = 1 - S(age + 10)/S(age), CVD risk = CHD risk + non-CHD risk (a sum, as printed).
