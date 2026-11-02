# LIFE-CVD2 — verification report

Generated 2026-11-01 by `reference/life_cvd2/report.py`.

## 1. Source

Hageman et al., Eur J Prev Cardiol 2024;31:1690-1699, and its online supplement, saved in
`source/`. `check_coefficients.py` parses Supplementary Tables 3-5 from the PDF: all 36
coefficients, 264 baseline survivals (ages 35-100) and 32 recalibration scales match.

The supplement does not print how predictors are centred or the recalibration formula; it says
both follow SCORE2. We use SCORE2's centring (age 60 per 5 years, SBP 120 per 20 mmHg, total
cholesterol 6, HDL 1.3 per 0.5) and recalibration, 1 - exp(-exp(scale1 + scale2 ln(-ln(1 - risk)))),
with the age terms evaluated at each attained age of the life table. Section 2 shows this
reproduces the authors' own charts.

## 2. The authors' charts (evidence level A)

Appendix 1 of the supplement prints 16 charts: lifetime risk to 80, and CVD-free life years
gained from lifelong SBP lowering to 135 mmHg (HR 0.80 per 10 mmHg), 40% LDL lowering
(HR 0.78 per mmol/L) and smoking cessation (HR 0.60 CVD, 0.73 non-CVD), each for the four ESC
risk regions. Cells are read from the PDF text layer (no transcription). Inputs are the band
midpoints: age (e.g. 77 for 75-79), SBP 110/130/150/170, non-HDL 3.5/4.5/5.5/6.5 mmol/L.

| Chart family | Cells | Exact after rounding | Within half a printed unit | Untreated cells shown as - |
|---|---|---|---|---|
| Lifetime risk to 80 (%) | 2048 | 2043 | 2048 | n/a |
| Gain from SBP lowering to 135 mmHg (years) | 1280 | 1279 | 1280 | 1280 |
| Gain from 40% LDL lowering (years) | 2560 | 2557 | 2560 | n/a |
| Gain from smoking cessation (years) | 1280 | 1277 | 1279 | 1280 |

**7167/7168 cells agree within half a printed unit** (plus 1% of a unit for ties); the largest
difference is 0.512 units. Cells outside: smoking high, female age 42, smoker 1, SBP 150, non-HDL 6.5: printed 6.1, ours 6.1512.

Two chart inputs are not stated in the paper and were found by fitting, and one chart row is defective:

- **HDL in the charts is 1.4 mmol/L for women and 1.2 for men** (total cholesterol = non-HDL + HDL).
- **LDL in the LDL-lowering charts is non-HDL - 0.8 mmol/L.**
- **Chart defect: the row "age 65-69, SBP 100-119" is computed at age 62**, the midpoint of the
  60-64 band, in all 16 charts. At age 67 the lifetime-risk row is higher than the SBP 120-139 row
  above it. At age 62 the row is reproduced like every other. With the row taken at 67, agreement
  falls to 7016/7168.

## 3. How tightly the charts pin the model

| Alternative | Cells within half a unit |
|---|---|
| As implemented | 7167 |
| HDL 1.3 both sexes | 5308 |
| HDL 1.5 women / 1.3 men | 5323 |
| LDL = non-HDL - 0.6 | 5488 |
| LDL = non-HDL - 1.0 | 5474 |
| age terms fixed at entry age, not attained age | 1076 |
| chart row 65-69 / SBP 100-119 at age 67 (its band midpoint) | 7016 |

Moving any one of the 36 coefficients or 32 recalibration scales by one unit of its last printed
digit pushes at least one chart cell outside half a printed unit in 136/136 cases.

## 4. Reference implementations

The official calculator is the web tool at u-prevent.com, which cannot be queried offline. No
open-source implementation was found.

## 5. Scope

- People without established CVD. The diabetes terms in Table 3 are left out, as the supplement
  instructs.
- Whole-year ages: 35-79 for lifetime risk, 35-99 for life expectancy. Life expectancy is the
  first crossing of 0.5 by event-free survival, linear within the year. When survival is still
  >= 0.5 at 101 it is inf; the supplement's last-survival rule for benefit beyond 100 is not
  implemented.
- Treatment effects enter as lifelong hazard ratios (`hr_cvd`, `hr_non_cvd`), applied to each
  yearly risk after recalibration, as in the supplement.
