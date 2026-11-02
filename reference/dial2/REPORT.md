# DIAL2 — verification report

Generated 2026-11-01 by `reference/dial2/report.py`.

## 1. Source

Østergaard et al., Eur J Prev Cardiol 2023;30:61-69, and its online supplement (PDF and Word),
saved in `source/`. `check_coefficients.py` reads Supplementary Tables 3-5 (core model) from the
Word file and confirms every value also appears in the PDF: 56 coefficients, 260 baseline
survivals (ages 30-94) and 16 recalibration scales match. Unlike LIFE-CVD2, the supplement prints
every transformation (centring and scaling), the life table and the lifetime-risk definition.

## 2. The paper's worked examples (evidence level C)

Figure 3 shows CVD-free life expectancy with and without lifelong 10 mmHg SBP and 1.5 mmol/L LDL
lowering (HR 0.80 x 0.78^1.5 on CVD) for two men aged 50 in the moderate-risk region (non-smoker,
SBP 140, total cholesterol 5.5, HDL 1.3). The printed 'with treatment' value equals the rounded life
expectancy plus the rounded gain in both examples, so we compare it that way.

| Example | Printed (LE, treated, gain) | Ours, rounded | Ours, unrounded |
|---|---|---|---|
| A (caption: eGFR 70) | (70.5, 74.1, 3.6) | (70.5, 74.1, 3.6) | 70.542 / 74.166 / 3.623 |
| B (caption: eGFR 70) | (76.0, 79.1, 3.1) | (74.4, 77.8, 3.4) | 74.409 / 77.781 / 3.371 |
| B with eGFR 90 | (76.0, 79.1, 3.1) | (76.0, 79.1, 3.1) | 75.993 / 79.135 / 3.141 |

- **Example A is reproduced** (all three numbers).
- **Example B is not reproduced with the caption's inputs** (life expectancy 74.4 instead of 76).
  Changing one input at a time over a plausible range, the only values that reproduce all three
  printed numbers are: egfr = 90.
  We read this as a caption error: the figure was drawn with eGFR 90, not 70.
- Sensitivity: moving any one of the 56 coefficients or 16 scales by 1% (relative) changes at
  least one of the six printed numbers in 4/72 cases. Six one-decimal numbers are weak
  evidence; there is no published chart or calculator output to compare against.

## 3. Reference implementations

The official calculator is the web tool at u-prevent.com, which cannot be queried offline. No
open-source implementation was found (searched November 2026).

## 4. Scope

- Core model. The extended model (adding insulin, retinopathy, BMI and albuminuria) has its own
  coefficients and baseline but no recalibration scales and was not externally validated; it is
  not implemented.
- Low- and moderate-risk ESC regions only; the paper has no scales for high and very high.
- Whole-year ages 30-94 (30-85 for 10-year risk). Lifetime risk runs to age 95 and life
  expectancy is inf when survival is still >= 0.5 at 95. The derivation population was 30-85.
