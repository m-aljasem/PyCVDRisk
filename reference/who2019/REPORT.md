# WHO 2019 — verification report

Generated 2026-11-01 by `reference/who2019/report.py`.

## 1. Sources

- Log hazard ratios: Kaptoge et al., Lancet Glob Health 2019, appendix 1 Table 1.6.
  `check_coefficients.py` confirms all 64 values against the PDF text.
- Baseline survival: the paper (Table 2) prints 3 decimals. The 7-decimal values come
  from the authors' Stata program `whocvdrisk` via WHORiskCalculator 1.1.0, and all 8
  round to the printed ones.
- Recalibration parameters (3,346 rows: 21 regions and 190 countries, by sex, for
  1990-2017): the authors' Stata data file via WHORiskCalculator 1.1.0 (MIT),
  exported by `extract_data.R`. These are not printed in the publication.

## 2. The official WHO risk charts

`parse_charts.py` reads all 42 published charts (supplementary appendix 2; 21 regions,
lab and non-lab), 44,100 cells in total. **Every cell is reproduced exactly**
(`test_reproduces_every_official_chart_cell`), with cells evaluated at band midpoints:
age band + 2.5; SBP 110/130/150/170/190; cholesterol 3.5-7.5; BMI 17.5-37.5. The
midpoints were found by search and are sharply determined: moving a BMI midpoint by
0.5 breaks 8-13% of that band's cells, and moving age to band + 2 breaks 31% of lab
cells. This independently confirms the region-level 2017 recalibration
parameters against WHO's own publication (evidence level A for the region model).

## 3. Agreement with the reference

Reference: WHORiskCalculator (R, CRAN), which by its own account is tested against
a re-implementation of the authors' Stata code (evidence level B).

```
reference: WHORiskCalculator (CRAN), R implementation of the 2019 WHO CVD risk models
package_version: 1.1.0
R: R version 4.3.3 (2024-02-29)
generated: 2026-11-01
rounding: none (package returns unrounded risks)
```

30,000 seeded synthetic patients: every country (so every region), every whole age
40-80, years 1990-2017.

| Model | Recalibration | Max rel diff | Time (s) |
|---|---|---|---|
| lab | region | 5.8e-14 | 0.045 |
| lab | country | 7.0e-14 | 0.031 |
| nonlab | region | 6.5e-14 | 0.033 |
| nonlab | country | 6.6e-14 | 0.033 |

Scoring 1,000,000 rows (lab, country recalibration): 0.99 s.

## 4. Mutation check

- Model coefficients, +1 in the last printed digit: 72/72 caught.
- Recalibration parameters, x (1 + 1e-6): 25,168 of 26,768 are used by at least one
  synthetic patient, and 25,168/25,168 of those are caught.

## 5. Notes

- v1 of this package validated against `whoishRisk`, which is the 2007 WHO/ISH model,
  not this one.
- Five territories (SSD, VIR, ASM, GUM, MNP) have no country parameters. With
  `recalibration="country"` they fall back to their region, as WHORiskCalculator does.
- The author email drafted in `README.md` is no longer needed.
