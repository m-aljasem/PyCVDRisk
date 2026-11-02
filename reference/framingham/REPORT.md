# Framingham 2008 general CVD — verification report

Generated 2026-11-01 by `reference/framingham/report.py`.

## 1. Sources

- Lab model: D'Agostino et al., Circulation 2008, Table 2; mean sums from the worked
  examples. BMI model: supplementary Table I. `check_coefficients.py` compares all 32
  printed values with the PDF text: 32/32 match.
- The BMI model's mean sums (26.0145, 23.9388) are not printed in either PDF. They come
  from the Framingham Heart Study's own calculator spreadsheets (`source/gencardio_*.xls`,
  framinghamheartstudy.org), which also confirm both baseline survivals.

## 2. Official calculator and worked examples (evidence level A + C)

- Each spreadsheet caches one example patient scored four ways (sex x treated SBP).
  All 8 cached risks are reproduced to < 1e-12 (`test_reproduces_official_spreadsheet`).
- Paper Score Sheet, case 1 (woman, 61) and case 2 (man, 53): 10.5% and 15.6%,
  as printed (`test_paper_worked_examples`). Together these exercise every term.

## 3. Agreement with CVrisk (evidence level B)

```
reference: CVrisk 1.2.1 (rounding, 1% floor and 30% cap removed)
official: Framingham Heart Study spreadsheets gencardio_lipids.xls, gencardio_bmi.xls (cached examples)
R: R version 4.3.3 (2024-02-29)
generated: 2026-11-01
```

20,000 seeded synthetic patients; 19,646 compared.

| Model | Max rel diff | Time (s) |
|---|---|---|
| lab | 2.1e-14 | 0.007 |
| bmi | 1.7e-14 | 0.006 |

Scoring 1,000,000 rows (lab): 0.68 s.

CVrisk differs from the source in three ways, all removed or excluded before comparing:
it returns NA for ages 74.01-74.99 (this package reads "30-74" as [30, 75)), floors
risks below 1% to 1%, and caps BMI-model risks at 30%. The publication does none of these.

## 4. Mutation check

Every value, +1 in its last printed digit: 34/34 caught by CVrisk.
