# PREVENT — verification report

Generated 2026-11-01 by `reference/prevent/report.py`.

## 1. Primary source and the authors' worked examples

Coefficients are parsed from Supplemental Tables S12.A-J of Khan et al., Circulation
2024;149:430-449 (`build_coefficients.py`; files in `source/`). The original (2023) and
revised (2024-01-26) supplement spreadsheets agree on all 267 coefficient rows.

Each S12 table also contains the authors' own worked example (one patient, 10 risks).
All 100 published values are reproduced to rtol 1e-12
(`test_authors_worked_example`), using the coefficients as printed.

**HbA1c erratum (2026-10-14, doi:10.1161/CIR.0000000000001491).** The coefficients
labelled *HbA1c in DM* and *HbA1c no DM* are swapped in the 10- and 30-year HbA1c
and full models. The erratum text (`source/erratum_CIR.0000000000001491.txt`) states
that the two row headers were inverted in exactly these four tabs (S12C, S12E, S12H,
S12J) and in Table 3 of the paper. Independently, preventr 0.12.2 differs from S12
in exactly these values and nothing else. The S12 worked examples reproduce with the
values as originally labelled, so they are tested against the uncorrected table.

## 2. Agreement with the reference

Reference: third-party R package preventr, with its 3-decimal rounding disabled
(evidence level B).

```
reference: preventr (CRAN), R implementation of the AHA PREVENT equations
package_version: 0.12.2
R: R version 4.3.3 (2024-02-29)
generated: 2026-11-01
rounding: disabled (preventr:::round_half_up replaced by identity)
models: base, uacr, hba1c, sdi, full (SDI decile passed via zip); HbA1c erratum 2026-10-14 included
```

5,000 seeded synthetic patients (`make_patients.py`): every whole age 30-79, all
inputs across the allowable ranges, about 10% missing UACR, HbA1c and SDI each (so the
missing-indicator terms are exercised).

| Model | Years | Patients | Reference NA | Max rel diff (5 outcomes) | Time (s) |
|---|---|---|---|---|---|
| base | 10 | 5,000 | 89 | 3.8e-13 | 0.011 |
| base | 30 | 5,000 | 89 | 3.0e-14 | 0.009 |
| uacr | 10 | 5,000 | 89 | 6.3e-13 | 0.009 |
| uacr | 30 | 5,000 | 89 | 5.8e-14 | 0.010 |
| hba1c | 10 | 5,000 | 89 | 4.2e-13 | 0.009 |
| hba1c | 30 | 5,000 | 89 | 2.8e-14 | 0.009 |
| sdi | 10 | 5,000 | 89 | 2.2e-13 | 0.009 |
| sdi | 30 | 5,000 | 89 | 3.8e-14 | 0.010 |
| full | 10 | 5,000 | 89 | 1.9e-13 | 0.010 |
| full | 30 | 5,000 | 89 | 3.1e-14 | 0.013 |

Scoring 1,000,000 rows (full model, 10-year total CVD): 2.35 s.

## 3. Coefficient mutation check

Each coefficient was nudged by one unit in its last printed decimal place (1e-7).
A nudge counts as caught if the comparison with preventr then fails.

| Model | Years | Coefficients mutated | Caught |
|---|---|---|---|
| base | 10 | 194 | 194 |
| base | 30 | 204 | 204 |
| uacr | 10 | 214 | 214 |
| uacr | 30 | 224 | 224 |
| hba1c | 10 | 224 | 224 |
| hba1c | 30 | 234 | 234 |
| sdi | 10 | 224 | 224 |
| sdi | 30 | 234 | 234 |
| full | 10 | 274 | 274 |
| full | 30 | 284 | 284 |

## 4. Findings

1. **preventr rejects ages 79.01-79.99.** The source's allowable range is
   "30-79 years", which this package reads as [30, 80), the same convention used for
   SCORE2. preventr returns NA for these ages (89 of 5,000 patients here). All other
   rows agree.
2. **The HbA1c labels in S12 are wrong** (corrected by the 2026-10-14 erratum;
   see section 1). Every implementation built from the supplement before then,
   including preventr <= 0.12.0, gives wrong HbA1c-model risks: up to 30%
   relative error in 10-year total CVD on this synthetic set.
3. **preventr's ranges differ from the source's.** Source: eGFR 15-150, HbA1c 3-15%,
   BMI 18.5 to <40. preventr: eGFR 15-140, HbA1c 4.5-15, BMI 18.5-39.9. This package
   follows the source.
4. **The original (2023) S12 tables label HDL-C "per 1 mml/L"**, but the term is
   (HDL - 1.3)/0.3. The revised supplement's own formulas and worked examples use
   0.3, and so does this package.

## 5. Open items

- Compare against the AHA online calculator (evidence level A).
