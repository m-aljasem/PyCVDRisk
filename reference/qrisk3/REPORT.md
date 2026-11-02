# QRISK3-2017 — verification report

Generated 2026-11-01 by `reference/qrisk3/report.py`.

## 1. Source

ClinRisk's open-source C implementation of QRISK3-2017 (LGPL-3.0-or-later), saved
unmodified as `source/qrisk3_2017.c` from the qrisk.org source page.
`extract_coefficients.py` generates `src/cvd_risk/qrisk3/coefficients.py` from it,
copying every literal as printed, and a test re-parses the C to confirm the file is unchanged.

## 2. Agreement with the reference

Primary reference: the same C code compiled unmodified (`harness.c`, `run_reference.py`).
This is the developers' own implementation (evidence level A). Secondary: the CRAN
package QRISK3, a third-party R port (level B).

```
reference: ClinRisk QRISK3-2017 C source (reference/qrisk3/source/qrisk3_2017.c), unmodified
compiler: cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 (-O0 -std=c99)
generated: 2026-11-01
secondary: QRISK3 (CRAN) 0.6.0 - R version 4.3.3 (2024-02-29)
```

30,000 seeded synthetic patients: every whole age 25-84 for both sexes, every
ethnicity, smoking and diabetes category, and every binary flag.

| Reference | Max rel diff |
|---|---|
| ClinRisk C | 4.1e-14 |
| QRISK3 (CRAN) | 4.1e-14 |

The CRAN package agrees with the C code bit for bit on these patients.

Scoring 30,000 rows: 0.051 s; 1,000,000 rows: 3.46 s.

## 3. Mutation check

Every non-zero number in the model (survivor, ethnicity and smoking terms, centring
means, 109 coefficients) multiplied by 1 + 1e-6: 151/151 caught.

## 4. Notes

- Age is truncated to whole years because ClinRisk's functions take an integer age.
- The QRISK3 web calculator imputes missing BMI, cholesterol ratio and SBP. That code is
  not in the open-source release, so these inputs are required here.
- Only the age range (25-84) is enforced. The other inputs are checked for being
  positive and finite only; ClinRisk's page has no input validation code.
- Licence: the module is LGPL-3.0-or-later (it carries ClinRisk's coefficients) while
  the rest of the package is MIT. ClinRisk's terms require their disclaimer to be held
  with any score shown; it is in the module docstring.
