# QRISK2-2014 — verification report

Generated 2026-11-01 by `reference/qrisk2/report.py`.

## 1. Source

ClinRisk's open-source C implementation of QRISK2-2014 (LGPL-3.0-or-later; release
packaged by Debian as `qrisk2`), saved unmodified in `source/` (female model
`Q80_model_4_0.c`, male `Q80_model_4_1.c`, `utils.c`, headers, licence).
`extract_coefficients.py` generates `src/cvd_risk/qrisk2/coefficients.py` from it with the
QRISK3 parser, and a test re-parses the C to confirm the file is unchanged.

## 2. Agreement with the reference (evidence level A)

The same C code compiled unmodified, called through its public functions `cvd_female` /
`cvd_male`, so ClinRisk's own input validation is part of the comparison.

```
reference: ClinRisk QRISK2-2014 C source (reference/qrisk2/source/c), unmodified, via cvd_female/cvd_male
compiler: cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 (-O0 -std=c99)
generated: 2026-11-01
```

30,000 seeded synthetic patients: every whole age 23-86 for both sexes, every
ethnicity, smoking and diabetes category, every flag, and continuous inputs running 5%
past ClinRisk's validation ranges with 2% of rows at each exact bound.

- Rows ClinRisk rejects: 9,188; the port returns NaN for exactly these rows.
- Other rows: max relative difference 2.9e-13.

## 3. Mutation check

Every non-zero number in the model (survivor, ethnicity and smoking terms, centring
means, coefficients) multiplied by 1 + 1e-6: 124/124 caught.

## 4. Notes

- QRISK2-2014's coefficients are not published in a paper (the 2008 BMJ paper describes
  an earlier version), so the C release is the only primary source.
- Age is truncated to whole years because ClinRisk's functions take an integer age.
- Validation ranges are ClinRisk's (inclusive): age 25-84, BMI 20-40, cholesterol/HDL
  ratio 1-12, SBP 70-210, Townsend -7 to 11.
- Licence: like QRISK3, the module is LGPL-3.0-or-later and carries ClinRisk's disclaimer.
