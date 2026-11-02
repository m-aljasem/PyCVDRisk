# Reynolds Risk Score: verification report

Generated 2026-11-01 by `reference/reynolds/report.py`.

## 1. Source

Model B, "the Reynolds Risk Score": women from Ridker et al., JAMA 2007 (reprinted with
corrections; Box), men from Ridker et al., Circulation 2008 (Appendix C). Both PDFs are in
`source/`. Each coefficient is quoted as a term of the printed formula: 19/19 found in the PDF text.

No independent implementation was available: the official calculator (reynoldsriskscore.org)
is gone, and no R or Python package implements these papers. Evidence is level C: the papers'
worked examples.

## 2. Worked examples

Women: Table 6 (age 50, smoker, no diabetes, BP 155/85, total cholesterol 240, HDL 35 mg/dL).
Men: Clinical Examples (age 65, non-smoker, SBP 130, total cholesterol 205, HDL 45 mg/dL, hsCRP 4.0,
parental history), model B. Printed risks have one decimal.

| Sex | hsCRP | Parental history | Printed % | This package % | Rounds to printed | Range at printed precision | Printed in range |
|---|---:|---|---:|---:|---|---|---|
| woman | 0.1 | no | 4.9 | 4.932 | yes | 4.825-5.041 | yes |
| woman | 0.1 | yes | 7.5 | 7.538 | yes | 7.374-7.706 | yes |
| woman | 0.5 | no | 6.5 | 6.534 | yes | 6.444-6.625 | yes |
| woman | 0.5 | yes | 9.9 | 9.941 | yes | 9.802-10.082 | yes |
| woman | 1.0 | no | 7.4 | 7.369 | yes | 7.292-7.447 | yes |
| woman | 1.0 | yes | 11.2 | 11.186 | yes | 11.066-11.306 | yes |
| woman | 3.0 | no | 8.9 | 8.907 | yes | 8.769-9.047 | yes |
| woman | 3.0 | yes | 13.4 | 13.460 | no | 13.250-13.673 | yes |
| woman | 5.0 | no | 9.7 | 9.722 | yes | 9.548-9.898 | yes |
| woman | 5.0 | yes | 14.6 | 14.656 | no | 14.395-14.921 | yes |
| woman | 8.0 | no | 10.5 | 10.533 | yes | 10.323-10.747 | yes |
| woman | 8.0 | yes | 15.8 | 15.842 | yes | 15.528-16.161 | yes |
| woman | 10.0 | no | 10.9 | 10.940 | yes | 10.711-11.174 | yes |
| woman | 10.0 | yes | 16.4 | 16.435 | yes | 16.094-16.782 | yes |
| woman | 20.0 | no | 12.3 | 12.301 | yes | 12.006-12.602 | yes |
| woman | 20.0 | yes | 18.4 | 18.405 | yes | 17.971-18.847 | yes |
| man | 4.0 | yes | 20.4 | 20.490 | no | 20.199-20.783 | yes |

14/17 round to the printed value; 17/17 lie within the range
the formula gives when every printed number (coefficients, S0 and mean) moves by up to half
its last printed digit. The papers computed with unrounded coefficients, so differences of
one in the last printed decimal are expected.
