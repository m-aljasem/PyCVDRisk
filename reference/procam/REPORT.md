# PROCAM score: verification report

Generated 2026-11-01 by `reference/procam/report.py`.

Evidence level C: Assmann et al., Circulation 2002 (`source/assmann2002.pdf`), Table 3 (points)
and Table 4 (points -> 10-year risk). The paper gives no worked example and no independent
implementation was available. Inputs are entered in mg/dL and converted, so each band edge also
tests the mmol/L round trip.

| Input | Band (low-high, source units) | Points at low / high | Printed | |
|---|---|---|---:|---|
| age | 35-39.99 | 0 / 0 | 0 | "Age, y 35-39 0 40-44 6 45-49 11 50-54 16 55-59 21 60-65 26" found |
| age | 40-44.99 | 6 / 6 | 6 | |
| age | 45-49.99 | 11 / 11 | 11 | |
| age | 50-54.99 | 16 / 16 | 16 | |
| age | 55-59.99 | 21 / 21 | 21 | |
| age | 60-65.99 | 26 / 26 | 26 | |
| ldl_chol | 50-99.99 | 0 / 0 | 0 | "LDL cholesterol, mg/dL <100 0 100-129 5 130-159 10 160-189 14 ≥190 20" found |
| ldl_chol | 100-129.99 | 5 / 5 | 5 | |
| ldl_chol | 130-159.99 | 10 / 10 | 10 | |
| ldl_chol | 160-189.99 | 14 / 14 | 14 | |
| ldl_chol | 190-300 | 20 / 20 | 20 | |
| hdl_chol | 20-34.99 | 11 / 11 | 11 | "HDL cholesterol, mg/dL <35 11 35-44 8 45-54 5 ≥55 0" found |
| hdl_chol | 35-44.99 | 8 / 8 | 8 | |
| hdl_chol | 45-54.99 | 5 / 5 | 5 | |
| hdl_chol | 55-100 | 0 / 0 | 0 | |
| triglycerides | 50-99.99 | 0 / 0 | 0 | "Triglycerides, mg/dL <100 0 100-149 2 150-199 3 ≥200 4" found |
| triglycerides | 100-149.99 | 2 / 2 | 2 | |
| triglycerides | 150-199.99 | 3 / 3 | 3 | |
| triglycerides | 200-500 | 4 / 4 | 4 | |
| smoker | 0-0 | 0 / 0 | 0 | "Smoker No 0 Yes 8" found |
| smoker | 1-1 | 8 / 8 | 8 | |
| diabetes | 0-0 | 0 / 0 | 0 | "Diabetes mellitus No 0 Yes 6" found |
| diabetes | 1-1 | 6 / 6 | 6 | |
| family_history | 0-0 | 0 / 0 | 0 | "MI in family history No 0 Yes 4" found |
| family_history | 1-1 | 4 / 4 | 4 | |
| sbp | 100-119.99 | 0 / 0 | 0 | "Systolic blood pressure, mm Hg <120 0 120-129 2 130-139 3 140-159 5 ≥160 8" found |
| sbp | 120-129.99 | 2 / 2 | 2 | |
| sbp | 130-139.99 | 3 / 3 | 3 | |
| sbp | 140-159.99 | 5 / 5 | 5 | |
| sbp | 160-220 | 8 / 8 | 8 | |

30/30 bands give the printed points at both ends; 8/8 printed rows found in the PDF.
Table 4 (points 21-59) found in the PDF as transcribed: yes.

Not implemented: the full Cox model (Table 2). The paper gives no example to check it against.
