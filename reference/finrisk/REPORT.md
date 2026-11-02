# FINRISK calculator: verification report

Generated 2026-11-01 by `reference/finrisk/report.py`.

Evidence level C: Vartiainen et al., Global Heart 2016 (`source/vartiainen2016.pdf`). No independent
implementation was available (the online calculator is not reachable here).

## 1. Equations

- male chd: rebuilt equation found in the PDF
- male stroke: rebuilt equation found in the PDF
- female chd: rebuilt equation found in the PDF
- female stroke: rebuilt equation found in the PDF

## 2. Coefficients vs Table 1 odds ratios

exp(Table 2 coefficient), rounded as printed, against the Table 1 odds ratio (HDL coefficient taken with
its sign as an effect). Table 1 also prints an OR for cholesterol in the stroke models; Table 2 drops that
term. Differences are expected if Table 2 was refitted (e.g. without that term); they are reported, not
used as a pass/fail check, since Figure 1 checks the equations themselves.

| Model | Factor | OR | Coefficient | exp(coef) rounds to OR |
|---|---|---:|---:|---|
| male chd | Age, yrs | 1.083 | 0.0802 | no |
| male chd | Smoking (no = 0, yes = 1) | 1.87 | 0.626 | yes |
| male chd | Cholesterol, mmol/l | 1.39 | 0.3293 | yes |
| male chd | Systolic blood pressure, mm Hg | 1.017 | 0.0166 | yes |
| male chd | HDL-cholesterol, mmol/l | 0.55 | 0.5893 | no |
| male chd | Diabetes (no = 0, yes = 1) | 2.10 | 0.7417 | yes |
| male chd | Parents | 1.40 | 0.3138 | no |
| female chd | Age, yrs | 1.101 | 0.0962 | yes |
| female chd | Smoking (no = 0, yes = 1) | 2.41 | 0.8776 | yes |
| female chd | Cholesterol, mmol/l | 1.24 | 0.2119 | yes |
| female chd | Systolic blood pressure, mm Hg | 1.018 | 0.0175 | yes |
| female chd | HDL-cholesterol, mmol/l | 0.33 | 1.1009 | no |
| female chd | Diabetes (no = 0, yes = 1) | 2.80 | 1.0303 | yes |
| female chd | Parents | 1.51 | 0.409 | yes |
| male stroke | Age, yrs | 1.122 | 0.1153 | yes |
| male stroke | Smoking (no = 0, yes = 1) | 1.65 | 0.4981 | yes |
| male stroke | Cholesterol, mmol/l | 0.97 | 0 | - |
| male stroke | Systolic blood pressure, mm Hg | 1.015 | 0.0149 | yes |
| male stroke | HDL-cholesterol, mmol/l | 0.65 | 0.4406 | no |
| male stroke | Diabetes (no = 0, yes = 1) | 2.42 | 0.879 | no |
| male stroke | Parents | 1.34 | 0.2933 | yes |
| female stroke | Age, yrs | 1.067 | 0.0633 | no |
| female stroke | Smoking (no = 0, yes = 1) | 1.52 | 0.4163 | yes |
| female stroke | Cholesterol, mmol/l | 0.98 | 0 | - |
| female stroke | Systolic blood pressure, mm Hg | 1.009 | 0.00893 | yes |
| female stroke | HDL-cholesterol, mmol/l | 0.47 | 0.7636 | no |
| female stroke | Diabetes (no = 0, yes = 1) | 3.45 | 1.2383 | yes |
| female stroke | Parents | 1.72 | 0.547 | no |

17/26 coefficients round to their printed odds ratio.

## 3. Figure 1 (10-year CHD risk, age 50)

The caption gives SBP 160 vs 120 and cholesterol 7 vs 4.5 mmol/L; HDL and family history are not stated.
HDL 1.3 (men) and 1.64 (women) with no family history reproduce the no-risk-factor bars.
Panel B is printed in ascending order within each group, so it is compared sorted.

| Sex | Printed | Package, SBP 160 throughout | Package, SBP 180 for BP+diabetes and cholesterol+BP |
|---|---|---|---|
| male | [1.1, 2.04, 2.12, 2.29, 2.48, 3.9, 4.2, 4.54, 5.07, 5.97, 6.44, 7.85, 8.46, 9.4, 16.25] | [1.1, 2.04, 2.12, 2.29, 2.48, 3.9, 4.2, 4.54, 5.07, 4.35, 4.71, 7.85, 8.46, 9.4, 16.24] (12/15) | [1.1, 2.04, 2.12, 2.29, 2.48, 3.9, 4.2, 4.54, 5.07, 5.97, 6.44, 7.85, 8.46, 9.4, 16.24] (14/15) |
| female | [0.31, 0.52, 0.62, 0.74, 0.86, 1.24, 1.45, 1.47, 1.48, 2.04, 2.41, 2.48, 2.87, 4.02, 6.64] | [0.31, 0.52, 0.62, 0.74, 0.86, 1.04, 1.24, 1.45, 1.47, 1.71, 2.04, 2.48, 2.87, 4.02, 6.64] (13/15) | [0.31, 0.52, 0.62, 0.74, 0.86, 1.24, 1.45, 1.47, 1.48, 2.04, 2.41, 2.48, 2.87, 4.02, 6.64] (15/15) |

With SBP 160 for every bar, 12/15 (men) and 13/15 (women) match to the printed
0.01% (the men's all-factor bar aside, see below). Two bars differ in each sex, both times "blood pressure and diabetes" and "cholesterol and blood
pressure"; computed with SBP 180 they match (14/15 men, 15/15 women; the remaining
men's bar, all risk factors, differs by 0.01, and no single HDL value matches all 15). So the figure was
probably drawn with SBP 180 for those two bars, against its caption. The equations are unaffected.

