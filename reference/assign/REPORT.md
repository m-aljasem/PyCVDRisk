# ASSIGN: verification report

Models: ASSIGN 1.0 (Woodward, Brindle, Tunstall-Pedoe; Heart 2007;93:172) and ASSIGN 2.0 (Welsh, Kimenai,
Woodward; Heart 2025;111:557), a sex-specific Cox model with eight inputs (age, total and HDL cholesterol, SBP,
diabetes, family history, cigarettes per day, SIMD score / 10). `cvd_risk.assign(..., version=1|2)`.

Evidence for version 2: level A, the official ASSIGN v2.0 calculator (Right Decision Service, NHS Scotland,
rightdecisions.scot.nhs.uk/assign-v20), plus level C, the printed equations in `source/`. Version 1 is level C only:
the 2007 appendix (`woodward2007_appendix.pdf`). The live calculator runs only version 2.

The calculator computes in the browser. Its `executeAlgorithm` method was copied verbatim from the page's
`app.js` (`source/rightdecisions_executeAlgorithm.js`). `run_calculator.py` typed 200 seeded patients
into the real form, with SIMD entered directly rather than by postcode, and saved each displayed score to
`fixtures/calculator_reference.jsonl`. The page shows 1 decimal below 10 and a whole number from 10.

## Checks (`tests/assign/test_assign.py`)

1. Every beta, mean and survival value is found in the PDF text: version 1 in the 2007 appendix, version 2 means
   and survival in the 2025 supplement.
2. The calculator's 32 constants (8 betas, 8 means, survival, per sex) equal the package's version-2 constants.
3. Official calculator run: **200/200 displayed scores reproduced exactly** (package risk rounded as the page
   rounds). 100 women, 75 smokers, 42 with diabetes; scores 0.4-76, 116 shown with a decimal (below 10).
4. A hand calculation (man, version 1) and the version-2 mean-person case agree to 1e-12.
5. Input handling: age outside 30-74, unknown sex, missing SIMD and non-binary diabetes give NaN.

## Printed discrepancies

| Item | 2007 appendix | 2025 supplement / Right Decisions text | Official calculator | Used |
|---|---|---|---|---|
| Men's total-cholesterol beta | 0.22286 (in `asm` and `Lbar`) | 0.22286 in `asm`, **0.2286** in `lbarm` and `lbarm2` | 0.22286 in both | 0.22286 |
| Women's family-history beta | 0.49159 | 0.492 | 0.492 | 0.49159 (v1), 0.492 (v2) |

The calculator settles the first item: 0.2286 is a typo in the printed equations only. Read literally, it would lower
men's hazard by 3.5% (version 1) and 3.1% (version 2).

## Calculator behaviour the package does not copy

- It accepts ages 25-90 and warns outside 30-74; the package returns NaN outside 30-74.
- It fills blank SBP, cholesterol, HDL and cigarettes with age- and sex-specific averages, and a blank SIMD or
  postcode with a median (16.23); the package returns NaN for missing inputs.
- It accepts SIMD 1-89, SBP 80-250, total cholesterol 2-12.5, HDL 0.3-3.5 (below total cholesterol) and 0-100
  cigarettes; the package does not apply these limits.

## Limits

- Age range 30-74 (the 2007 cohort and the calculator's "usual" range); the 2025 derivation used ages 40-69.
- SIMD is the score on its 0-100 scale. Version 1 was derived on SIMD 2004 and version 2 on SIMD 2020; a score from
  another release is not equivalent.
- The 2025 paper calls the outcome ASCVD (hospitalisation, revascularisation or cardiovascular death); the 2007
  outcome was CHD, stroke and TIA, fatal or not.
- The Right Decisions page lists groups to treat as high risk without scoring (CKD, familial hypercholesterolaemia,
  diabetes over 40); the function does not apply that rule, and neither does it apply the 10% (version 2) or 20%
  (version 1) threshold.
