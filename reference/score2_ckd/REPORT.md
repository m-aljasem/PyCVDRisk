# SCORE2-CKD — verification report

Generated 2026-11-01 by `reference/score2_ckd/report.py`.

## 1. Source

Matsushita et al., Eur J Prev Cardiol 2023, doi:10.1093/eurjpc/zwac176, supplementary
material section 1.1.2, Tables A-C (file in `source/`): expected eGFR and expected ACR
equations, eGFR / ACR / dipstick add-ons for SCORE2 and SCORE2-OP. The base risks are
the package's already verified SCORE2 / SCORE2-OP.

## 2. Worked example (patient A)

The supplement's step-by-step example is reproduced to the precision it prints
(products rounded to 4 decimals): expected eGFR 103.1941, expected ACR 15.3906,
4.68% after the eGFR add-on and 6.07% after the ACR add-on, starting from its stated base.

## 3. Agreement with the reference

Reference: RiskScorescvd `SCORE2_CKD` (third-party R, rounding disabled; level B).

```
reference: RiskScorescvd SCORE2_CKD (vendored at reference/vendor/RiskScorescvd)
package_version: 0.3.1
R: R version 4.3.3 (2024-02-29)
generated: 2026-11-01
rounding: disabled (round replaced by identity)
```

20,000 seeded synthetic patients, every whole age 40-89 (SCORE2 and SCORE2-OP), eGFR
15-130 (all three spline segments), ACR 1-3000 mg/g, every dipstick grade.

| Add-on | Patients compared | Max rel diff |
|---|---|---|
| egfr | 20,000 | 6.0e-14 |
| acr | 20,000 | 8.2e-14 |
| dipstick | 13,589 | 5.3e-14 |

## 4. Mutation check

Each published add-on coefficient nudged by +1 in its last decimal: 45/49 caught.

Not caught: `CKD_DIPSTICK_ADDON.score2_op.1`, `CKD_DIPSTICK_ADDON.score2_op.2`, `CKD_DIPSTICK_ADDON.score2_op.3`, `CKD_DIPSTICK_ADDON.score2_op.4`

The uncaught values are the SCORE2-OP dipstick coefficients, which the reference does
not implement (finding 1). `check_coefficients.py` checks them, and every other add-on
coefficient, against the text of its own equation in the supplement.

## 5. Findings

1. **RiskScorescvd uses the SCORE2 dipstick coefficients for SCORE2-OP.** The
   supplement (Table C) gives separate SCORE2-OP values (trace 0.2517, + 0.3872,
   ++ 0.5458, +++ 0.4207); RiskScorescvd applies the SCORE2 ones (0.2644, 0.4126,
   0.4761) at all ages. Substituting them reproduces the reference exactly, so this
   is the only disagreement (all 6,411 rows aged >= 70 with a positive dipstick; up to
   6.9% relative). `test_finding_reference_uses_score2_dipstick_coefficients_for_op`.
2. **Patient A's base risk is not reproducible.** The supplement states SCORE2 = 1.96%
   (low-risk region); the stated inputs give 2.48% (RiskScorescvd agrees) and no
   region gives 1.96%. The add-on steps themselves reproduce from 1.96%.
3. **Sign typo in the example sum.** The expected-ACR line subtracts the eGFR<60
   term (-0.4057), but the printed result (15.3906) requires +0.4057, as in the
   equation. The equation is followed.
4. The SCORE2 eGFR add-on has no eGFR >= 90 segment (Table A footnote); the SCORE2-OP
   add-on does. Both implemented as published.
