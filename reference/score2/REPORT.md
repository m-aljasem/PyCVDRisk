# SCORE2 family — verification report

Generated 2026-11-01 by `reference/score2/report.py`.

## 1. Agreement with the reference

Reference: third-party R package RiskScorescvd, run with its 0.1% rounding
disabled so answers compare at machine precision (evidence level B).

```
reference: RiskScorescvd (CRAN) — vendored source, see reference/vendor/RiskScorescvd/DESCRIPTION
package_version: 0.3.1
R: R version 4.3.3 (2024-02-29)
dplyr: 1.1.4
generated: 2026-11-01
rounding: disabled (round and round_to_nearest_digit overridden with identity)
```

| Variant | Patients | NaN | Max abs diff | Max rel diff | Pass (rtol 1e-12) | Time (s) |
|---|---|---|---|---|---|---|
| score2 | 20,000 | 0 | 4.44e-16 | 5.56e-14 | yes | 0.016 |
| score2_op | 20,000 | 0 | 4.44e-16 | 4.15e-15 | yes | 0.015 |
| score2_diabetes | 20,000 | 0 | 5.55e-16 | 1.14e-13 | yes | 0.013 |
| score2_asia_pacific | 20,000 | 0 | 1.14e-15 | 4.38e-14 | yes | 0.011 |

Patients: seeded synthetic set (`make_patients.py`) covering every sex x region x
smoker x diabetes combination and every integer age in range, both ends included.

## 2. Coefficient mutation check

Each published coefficient, baseline survival, mean linear predictor and region
scale was nudged by one unit in its last published decimal place (e.g. 0.3742 ->
0.3743). The change counts as caught if the reference comparison then fails.

| Variant | Coefficients mutated | Caught |
|---|---|---|
| score2 | 40 | 40 |
| score2_op | 42 | 42 |
| score2_diabetes | 52 | 52 |
| score2_asia_pacific | 40 | 40 |

## 3. Behaviour of the published models found by property testing

Reproduced identically by the R reference, so these are properties of the
published equations, not implementation errors. Each is pinned by a
`test_finding_*` test in `tests/score2/test_score2.py`.

1. **Region order is not monotone at low risk.** Because recalibration slopes
   differ by region, the *moderate* region gives a higher SCORE2 risk than the
   *high* region for low-risk people (11.9% of a uniform synthetic population,
   risks up to ~4.9%, i.e. across the 2.5% and 5% treatment thresholds), and
   *low* exceeds *moderate* for some women (risks < 0.7%).
2. **SCORE2 risk can fall with age** for extreme profiles (e.g. male diabetic
   smoker, SBP 175, total cholesterol 9, HDL 0.75): interaction terms cancel the
   age effect.
3. **SCORE2-OP: smoking lowers risk in men older than ~87.3 years**
   (0.3524 - 0.0247 x (age - 73) < 0), and risk can fall with age for some
   profiles (1.7% of synthetic 70-88-year-olds).

4. **SCORE2 Asia-Pacific region order inverts at both ends** (uniform synthetic
   population): *high* exceeds *very high* for 1.4% (high-risk profiles, risks up
   to ~87%), and *low* exceeds *moderate* for 4.5% (risks up to ~5%). Follows
   directly from the published rescaling factors.

To do before publication: confirm each against the official ESC risk charts.

## 4. Coefficient provenance — open items

- SCORE2 Asia-Pacific rescaling factors were transcribed from the primary
  source (Supplementary Table 5, file in `source/`) and agree exactly with
  RiskScorescvd. Its *very high* region has a lower intercept than *high* for
  both sexes, as printed in the source.
- SCORE2/-OP/-Diabetes values were transcribed twice independently (this package; RiskScorescvd)
  and agree exactly. `check_coefficients.py` also compares all 116 of them with the
  supplement text (files in `source/`), located by row label: 116/116 match. It
  flags a mutated value (checked by hand with four mutations).
- Not yet compared with the official ESC calculator (HeartScore) or charts
  (evidence level A).
- No published worked examples have been located yet.
