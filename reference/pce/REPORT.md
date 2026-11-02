# Pooled Cohort Equations (2013 ACC/AHA) — verification report

Generated 2026-11-01 by `reference/pce/report.py`.

## 1. Source

Goff et al., 2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk,
Appendix 7 Table A (author manuscript PMC4700825, saved in `source/`).
`check_coefficients.py` parses Table A from that file: all 46 values (coefficients,
means, baseline survivals) equal `coefficients.py`.

**Worked example (Table A/B).** For the guideline's example patient the package gives
2.1%, 3.0%, 5.4% and 6.1% (White women, African American women, White men, African
American men); the guideline prints 2.1, 3.0, **5.3** and 6.1. For White men the exact
individual sum is 60.6995 but the table prints 60.69 (truncated), and Table B computes
1 - 0.9144^exp(60.69 - 61.18) = 5.33%. With the exact sum the risk is 5.38%. The
difference is the guideline's rounding of an intermediate value, not the equation.

## 2. Agreement with the references

```
reference: CVrisk 1.2.1 (rounding and 1% floor removed)
secondary: PooledCohort 0.0.2 (equation_version = Goff_2013)
R: R version 4.3.3 (2024-02-29)
generated: 2026-11-01
```

20,000 seeded synthetic patients (every whole age 40-79, both sexes and races,
treated and untreated SBP, within CVrisk's accepted lipid and SBP ranges).

| Reference | Rows compared | Max rel diff |
|---|---|---|
| CVrisk 1.2.1 (rounding removed) | 19,599 | 2.2e-13 |
| PooledCohort 0.0.2, as is | 20,000 | 1.0e-02 (median 1.1e-03) |
| PooledCohort 0.0.2, its constants substituted | 20,000 | < 1e-12 (test) |

1. **CVrisk returns NA for ages 79.01-79.99** (it requires age <= 79). The guideline's
   "40 to 79 years" is read here as [40, 80), as for SCORE2 and PREVENT. CVrisk also
   floors risks below 1% to 1% and accepts ages from 20; the guideline does neither.
2. **PooledCohort uses more decimals than the guideline prints** for the baseline
   survivals, means, and all African American women coefficients (e.g. mean 86.6081
   vs 86.61). This shifts risks by up to 1% relative. With its constants substituted,
   this package matches it to < 1e-12, so the equations agree and only the constants
   differ. Their provenance (presumably the ACC/AHA spreadsheet) is not documented; this
   package uses the published values.

Scoring 20,000 rows: 0.019 s; 1,000,000 rows: 1.97 s.

## 3. Mutation check

Every value in Table A, +1 in its last printed digit: 46/46 caught by CVrisk.

## 4. Open items

- Compare against the official ACC/AHA ASCVD Risk Estimator (evidence level A); its
  site did not respond to automated requests.
