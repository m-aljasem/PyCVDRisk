# Globorisk — verification report

Generated 2026-11-01 by `reference/globorisk/report.py`.

## 1. Agreement with the reference

Reference: the authors' own R package `globorisk` (evidence level A).

```
reference: globorisk (authors' R package), vendored at reference/vendor/globorisk
package_version: 1.0.2
R: R version 4.3.3 (2024-02-29)
generated: 2026-11-01
options: updated_lac = FALSE (default)
```

30,000 seeded synthetic patients: 183 countries, baseline years 2000-2020, every
whole age 40-80 plus fractional ages, horizons 1-10 years (70% at 10).

| Version | Patients | NaN (both) | NaN pattern identical | Max rel diff | Time (s) |
|---|---|---|---|---|---|
| lab | 30,000 | 157 | yes | 4.2e-13 | 0.063 |
| office | 30,000 | 157 | yes | 3.6e-13 | 0.046 |
| fatal | 30,000 | 157 | yes | 1.8e-12 | 0.039 |

All NaN rows are South Sudan (SSD): it has baseline rates but no population
mean risk factors in the source data, so neither implementation can score it.

## 2. Coefficient mutation check

Each coefficient multiplied by (1 + 1e-06); caught if the reference comparison fails.

| Version | Coefficients | Caught |
|---|---|---|
| lab | 10 | 10 |
| office | 7 | 7 |
| fatal | 10 | 10 |

## 3. Data provenance

Coefficients, country baseline hazards and population means come from the
authors' package data (MIT). `extract_data.R` proves the package's 8.9 M
per-follow-up-year hazards reduce exactly (zero conflicts) to 1.06 M rates per
(outcome, country, sex, calendar year, attained age); `build_package_data.py`
packs them into `src/cvd_risk/globorisk/data.npz` (1.3 MB).

## 4. Findings

1. **Smoking lowers risk at the oldest ages.** Lab model smoking log-HR is
   1.8467 - 0.02205 x attained age, negative after ~83.7 years, so a 79-year-old
   smoker's 10-year risk can be below a non-smoker's. Same in the reference.
2. **Reference contradicts its own warning for age > 80.** It warns these ages
   "will be returned as NAs" but returns values when the horizon is short.
   This package follows the documented rule (NaN at age >= 81).
3. **Possible bug in the reference's optional LAC update (office version):** the
   term `main_sbpsexc` (an SBP x sex interaction by name) is multiplied by
   sex x smoking instead of sex x SBP. Not used by default (`updated_lac = FALSE`);
   the LAC update is not implemented here. To confirm with the authors.

## 5. Open items

- Check rounded coefficients against the papers' printed tables.
- Reproduce published country risk charts (Ueda 2017) cell by cell.
- LAC-recalibrated equations (`updated_lac`) not implemented.
