# Jcvrisk 0.1.3 (R): comparison with the papers

Jcvrisk (CRAN 2025-10-29; Okumiyama H, Fujii R; MIT) implements 14 Japanese risk functions. Its sources are vendored
unmodified in `reference/vendor/Jcvrisk` (all files match the package's MD5 list). `make_patients.py` writes 3,000
seeded patients (`fixtures/patients.csv`, integer blood pressure and lipids so category edges occur);
`run_jcvrisk.R` runs every function on them (`fixtures/jcvrisk_output.csv`; for JALS also the total score).

| Jcvrisk function | Package | Result |
|---|---|---|
| `hisayama` | `hisayama_points` + `hisayama_points_risk` | equal, 2,702/2,702 aged 40-84 (1e-12) |
| `suita_wECG`, `suita_woECG` | `suita_cvd` + `suita_cvd_risk` | same band 2,409/2,416 and 2,411/2,416; differences only at SBP 139 / DBP 89 (its cut points leave them uncategorised) |
| `epoch_stroke`, `epoch_cvd` | `epoch_japan` | equal, 2,416/2,416 aged 40-79 (1e-12) |
| `epoch_chd` | `epoch_japan(outcome="chd")` | differs for every patient: it enters ln(TC) for ln(TC/HDL-C) |
| `jals_*_wECG`, `jals_*_woECG` (8) | `jals` + `jals_risk` | scores equal for 32-70% of patients: blood-pressure categorisation and several points differ from the paper (`reference/jals/REPORT.md`) |

The tests that use these fixtures are in `tests/hisayama`, `tests/epoch_japan` and `tests/suita/test_suita_cvd.py`.
