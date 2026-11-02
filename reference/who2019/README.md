# WHO 2019 — reference material

Status: implemented and verified (see `REPORT.md`). The earlier blocker,
unpublished recalibration factors, is resolved: WHORiskCalculator 1.1.0 (CRAN,
MIT) ships the authors' recalibration parameters from their Stata program
`whocvdrisk`. The draft email to the authors is no longer needed.

- `source/` — appendix 1 (model, Table 1.6) and appendix 2 (official charts).
- `extract_data.R` — exports recalibration parameters and the country map into
  `src/cvd_risk/who2019/`.
- `check_coefficients.py` — every coefficient against the PDF text.
- `parse_charts.py` — all 44,100 chart cells -> `fixtures/charts.csv`.
- `make_patients.py`, `run_reference.R` — synthetic patients, R answers.
- `report.py` -> `REPORT.md`.
