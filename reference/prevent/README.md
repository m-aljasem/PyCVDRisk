# PREVENT — reference material

- `source/` — the official supplement, both versions (original xlsx/pdf;
  revised 2024-01-26 xlsx and 2024-03-04 docx), copied from the owner's files.
- `read_xlsx.py` — stdlib xlsx -> TSV dumper used to read the supplement.
- `build_coefficients.py` — parses Tables S12.A-J into
  `src/cvd_risk/prevent/coefficients.csv` and `fixtures/worked_examples.csv`,
  checks both supplement versions agree, and applies the 2026-10-14 HbA1c erratum.
- `make_patients.py`, `run_reference.R` — synthetic patients and preventr answers.
- `report.py` -> `REPORT.md` — results, mutation check, findings.
