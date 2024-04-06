# PREVENT — reference answers (implementation pending)

Status: **blocked on the primary source.** The coefficients must be transcribed
from the PREVENT Data Supplement (Khan SS et al., Circulation 2024;149:430-449,
doi:10.1161/CIRCULATIONAHA.123.067626, "Supplemental Tables" xlsx), not from
`preventr`, otherwise verifying against `preventr` would be circular.
Automated download is blocked (publisher 403; PubMed Central serves the file
behind a browser challenge; Europe PMC API: "not open access"). Download it
in a browser from https://pmc.ncbi.nlm.nih.gov/articles/PMC10910659/ and save
it as `reference/prevent/source/Supplemental_Tables.xlsx`.

Ready now:
- `make_patients.py` — 5,000 seeded patients (every whole age 30-79).
- `run_reference.R` — preventr 0.12.2 answers, rounding disabled, for the base,
  HbA1c and UACR models, 10- and 30-year, 5 outcomes (`fixtures/`).

Observations about the reference (to report once confirmed):
1. preventr returns NA for ages 79.01-79.99 (treats the 30-79 range as whole
   years); fractional ages just under 80 cannot be scored.
2. preventr returns 30-year risks for ages 60-79, while the PREVENT paper
   restricts 30-year estimates to ages 30-59.
