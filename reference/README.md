# Verification status

Each score lives in `src/cvd_risk/<family>/`; its evidence lives here.
Regenerate any family: `python reference/<family>/make_patients.py`,
`Rscript reference/<family>/run_reference.R`, `python reference/<family>/report.py`.

| Score | Status | Reference | Evidence | Report |
|---|---|---|---|---|
| SCORE2 | verified | RiskScorescvd 0.3.1 (R); paper worked example | B + C | `score2/REPORT.md` |
| SCORE2-OP | verified | RiskScorescvd 0.3.1 | B | `score2/REPORT.md` |
| SCORE2-Diabetes | verified | RiskScorescvd 0.3.1 | B | `score2/REPORT.md` |
| SCORE2 Asia-Pacific | verified; scales transcribed from primary source | RiskScorescvd 0.3.1 | B | `score2/REPORT.md` |
| Globorisk (lab, office, fatal) | verified | authors' R package 1.0.2 | A | `globorisk/REPORT.md` |
| SCORE2-CKD | **blocked: needs source PDF** | RiskScorescvd (ready) | — | — |
| PREVENT | **blocked: needs source xlsx**; reference answers ready | preventr 0.12.2 | — | `prevent/README.md` |
| WHO 2019 | **blocked: needs authors** (recalibration factors unpublished) | — | — | `who2019/README.md` |
| QRISK3 | **needs a decision** (see below) | ClinRisk code (LGPL) | — | — |

## Owner actions needed

1. **PREVENT** — your Zotero has only the main article (no coefficients; they
   are in the supplement). In a browser, open https://pmc.ncbi.nlm.nih.gov/articles/PMC10910659/
   and save "Supplemental Tables" (xlsx) as `reference/prevent/source/Supplemental_Tables.xlsx`.
2. **SCORE2-CKD** — not in your Zotero library. Add Matsushita K et al., Eur J
   Prev Cardiol 2023, doi:10.1093/eurjpc/zwac176 (PMC9839538) with its supplement.
   Also missing from Zotero: SCORE2-OP (Eur Heart J 2021, doi:10.1093/eurheartj/ehab312).
3. **WHO 2019** — the recalibration factors are not published (see `who2019/README.md`).
4. **QRISK3** — the BMJ 2017 paper does not print the full model; the only
   complete source is ClinRisk's LGPL code. Implementing from it means either
   accepting LGPL obligations for that module or getting written permission
   from ClinRisk to publish an MIT re-implementation. Your call.

Automated downloads of 1, 2 were refused by the publishers or behind a browser
challenge; they were not worked around.
