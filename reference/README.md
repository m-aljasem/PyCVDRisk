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
| PREVENT (5 models x 10/30 y x 5 outcomes) | verified; coefficients from primary source | authors' worked examples (100 values); preventr 0.12.2 | C + B | `prevent/REPORT.md` |
| WHO 2019 | **blocked: needs authors** (recalibration factors unpublished) | — | — | `who2019/README.md` |
| QRISK3 | **needs a decision** (see below) | ClinRisk code (LGPL) | — | — |

## Owner actions needed

1. **PREVENT HbA1c erratum** — open doi:10.1161/CIR.0000000000001491 (2026-10-14)
   in a browser, save the PDF to `reference/prevent/source/`, and confirm it swaps
   the "HbA1c in DM" / "HbA1c no DM" coefficients (inferred from preventr; see
   `prevent/REPORT.md`).
2. **SCORE2-CKD** — not in your Zotero library. Add Matsushita K et al., Eur J
   Prev Cardiol 2023, doi:10.1093/eurjpc/zwac176 (PMC9839538) with its supplement.
   Also missing from Zotero: SCORE2-OP (Eur Heart J 2021, doi:10.1093/eurheartj/ehab312).
3. **WHO 2019** — the recalibration factors are not published (see `who2019/README.md`).
4. **QRISK3** — the BMJ 2017 paper does not print the full model; the only
   complete source is ClinRisk's LGPL code. Implementing from it means either
   accepting LGPL obligations for that module or getting written permission
   from ClinRisk to publish an MIT re-implementation. Your call.

Automated downloads of 1 and 2 were refused by the publishers or behind a browser
challenge; they were not worked around.
