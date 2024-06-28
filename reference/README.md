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
| SCORE2-CKD (eGFR, ACR, dipstick add-ons) | verified; 1 reference bug found | supplement worked example; RiskScorescvd 0.3.1 | C + B | `score2_ckd/REPORT.md` |
| PREVENT (5 models x 10/30 y x 5 outcomes) | verified; coefficients from primary source | authors' worked examples (100 values); preventr 0.12.2 | C + B | `prevent/REPORT.md` |
| WHO 2019 (lab, non-lab; 21 regions, 190 countries, 1990-2017) | verified | all 44,100 official chart cells; WHORiskCalculator 1.1.0 | A (charts) + B | `who2019/REPORT.md` |
| QRISK3 | **needs a decision** (see below) | ClinRisk code (LGPL) | — | — |

## Owner actions needed

1. **PREVENT HbA1c erratum** — open doi:10.1161/CIR.0000000000001491 (2026-10-14)
   in a browser, save the PDF to `reference/prevent/source/`, and confirm it swaps
   the "HbA1c in DM" / "HbA1c no DM" coefficients (inferred from preventr; see
   `prevent/REPORT.md`).
2. **QRISK3** — the BMJ 2017 paper does not print the full model; the only
   complete source is ClinRisk's LGPL code. Implementing from it means either
   accepting LGPL obligations for that module or getting written permission
   from ClinRisk to publish an MIT re-implementation. Your call.

The automated download for 1 was refused by the publisher; it was not worked around.
