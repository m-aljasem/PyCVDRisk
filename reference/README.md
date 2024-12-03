# Verification status

Each score lives in `src/cvd_risk/<family>/`; its evidence lives here.
Regenerate any family: `python reference/<family>/make_patients.py`,
`Rscript reference/<family>/run_reference.R` (QRISK3: `python reference/qrisk3/run_reference.py`
then `Rscript reference/qrisk3/run_reference_cran.R`), `python reference/<family>/report.py`.

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
| QRISK2-2014 | verified; module is LGPL-3.0 | ClinRisk C source compiled unmodified, incl. its input validation | A | `qrisk2/REPORT.md` |
| QRISK3-2017 | verified; module is LGPL-3.0 (see below) | ClinRisk C source compiled unmodified; QRISK3 0.6.0 (R) | A + B | `qrisk3/REPORT.md` |
| Pooled Cohort Equations 2013 | verified; 46/46 values vs Table A | guideline worked example; CVrisk 1.2.1; PooledCohort 0.0.2 | C + B | `pce/REPORT.md` |
| Framingham 2008 (lab, BMI) | verified; 32/32 values vs paper | FHS calculator spreadsheets; paper worked examples; CVrisk 1.2.1 | A + C + B | `framingham/REPORT.md` |
| SCORE 2003 (total-cholesterol model) | verified; 22/22 values vs paper; 798/800 chart cells within printed precision | official charts (Figs 1-2) | A | `score/REPORT.md` |
| SMART-REACH (10-year risk, CVD-free life expectancy) | verified; 24/24 coefficients vs calculator; calculator life-expectancy defect found | authors' calculator spreadsheet recalculated by LibreOffice | A | `smart_reach/REPORT.md` |
| LIFE-CVD2 (lifetime risk, life expectancy, treatment benefit) | verified; 36/36 coefficients vs supplement; 7,167/7,168 chart cells; centring and chart inputs inferred and pinned by the charts | authors' charts (Appendix 1, 16 charts) | A | `life_cvd2/REPORT.md` |
| DIAL2 core model | verified; 56/56 coefficients vs supplement (Word and PDF); weak evidence: 6 printed values | paper worked examples (Figure 3) | C | `dial2/REPORT.md` |
| WHO/ISH charts 2007 (14 subregions, with/without cholesterol) | verified; 3 whoishRisk transcription errors corrected | guideline's 2 printed charts (768 cells); whoishRisk 1.0 table (10,752 cells) and its R lookup logic (transliterated, not run) | A (2 subregions) + B | `who_ish/REPORT.md` |
| HEART | verified; 18/18 published rules | Melki & Jernberg 2013 (Table 1, Methods); RiskScorescvd 0.3.1 | C + B | `heart/REPORT.md` |
| CHADS2, CHA2DS2-VASc, HAS-BLED | verified; 32/32 published rules, definitions quoted from the PDFs | Gage 2001; Lip 2010; Pisters 2010 | C | `af/REPORT.md` |

## Owner actions needed

1. **QRISK licence** — the papers do not print the full models, so
   `src/cvd_risk/qrisk2/` and `src/cvd_risk/qrisk3/` are built from ClinRisk's LGPL C code and are themselves
   LGPL-3.0-or-later (licence texts in that folder); the rest stays MIT. If you
   want the whole package MIT, ask ClinRisk for written permission; otherwise
   declare the package licence as `MIT AND LGPL-3.0-or-later`.
2. **Reynolds Risk Score — sources needed.** Both papers (Ridker 2007 JAMA 297:611; Ridker
   2008 Circulation 118:2243) return 403 to automated download and are not in Zotero as files,
   and the official site (reynoldsriskscore.org) is now a parked domain. v1's coefficients do not
   match the published formula. Please add both PDFs to `Formulas/` or Zotero.
3. **SMART2 — question for the authors (Hageman, Dorresteijn, Visseren; UMC Utrecht).** The
   supplement (Table S2) gives the coefficients for uncentred predictors, a 10-year baseline risk
   of 0.1658 and a "mean linear predictor" of -0.0464. At the derivation cohort's average values
   (Table 1) the linear predictor is about 2.25, so the printed formula would give an average
   patient a 10-year risk near 84%, against an observed incidence near 20%. The -0.0464 only makes
   sense for centred predictors, and the centring values are not published. Question to send:
   *which value of LP_mean (or which centring constants) should be used with the coefficients
   in Supplementary Table 2?* U-Prevent.com presumably has the answer.
4. **SMART-REACH — optional note to the authors.** Their calculator's life expectancy is wrong
   whenever a person's yearly risks sum to more than 1 at some age (see `smart_reach/REPORT.md`
   section 3); in some cases it is below the person's current age.
5. **LIFE-CVD2 — optional note to the authors.** In all 16 charts of Appendix 1 the row
   "age 65-69, SBP 100-119" is computed at age 62 instead of 67 (see `life_cvd2/REPORT.md`). The
   supplement also does not state the predictor centring or the chart HDL/LDL values; the report
   shows which values the charts imply.
6. **DIAL2 — optional note to the authors.** Figure 3B is reproduced with eGFR 90, not the 70 in
   its caption and the text (see `dial2/REPORT.md`). A few model outputs from u-prevent.com
   (or from the authors) would raise DIAL2's evidence from level C to A.
7. **WHO/ISH 2007: regional charts needed.** The guideline prints only the SEAR-D and EMR-B charts.
   The other 12 subregions currently rest on whoishRisk's table, which has 3 errors in SEAR-D.
   The WHO/ISH pocket guidelines (WHO IRIS, 2007; one per subregion) hold the official charts,
   but IRIS timed out here. Please add the 12 PDFs to `Formulas/WHO/`. The report then checks
   every cell, including the 10 cells where a higher risk factor gives a lower band.
8. **Sources blocked, please add to `Formulas/` or Zotero.**
   - **SMART (2013):** Dorresteijn, Heart 2013, online supplements 2 (calculation sheet) and 6
     (baseline survival and centring). The paper gives only the coefficients.
   - **QRISK1:** the paper prints rounded hazard ratios only, with no means or baseline survival.
     The full model (means, baseline survival) is needed from ClinRisk or the authors.
   - **QRISK4:** the software is available only under an academic licence from Oxford University
     Innovations.
   - **Group 2 papers (from v1):** ASSIGN, PROCAM, REGICOR, Progetto Cuore, FINRISK, PREDICT, Jackson
     Heart, ARIC, CARDIA, Singapore, Malaysian, Gulf RACE, Dundee, Cambridge, EPIC-Norfolk, Heinz
     Nixdorf, Rotterdam, INTERHEART, D:A:D, RISC, New Zealand. Europe PMC lists none as open access.
   - **Point scores (from v1):** TIMI UA/NSTEMI (Antman, JAMA 2000), EDACS (Than, Emerg Med
     Australas 2014), GRACE (Granger, Arch Intern Med 2003; Fox, BMJ 2006), None is open access.
     RiskScorescvd implements TIMI, EDACS and GRACE, but with dataset-specific inputs (fixed
     hs-troponin I limits, previous PCI/CABG standing in for known CAD). Building from it alone would
     make the check circular, so these wait for the papers. HEART is done.
9. **Redistribution of `reference/*/source/`.** These folders hold copies of papers, supplements,
   charts and third-party code, e.g. `who_ish/source/whoishRisk/` (GPL-3). Check that you may
   publish them, or drop them from the public repository and keep only checksums and URLs.
