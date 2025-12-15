# Verification status

Each score lives in `src/cvd_risk/<family>/`; its evidence lives here.
Regenerate any family: `python reference/<family>/make_patients.py`,
`Rscript reference/<family>/run_reference.R` (QRISK3: `python reference/qrisk3/run_reference.py`
then `Rscript reference/qrisk3/run_reference_cran.R`), `python reference/<family>/report.py`.

| Score | Status | Reference | Evidence | Report |
|---|---|---|---|---|
| SCORE2 | verified; 1,532/1,536 official chart cells within printed precision (1,529 exact) | ESC 2021 guideline charts (4 regions); RiskScorescvd 0.3.1 (R); paper worked example | A + B + C | `score2/REPORT.md` |
| SCORE2-OP | verified; 1,024/1,024 official chart cells within printed precision (852 exact) | ESC 2021 guideline charts = authors' supplement print; RiskScorescvd 0.3.1 | A + B | `score2/REPORT.md` |
| SCORE2-Diabetes | verified; U-Prevent calculator 21/22 displayed values (the other 0.0003 points from a rounding edge; its form accepts ages 55-90 only); 7/8 printed worked examples (1 misprint documented) | U-Prevent (browser run by the owner); RiskScorescvd 0.3.1; paper worked examples | A + B + C | `score2/REPORT.md` |
| SCORE2 Asia-Pacific | verified; 1,535/1,536 official chart cells exact (1 print slip) | authors' charts (Figure 3, 4 regions); RiskScorescvd 0.3.1 | A + B | `score2/REPORT.md` |
| Globorisk (lab, office, fatal) | verified | authors' R package 1.0.2 | A | `globorisk/REPORT.md` |
| SCORE2-CKD (eGFR, ACR, dipstick add-ons) | verified; 1 reference bug found; the developers' calculator drops the SCORE2-OP eGFR >= 90 term | CKD-PC online calculator (568 add-on values, eGFR and ACR add-ons; 71 patients); supplement worked example; RiskScorescvd 0.3.1 | A + C + B | `score2_ckd/REPORT.md` |
| PREVENT (5 models x 10/30 y x 5 outcomes) | verified; coefficients from primary source; AHA online calculator 189/189 displayed risks (34 patients; base, UACR, HbA1c and full equations) | AHA online calculator (browser run by the owner); authors' worked examples (100 values); preventr 0.12.2 | A + C + B | `prevent/REPORT.md` |
| WHO 2019 (lab, non-lab; 21 regions, 190 countries, 1990-2017) | verified | all 44,100 official chart cells; WHORiskCalculator 1.1.0 | A (charts) + B | `who2019/REPORT.md` |
| QRISK2-2014 | verified; module is LGPL-3.0 | ClinRisk C source compiled unmodified, incl. its input validation | A | `qrisk2/REPORT.md` |
| QRISK3-2017 | verified; module is LGPL-3.0 (see below) | ClinRisk C source compiled unmodified; QRISK3 0.6.0 (R) | A + B | `qrisk3/REPORT.md` |
| Pooled Cohort Equations 2013 | verified; 46/46 values vs Table A; official ACC calculator code run on 20,000 patients: identical equations (max rel diff < 1e-12 with its constants), its 14 extra-decimal constants shift risk by up to 0.49 points (12 patients cross 7.5%) | ACC CVD Risk Estimator+ JS (run); guideline worked example; CVrisk 1.2.1; PooledCohort 0.0.2 | A + C + B | `pce/REPORT.md` |
| Framingham 2008 (lab, BMI) | verified; 32/32 values vs paper | FHS calculator spreadsheets; paper worked examples; CVrisk 1.2.1 | A + C + B | `framingham/REPORT.md` |
| SCORE 2003 (total-cholesterol model) | verified; 22/22 values vs paper; 798/800 chart cells within printed precision | official charts (Figs 1-2) | A | `score/REPORT.md` |
| SMART-REACH (10-year risk, CVD-free life expectancy) | verified; 24/24 coefficients vs calculator; calculator life-expectancy defect found | authors' calculator spreadsheet recalculated by LibreOffice | A | `smart_reach/REPORT.md` |
| LIFE-CVD2 (lifetime risk, life expectancy, treatment benefit) | verified; 36/36 coefficients vs supplement; 7,167/7,168 chart cells; centring and chart inputs inferred and pinned by the charts | authors' charts (Appendix 1, 16 charts) | A | `life_cvd2/REPORT.md` |
| DIAL2 core model | verified; 56/56 coefficients vs supplement (Word and PDF); U-Prevent calculator 58/60 displayed values (20 patients; 2 mis-copied); 6/6 printed values | U-Prevent (browser run by the owner); paper worked examples (Figure 3) | A + C | `dial2/REPORT.md` |
| SMART2 (10-year recurrent ASCVD risk) | verified; coefficients, baselines and region ratios from Supplementary Table 2; U-Prevent calculator 50/50 comparable patients (2 runs; 2 more use treatment options outside SMART2); SBP per 10 mmHg and a missing constant of 0.211 recovered from the calculator | U-Prevent (browser run by the owner); Hageman 2022 Supplementary Table 2 | A | `smart2/REPORT.md` |
| Reynolds Risk Score (model B, women and men) | verified; 19/19 formula terms vs PDFs; 17/17 worked examples within printed precision (14 exact) | Ridker 2007 (Table 6); Ridker 2008 (Clinical Examples) | C | `reynolds/REPORT.md` |
| FINRISKI 2.0 (CHD, stroke, combined) | 4/4 equations vs Table 1; official calculator: 300 patients, 560 values at max rel diff 3e-16 (40 more floored by the calculator's display at its optimal profile); replaces FINRISK 2016 | Duodecim calculator (server, run by the owner); Vartiainen 2020 Table 1 | A + C | `finrisk/REPORT.md` |
| ADRS (Abu Dhabi 2024; ASCVD, CAD, stroke) | 32/32 terms: Data S1 syntax = Table 2; GFR mean, MAP and stroke mean score checked against Table 1 | JAHA 2024 Table 2, Data S1 (PMC11681588) | C (weak) | `adrs/REPORT.md` |
| PREDICT (NZ): 2018 primary prevention, BMI, VARIANZ policy, T2D, bleeding, post-CVD, post-ACS | verified; coefficients identical to the authors' R source (7/7 models); authors' R code run unmodified: 21,000 patients, max diff 5e-13; 49/49 paper rows; 2/2 worked examples | PredictRiskScores 0.9 (VIEW team); Pylypchuk 2018; Selak 2019 | A + C | `predict/REPORT.md` |
| ASSIGN 1.0 and 2.0 | verified; v2.0 **official calculator: 200/200 displayed scores reproduced**, its 32 constants equal ours; v1.0 constants vs the 2007 appendix; printed 0.2286 is a typo (calculator uses 0.22286) | Woodward 2007; Welsh 2025; rightdecisions.scot.nhs.uk | A (v2), C (v1) | `assign/REPORT.md` |
| PROCAM 2007 (coronary Weibull score, men and women; stroke score) | verified; 50/50 Table 2 bands, 95/95 rows of Tables 3-4, 39/39 Table 7 values vs PDF; 1 printed overlap (Table 4, women 46) and the "<" boundaries documented; replaces PROCAM 2002 | Assmann 2007 (Tables 2-7) | C | `procam/REPORT.md` |
| REGICOR (Wilson 1998 + Girona calibration, Rev Esp Cardiol 2003) | **official charts reproduced for men** (198/200, misses within 0.01 points); women within 1 point in every cell (their 65-74 panel repeats 55-64); Wilson Table 6 verbatim; Rev Esp Table 1 18/19 rows (men's optimal-BP sign differs, Wilson followed); 40/40 bands; Wilson and Rev Esp worked examples | Wilson 1998 (Table 6, Appendix); Marrugat, Rev Esp Cardiol 2003 (Table 1, Appendix); official charts (regicor.cat) |
| WHO/ISH charts 2007 (14 subregions, with/without cholesterol) | verified; 3 whoishRisk transcription errors corrected | guideline's 2 printed charts (768 cells); whoishRisk 1.0 table (10,752 cells) and its R lookup logic (transliterated, not run) | A (2 subregions) + B | `who_ish/REPORT.md` |
| HEART | verified; 18/18 published rules | Melki & Jernberg 2013 (Table 1, Methods); RiskScorescvd 0.3.1 | C + B | `heart/REPORT.md` |
| CHADS2, CHA2DS2-VASc, HAS-BLED | verified; 32/32 published rules, definitions quoted from the PDFs | Gage 2001; Lip 2010; Pisters 2010 | C | `af/REPORT.md` |
| TIMI UA/NSTEMI | verified; 11/11 published rules | Antman 2000 (Table 1); RiskScorescvd 0.3.1 | C + B | `timi/REPORT.md` |
| EDACS | verified; 29/29 published rules | Than 2014 (Box 3); RiskScorescvd 0.3.1 | C + B | `edacs/REPORT.md` |
| GRACE in-hospital, 6-month | verified; 78/78 bands vs the printed nomograms; worked-example discrepancy confirmed by the authors (table correct); 6-month score equals the developers' GRACE tool (2004/2013 builds, archived) on all 87,808 band combinations; the tool's in-hospital score is a different, later nomogram | Granger 2003 (Fig 4); Eagle 2004 (Fig 2); developers' ACS Risk Model tool | 6-month A + C; in-hospital C | `grace/REPORT.md` |
| GRACE 2.0 (1- and 3-year death, death or MI; full and substituted) | verified; developers' calculator code (acs_risk2, archived) run unmodified: 24,572/24,572 displayed values (6,143 dropdown combinations x 4 outcomes), its 836 lookup values reproduced, its grace.js to 1e-15 and its 4 worked examples; 24/24 hazard ratios of Fox 2014 Table 2 | developers' GRACE 2.0 calculator; Fox 2014 (Table 2) | A + C | `grace2/REPORT.md` |

## Owner actions needed

1. **QRISK licence** — the papers do not print the full models, so
   `src/cvd_risk/qrisk2/` and `src/cvd_risk/qrisk3/` are built from ClinRisk's LGPL C code and are themselves
   LGPL-3.0-or-later (licence texts in that folder); the rest stays MIT. If you
   want the whole package MIT, ask ClinRisk for written permission; otherwise
   declare the package licence as `MIT AND LGPL-3.0-or-later`.
2. **Reynolds Risk Score — done** (papers added). v1's coefficients did not match the
   published formula; v2 follows the papers. Evidence is level C only (worked examples).
3. **SMART2 — done; optional confirmation from the authors (Hageman, Dorresteijn, Visseren; UMC Utrecht).**
   The supplement's Table S2 does not reproduce U-Prevent as printed: the SBP coefficient is per 10 mmHg,
   and the linear predictor needs a further 0.211 (0.2104-0.2111 fits all 30 round-1 patients), which
   equals -ln(0.81), the antithrombotic offset. Both recovered from two U-Prevent runs (see
   `smart2/REPORT.md`).
4. **SMART-REACH — optional note to the authors.** Their calculator's life expectancy is wrong
   whenever a person's yearly risks sum to more than 1 at some age (see `smart_reach/REPORT.md`
   section 3); in some cases it is below the person's current age.
5. **LIFE-CVD2 — optional note to the authors.** In all 16 charts of Appendix 1 the row
   "age 65-69, SBP 100-119" is computed at age 62 instead of 67 (see `life_cvd2/REPORT.md`). The
   supplement also does not state the predictor centring or the chart HDL/LDL values; the report
   shows which values the charts imply.
6. **DIAL2 — optional note to the authors.** Figure 3B is reproduced with eGFR 90, not the 70 in
   its caption and the text (see `dial2/REPORT.md`). U-Prevent run done (level A); re-read the life
   expectancy of patients 18 and 19 by hand (the browser assistant copied the previous patient's).
7. **Dropped by the owner:** models older than about two decades are not pursued further (WHO/ISH
   regional charts, QRISK1). Implemented ones stay as they are.
8. **Sources still blocked, please add to `Formulas/` or Zotero.**
   - **SMART (2013):** Dorresteijn, Heart 2013, online supplements 2 (calculation sheet) and 6
     (baseline survival and centring). The paper gives only the coefficients.
   - **QRISK4:** the software is available only under an academic licence from Oxford University
     Innovations.
   - **Group 2 papers (from v1):** Progetto Cuore, Jackson Heart, ARIC, CARDIA,
     Singapore, Malaysian, Gulf RACE, Dundee, Cambridge, EPIC-Norfolk, Heinz Nixdorf, Rotterdam,
     INTERHEART, D:A:D, RISC, New Zealand. Europe PMC lists none as open access.
   - **GRACE 2.0: done** from the developers' calculator code (`grace2/REPORT.md`).
9. **Redistribution of `reference/*/source/`.** These folders hold copies of papers, supplements,
   charts and third-party code, e.g. `who_ish/source/whoishRisk/` (GPL-3). Check that you may
   publish them, or drop them from the public repository and keep only checksums and URLs.
10. **GRACE: done.** In Granger 2003 Figure 4, both worked examples score a
    value on a band's printed lower edge in the band below (SBP 100: 53, not 43; creatinine 0.4: 1,
    not 4), and example 1's terms add to 195, not 196. The package follows the printed table, and the
    authors confirmed the table is correct (personal communication).
11. **REGICOR — done.** Switched to the full-precision function of the official charts (Wilson 1998 + Rev Esp
    Cardiol 2003). Optional note to the REGICOR authors: Rev Esp Cardiol Table 1 prints men's optimal blood
    pressure as +0.0023 (Wilson: -0.00226), the Appendix step "4.496 - 3.489 = 0.986" should read 1.007, and the
    women's 65-74 chart panel is identical to 55-64. Optional: the shiny app (regicor.shinyapps.io/regicor2) for a
    continuous check of women.
12. **PCE — done** (ACC code run).
13. **PREDICT papers.** The Zotero attachment for Pylypchuk 2021 (Lancet, diabetes) is a different article
    (Cacciari, *Nomes de lugar: confim*, JSTOR). Please replace it. Mehta 2018 (VARIANZ, IJE) and Poppe 2017/2019
    (Heart; post-CVD, post-ACS) would let those models be checked against print too.
14. **PREDICT: optional note to the VIEW team** listing the R defects in `predict/REPORT.md` section 4
    (integer-sequence BMI/HbA1c/creatinine bands, `sex` case in the post-event models, the `acr` scoping bug).
15. **ADRS: optional note to the authors.** Ask them to confirm the cohort GFR mean (111.4, derived from
    Table 1), which of the two stroke formulas in Data S1 is intended, and for a few calculated risks.
16. **PREVENT: accept the AHA source-code licence.** The official PREVENT code is in a private GitHub repository
    (AHA-DS-Analytics/PREVENT), released free to researchers after accepting the licence at heart.org/PREVENT.
    With it, PREVENT can be run against the official code at full precision; level A is already reached through
    the AHA online calculator (189/189 displayed risks), which refuses automated requests.
17. **FINRISK — done** (official calculator run, 300 patients). The package now implements FINRISKI 2.0
    (Vartiainen, Suomen Lääkärilehti 2020;75:2778), the calculator's model. FINRISK 3.0 (Elonheimo, Eur J Public
    Health 2025;35 Suppl 4:ckaf161.018; Fine-Gray, LDL option, 30-year risk) is a conference abstract without
    coefficients; the calculator (en-GB and fi-FI are the same questionnaire and API) took total
    cholesterol only, gave 10-year risk only, and matched 2.0 to 3e-16, so it still ran 2.0. Re-run
    `finrisk/run_calculator.py` before submission; if values change, 3.0 has gone live and needs its paper.
18. **PROCAM 2007 — done** (paper added; the package now implements its point scores). Optional
    question for the Assmann-Stiftung: the paper's annex with the full Weibull function (not printed), whether
    Tables 3-4's "<" means "<=", and the overlapping Table 4 row (women 46: "63-69" and ">= 68").
19. **SCORE2-CKD: question for the authors (K. Matsushita, CKD-PC).** Their online calculator
    (ckdpcrisk.org/ckdpatchscore2) leaves out the SCORE2-OP eGFR >= 90 term that the
    supplement's equation prints (-0.4616, age x -0.0075); at age >= 70 and eGFR >= 90 it gives up to 28.9
    points less. Which is intended? The package follows the paper. Also: patient A's base 1.96% (the calculator
    itself shows 2.5%). The dipstick add-on is not in the calculator, so it stays at C + B.

## Evidence ceiling per score

Why the scores below A stay where they are:

| Score | Level | What would raise it |
|---|---|---|
| Reynolds | C | The official site (reynoldsriskscore.org) is now a parked domain; the papers' examples are the only official outputs. |
| PROCAM 2007 (points) | C | The full Weibull function is in an unpublished annex; the Assmann-Stiftung online test returns its continuous risk, which cannot check the point score cell by cell. Owner action 18. |
| ADRS | C (weak) | No calculator or charts found; owner action 15. |
| GRACE in-hospital (Granger 2003) | C | The developers' archived calculator (2004-2013) uses a later in-hospital nomogram, not Figure 4; today's online GRACE is GRACE 2.0, a different model (implemented, at A). The 6-month score (Eagle 2004) is at A. |
| HEART, TIMI, EDACS, CHADS2, CHA2DS2-VASc, HAS-BLED | C (+ B) | Point scores: the published rules are the definition, so C is the ceiling. |
