# cvd-risk

Verified, vectorised Python implementations of cardiovascular risk scores.

Every score is implemented from its original publication, every coefficient
cites its source, and every score is checked against an independent reference
implementation before it is released. See `reference/<score>/REPORT.md`.

```python
from cvd_risk import score2
score2(age=[50, 62], sex=["male", "female"], smoker=[1, 0], sbp=[140, 130],
       total_chol=[5.5, 6.1], hdl_chol=[1.3, 1.6], diabetes=0, region="low")
# array([0.0591..., 0.0379...])   10-year risk as a fraction
```

| Score | Status | Evidence |
|---|---|---|
| SCORE2, SCORE2-OP, SCORE2-Diabetes, SCORE2 Asia-Pacific, SCORE2-CKD | implemented, verified vs the official ESC and Asia-Pacific charts (4,091/4,096 cells), R `RiskScorescvd` and the published worked examples; SCORE2-Diabetes also vs U-Prevent, SCORE2-CKD vs the developers' CKD-PC online calculator | A + B (+ C for SCORE2, SCORE2-Diabetes, SCORE2-CKD) |
| Globorisk (lab, office, fatal; 182 countries) | implemented, verified vs the authors' R package | A |
| PREVENT (base, UACR, HbA1c, SDI, full; 10/30-year; 5 outcomes) | implemented, verified vs the official AHA online calculator (34 patients, 189/189 displayed risks), the authors' worked examples and R `preventr` | A + C + B |
| WHO 2019 (lab, non-lab; 21 regions / 190 countries) | implemented, verified vs every cell of the official WHO charts and R `WHORiskCalculator` | A + B |
| QRISK2-2014 | implemented (LGPL-3.0 module), verified vs ClinRisk's own C code, validation included (30,000 patients, 2.9e-13) | A |
| QRISK3-2017 | implemented (LGPL-3.0 module), verified vs ClinRisk's own C code and R `QRISK3` | A + B |
| Pooled Cohort Equations (ACC/AHA 2013) | implemented, verified vs the guideline's Table A and R `CVrisk` / `PooledCohort` | A + C + B |
| Framingham 2008 general CVD (lab, BMI) | implemented, verified vs the Framingham Heart Study's own calculator spreadsheets, the paper's worked examples and R `CVrisk` | A + C + B |
| SCORE (2003), fatal CVD, total-cholesterol model | implemented, verified vs the 800 cells of the official charts (798 within printed-coefficient precision; 2 chart anomalies documented); no working third-party implementation exists | A |
| SMART-REACH (10-year risk, CVD-free life expectancy) | implemented, verified vs the authors' calculator spreadsheet (600 patients, 3e-15); a defect in the calculator's life expectancy documented | A |
| LIFE-CVD2 (lifetime risk to 80, CVD-free life expectancy, treatment benefit; 4 ESC regions) | implemented, verified vs the authors' 16 published charts (7,167/7,168 cells within printed precision); a defective chart row documented | A |
| DIAL2 core model (type 2 diabetes: 10-year and lifetime risk, CVD-free life expectancy; low/moderate ESC regions) | implemented, verified vs the official U-Prevent calculator (20 patients, 58/60 displayed values; the 2 others were mis-copied) and the paper's two worked examples (6/6 printed values; the caption's eGFR for example B is 90, not 70) | A + C |
| SMART2 (established atherosclerotic disease: 10-year recurrent MI, stroke or vascular death; ages 40-80; 4 European and 5 other regions) | implemented, verified vs the official U-Prevent calculator (50/50 comparable patients at its 0.1% display); the supplement's SBP coefficient is per 10 mmHg and a constant of 0.211 (= -ln 0.81) is missing from it, both recovered from the calculator | A |
| Reynolds Risk Score (women 2007, men 2008) | implemented from the printed formulas (19/19 terms quoted from the PDFs), checked vs the papers' 17 worked examples (14 round exactly, all within printed precision); no calculator or third-party implementation available | C |
| FINRISKI 2.0 calculator (Finland 2020; 10-year CHD, stroke, combined; ages 30-74) | equations equal the article's Table 1 (4/4); matches the official Finnish calculator on 560 values (max relative difference 3e-16; 40 more are floored by its display); the calculator floors its display at an "optimal" profile, the package does not | A + C |
| Abu Dhabi Risk Study, ADRS (UAE 2024; 10-year ASCVD, hard CAD, stroke; ages 18+) | implemented from the paper's Table 2 and the supplement's SPSS syntax (32/32 terms agree); no worked example or calculator exists, so evidence is weak; the unprinted cohort GFR mean (111.4) is derived from Table 1, and of the supplement's two stroke formulas the one matching the printed mean score is used | C (weak) |
| PREDICT, New Zealand (7 equations: primary prevention 2018 and with BMI, VARIANZ policy, type 2 diabetes, major bleeding, after CVD, after ACS) | implemented from the authors' R package PredictRiskScores; all coefficients identical to its source, 49/49 printed table rows and both worked examples of the 2018 and bleeding papers reproduced; the authors' R code run unmodified (21,000 patients, max difference 5e-13); 6 defects in the R code documented | A + C |
| ASSIGN 1.0 (Woodward 2007) and 2.0 (Welsh 2025) (Scotland, 10-year CVD, ages 30-74; SIMD deprivation) | version 2.0: official NHS Scotland calculator, 200/200 displayed scores reproduced and its 32 constants identical; version 1.0: every constant found in the 2007 appendix; a printed typo in the 2025 equations (0.2286 for 0.22286) confirmed by the calculator code | A (2.0), C (1.0) |
| PROCAM 2007 (coronary: men and women 20-75, points and 10-year risk category; ischaemic stroke/TIA: 35-65) | Tables 2-4 and 6-7 checked against the PDF: 50/50 point bands at both ends, 95/95 age rows of the risk tables, 39/39 stroke risks; Table 4 prints an overlapping row (women 46) | C |
| REGICOR (Wilson 1998 Framingham coronary function calibrated to Girona, Spain; Rev Esp Cardiol 2003; ages 35-74) | official regicor.cat charts: men 198/200 cells (misses within 0.01 points), women every cell within 1 point; Wilson 1998 Table 6 found verbatim in the PDF; Rev Esp Cardiol Table 1 18/19 rows (1 printed sign differs from Wilson, documented); 40/40 bands; both worked examples reproduced | A + C |
| CUORE Project individual score (Italy 2007; 10-year first major CVD event, ages 35-69) | official ISS online calculator: 300/300 displayed values reproduced; Table 2 hazard ratios 13/14 (the men's HDL HR is misprinted); the calculator rounds the mean of its two SBP readings half to even | A + C |
| China-PAR (China 2016; 10-year ASCVD with waist, north/south and urban/rural) | official cvdrisk.com.cn calculator: 550/550 displayed values reproduced; coefficients recovered within 0.005 of the paper's 2-decimal values (a set from 400 patients matched 144/150 held out, the rest 0.1 point off at a rounding edge); at face value the printed values miss the calculator by up to 12.7 points | A + C |
| Suita score (Japan 2014; 10-year CHD points with CKD, total- and LDL-cholesterol sheets) | point sheets checked against the paper's Cox coefficients (LDL sheet all equal to 10 x beta; 3 TC-sheet entries differ, documented); one baseline survival fits every row of both probability tables; no calculator for the 2014 sheets exists; two web calculators of the JAS 2017 adaptation agree on every shared item | C (+ partial B) |
| WHO/ISH charts (2007; 14 subregions, with and without cholesterol) | implemented as the chart lookup, verified vs every cell of the guideline's two printed charts (768/768) and R `whoishRisk` (10,749/10,752 cells; the 3 others are whoishRisk transcription errors corrected from the guideline) | A + B |
| HEART (chest pain, 0-10) | implemented from the published point rules (18/18), verified vs R `RiskScorescvd` (5,000 patients identical) | C + B |
| CHADS2, CHA2DS2-VASc, HAS-BLED (atrial fibrillation) | implemented from each paper's point table, 32/32 published rules and Gage's worked example; no independent implementation used | C |
| TIMI UA/NSTEMI (0-7) | implemented from the paper (11/11 rules), verified vs R `RiskScorescvd` (5,000 patients identical) | C + B |
| EDACS (chest pain) | implemented from the paper (29/29 rules), verified vs R `RiskScorescvd` (5,000 patients identical) | C + B |
| GRACE in-hospital (Granger 2003) and 6-month post-discharge (Eagle 2004) | implemented from the published nomograms, 78/78 bands checked at both ends against the PDFs; Granger's worked examples disagree with its own table (documented; the authors confirmed the table); the 6-month score also matches the developers' archived GRACE calculator on all 87,808 band combinations | 6-month A + C; in-hospital C |
| GRACE 2.0 (Fox 2014; 1- and 3-year death, and death or MI, after ACS admission; with substitutions for creatinine and Killip class) | spline coefficients from the developers' calculator code; verified vs that code run unmodified (24,572/24,572 displayed values) and Fox 2014 Table 2 (24/24 hazard ratios) | A + C |

Evidence: **A** the developers' own code, calculator or charts; **B** an independent package; **C** the paper's worked examples and tables. Details per score: `reference/README.md`.

### One call for any country

`uniscore` gives each person the score their country's guideline body recommends:
PREVENT (USA), QRISK3 (UK), SCORE2/SCORE2-OP (the 55 ESC countries of SCORE2 Fig 5),
SCORE2 Asia-Pacific (48 countries) and WHO 2019 everywhere else. The routing table,
with a source for every country, is `src/cvd_risk/uniscore/routing.csv`.

```python
from cvd_risk import uniscore, guideline_score
uniscore(["USA", "DEU", "JPN", "BRA"], age=60, sex="female", sbp=135, total_chol=5.8,
         hdl_chol=1.4, smoker=0, diabetes=0, bp_treated=1, statin=0, egfr=85, bmi=29)
guideline_score(["USA", "DEU", "JPN", "BRA"])  # prevent, score2, score2_asia_pacific, who2019
```
