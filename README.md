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

| Score | Status |
|---|---|
| SCORE2, SCORE2-OP, SCORE2-Diabetes, SCORE2 Asia-Pacific, SCORE2-CKD | implemented, verified vs R `RiskScorescvd` (and published worked examples) |
| Globorisk (lab, office, fatal; 182 countries) | implemented, verified vs the authors' R package |
| PREVENT (base, UACR, HbA1c, SDI, full; 10/30-year; 5 outcomes) | implemented, verified vs the authors' worked examples and R `preventr` |
| WHO 2019 (lab, non-lab; 21 regions / 190 countries) | implemented, verified vs every cell of the official WHO charts and R `WHORiskCalculator` |
| QRISK2-2014 | implemented (LGPL-3.0 module), verified vs ClinRisk's own C code, validation included (30,000 patients, 2.9e-13) |
| QRISK3-2017 | implemented (LGPL-3.0 module), verified vs ClinRisk's own C code and R `QRISK3` |
| Pooled Cohort Equations (ACC/AHA 2013) | implemented, verified vs the guideline's Table A and R `CVrisk` / `PooledCohort` |
| Framingham 2008 general CVD (lab, BMI) | implemented, verified vs the Framingham Heart Study's own calculator spreadsheets, the paper's worked examples and R `CVrisk` |
| SCORE (2003), fatal CVD, total-cholesterol model | implemented, verified vs the 800 cells of the official charts (798 within printed-coefficient precision; 2 chart anomalies documented); no working third-party implementation exists |
| SMART-REACH (10-year risk, CVD-free life expectancy) | implemented, verified vs the authors' calculator spreadsheet (600 patients, 3e-15); a defect in the calculator's life expectancy documented |
| LIFE-CVD2 (lifetime risk to 80, CVD-free life expectancy, treatment benefit; 4 ESC regions) | implemented, verified vs the authors' 16 published charts (7,167/7,168 cells within printed precision); a defective chart row documented |
| DIAL2 core model (type 2 diabetes: 10-year and lifetime risk, CVD-free life expectancy; low/moderate ESC regions) | implemented, verified vs the paper's two worked examples (6/6 printed values; the caption's eGFR for example B is 90, not 70); no calculator or third-party implementation available |
| Reynolds Risk Score (women 2007, men 2008) | implemented from the printed formulas (19/19 terms quoted from the PDFs), checked vs the papers' 17 worked examples (14 round exactly, all within printed precision); no calculator or third-party implementation available |
| FINRISK calculator (Finland 2016; 10-year CHD, stroke, CVD; ages 30-64) | implemented from the paper's Table 2 equations (4/4 found in the PDF); Figure 1 reproduced, 29/30 bars exact and 1 off by 0.01 (after inferring the unstated HDL; two bars per sex were drawn with SBP 180, not the captioned 160) |
| Abu Dhabi Risk Study, ADRS (UAE 2024; 10-year ASCVD, hard CAD, stroke; ages 18+) | implemented from the paper's Table 2 and the supplement's SPSS syntax (32/32 terms agree); no worked example or calculator exists, so evidence is weak; the unprinted cohort GFR mean (111.4) is derived from Table 1, and of the supplement's two stroke formulas the one matching the printed mean score is used |
| PREDICT, New Zealand (7 equations: primary prevention 2018 and with BMI, VARIANZ policy, type 2 diabetes, major bleeding, after CVD, after ACS) | implemented from the authors' R package PredictRiskScores; all coefficients identical to its source, 49/49 printed table rows and both worked examples of the 2018 and bleeding papers reproduced; comparison by running the R code pending; 6 defects in the R code documented |
| PROCAM score (men 35-65; points and the published points-to-risk table) | implemented from the paper's Tables 3-4, 30/30 bands checked at both ends against the PDF |
| REGICOR (Framingham hard-CHD function recalibrated to Gerona, Spain; ages 35-74) | implemented from the paper's Tables 1-2 (39/39 rows found in the PDF, 40/40 bands checked at both ends); the paper's worked example contains two arithmetic slips (documented), so it cannot confirm the output |
| WHO/ISH charts (2007; 14 subregions, with and without cholesterol) | implemented as the chart lookup, verified vs every cell of the guideline's two printed charts (768/768) and R `whoishRisk` (10,749/10,752 cells; the 3 others are whoishRisk transcription errors corrected from the guideline) |
| HEART (chest pain, 0-10) | implemented from the published point rules (18/18), verified vs R `RiskScorescvd` (5,000 patients identical) |
| CHADS2, CHA2DS2-VASc, HAS-BLED (atrial fibrillation) | implemented from each paper's point table, 32/32 published rules and Gage's worked example; no independent implementation used |
| TIMI UA/NSTEMI (0-7) | implemented from the paper (11/11 rules), verified vs R `RiskScorescvd` (5,000 patients identical) |
| EDACS (chest pain) | implemented from the paper (29/29 rules), verified vs R `RiskScorescvd` (5,000 patients identical) |
| GRACE in-hospital (Granger 2003) and 6-month post-discharge (Eagle 2004) | implemented from the published nomograms, 78/78 bands checked at both ends against the PDFs; Granger's worked examples disagree with its own table (documented) |

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
