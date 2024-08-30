# cvd-risk v2

Verified, vectorised Python implementations of cardiovascular risk scores.

Every score is implemented from its original publication, every coefficient
cites its source, and every score is checked against an independent reference
implementation before it is released. See `reference/<score>/REPORT.md`.

> v2 is under development. v1 (0.1.x) remains on PyPI and on the `master` branch.

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
| QRISK3-2017 | implemented (LGPL-3.0 module), verified vs ClinRisk's own C code and R `QRISK3` |
| Pooled Cohort Equations (ACC/AHA 2013) | implemented, verified vs the guideline's Table A and R `CVrisk` / `PooledCohort` |
| Framingham 2008 general CVD (lab, BMI) | implemented, verified vs the Framingham Heart Study's own calculator spreadsheets, the paper's worked examples and R `CVrisk` |

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
