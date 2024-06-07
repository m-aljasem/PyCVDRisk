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
| SCORE2, SCORE2-OP, SCORE2-Diabetes, SCORE2 Asia-Pacific | implemented, verified vs R `RiskScorescvd` |
| Globorisk (lab, office, fatal; 182 countries) | implemented, verified vs the authors' R package |
| PREVENT (base, UACR, HbA1c, SDI, full; 10/30-year; 5 outcomes) | implemented, verified vs the authors' worked examples and R `preventr` |
| SCORE2-CKD, WHO 2019, QRISK3 | planned (see `reference/` for status) |
