<div align="center">
  <img src="https://raw.githubusercontent.com/m-aljasem/PyCVDRisk/master/docs/pycvdrisk-github-header-banner.png" alt="PyCVDRisk" width="100%">

### Cardiovascular risk scores from across the world, verified, in Python

[![PyPI version](https://img.shields.io/pypi/v/cvd-risk.svg)](https://pypi.org/project/cvd-risk/)
[![Python versions](https://img.shields.io/pypi/pyversions/cvd-risk.svg)](https://pypi.org/project/cvd-risk/)
[![CI](https://github.com/m-aljasem/PyCVDRisk/actions/workflows/ci.yml/badge.svg)](https://github.com/m-aljasem/PyCVDRisk/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20LGPL--3.0%20(QRISK)-blue.svg)](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/LICENSE)
[![Website](https://img.shields.io/badge/website-pycvdrisk.aljasem.eu.org-db2777.svg)](https://pycvdrisk.aljasem.eu.org)

[🌐 Website](https://pycvdrisk.aljasem.eu.org) • [🚀 Quick start](#-quick-start) • [📊 Scores](#-the-scores) • [🔬 Verification](#-how-each-score-is-verified) • [🤝 Contributing](#-contributing)

</div>

---

## 🫀 What is PyCVDRisk?

**PyCVDRisk** (`cvd-risk` on PyPI) is a Python library of **30 published cardiovascular risk scores** from the
Americas, Europe, the Middle East and the Asia-Pacific region: SCORE2, PREVENT, the Pooled Cohort Equations,
QRISK3, Framingham, the WHO charts, PREDICT and many more, under **one fast, vectorised API**.

Implementations of these scores already exist, but they are scattered across languages, packages and web
calculators, and their outputs are rarely shown to match the equations of the original papers. PyCVDRisk is
built so that every number can be traced and checked:

- 📖 **Implemented from the source.** Every coefficient, baseline survival and point table comes from the
  original publication or its supplement, and the code cites the table it came from.
- ✅ **Verified, not just implemented.** 19 of the 30 scores are verified against an independent reference
  implementation (the authors' own code or calculator, official charts, or an established R package); the rest
  are checked line by line against their papers.
- 🔍 **Transparent.** Each score ships a verification report recording what was compared, the agreement reached
  and every discrepancy found, including errors in the original publications.
- ⚡ **Fast and vectorised.** Plain functions over NumPy arrays: pass one patient or a million.
- 🛡️ **Safe by default.** Invalid inputs return `NaN` instead of being silently clamped.

## 📦 Installation

```bash
pip install cvd-risk
```

Requires Python 3.10+ and NumPy.

## 🚀 Quick start

```python
from cvd_risk import score2, prevent, qrisk3

# SCORE2 (Europe): two patients at once
score2(age=[50, 62], sex=["male", "female"], smoker=[1, 0], sbp=[140, 130],
       total_chol=[5.5, 6.1], hdl_chol=[1.3, 1.6], diabetes=0, region="low")
# array([0.0591, 0.0379])   10-year risk as a fraction

# PREVENT (AHA 2024)
prevent(age=50, sex="female", sbp=160, bp_treated=1, total_chol=5.2, hdl_chol=1.4,
        statin=0, diabetes=1, smoker=0, egfr=90, bmi=35, years=10, outcome="total_cvd")

# QRISK3 (UK)
qrisk3(age=64, sex="female", ethnicity="indian", smoking="non", bmi=27, chol_hdl_ratio=4.0,
       sbp=140, sbp_sd=8, townsend=0)
```

Works directly on data frames:

```python
import pandas as pd
df = pd.read_csv("cohort.csv")
df["score2"] = score2(df.age, df.sex, df.smoker, df.sbp, df.total_chol, df.hdl_chol,
                      df.diabetes, region="moderate")
```

**Conventions shared by every score:** cholesterol in mmol/L, blood pressure in mmHg, any argument may be an
array, and risks are returned as fractions in [0, 1].

## 📊 The scores

✅ **Reference-verified**: matches an independent implementation. 📄 **Paper-checked**: no independent
implementation exists, so the score is checked against every rule, table and worked example of its paper.
Each score's page on the [website](https://pycvdrisk.aljasem.eu.org) has a runnable example.

#### Primary prevention

| Score | Region | Year | Function | Evidence |
|---|---|---|---|---|
| **SCORE2** | Europe | 2021 | `score2` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/score2/REPORT.md)) |
| **SCORE2-OP** | Europe | 2021 | `score2_op` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/score2/REPORT.md)) |
| **SCORE2 Asia-Pacific** | Asia-Pacific | 2024 | `score2_asia_pacific` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/score2/REPORT.md)) |
| **SCORE** | Europe | 2003 | `score` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/score/REPORT.md)) |
| **Globorisk** | Global, 182 countries | 2015-2017 | `globorisk` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/globorisk/REPORT.md)) |
| **WHO 2019** | Global, 21 regions / 190 countries | 2019 | `who2019` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/who2019/REPORT.md)) |
| **WHO/ISH** | Global, 14 WHO subregions | 2007 | `who_ish` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/who_ish/REPORT.md)) |
| **PREVENT** | United States | 2024 | `prevent` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/prevent/REPORT.md)) |
| **Pooled Cohort Equations** | United States | 2013 | `pce` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/pce/REPORT.md)) |
| **Framingham** | Framingham, Massachusetts | 2008 | `framingham` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/framingham/REPORT.md)) |
| **Reynolds** | United States (Boston) | 2007-2008 | `reynolds` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/reynolds/REPORT.md)) |
| **QRISK3** | United Kingdom | 2017 | `qrisk3` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/qrisk3/REPORT.md)) |
| **QRISK2** | United Kingdom | 2014 | `qrisk2` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/qrisk2/REPORT.md)) |
| **FINRISK** | Finland | 2016 | `finrisk` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/finrisk/REPORT.md)) |
| **ADRS** | United Arab Emirates | 2024 | `adrs` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/adrs/REPORT.md)) |
| **PROCAM** | Germany | 2002 | `procam` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/procam/REPORT.md)) |
| **REGICOR** | Spain | 2003 | `regicor` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/regicor/REPORT.md)) |
| **PREDICT** | New Zealand | 2018-2022 | `predict_cvd`, `predict_cvd_bmi`, `predict_cvd_policy`, `predict_t2d`, `predict_bleed`, `predict_post_cvd`, `predict_post_acs` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/predict/REPORT.md)) |
| **LIFE-CVD2** | Europe | 2024 | `life_cvd2` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/life_cvd2/REPORT.md)) |

#### Diabetes, kidney disease, established CVD and atrial fibrillation

| Score | Region | Year | Function | Evidence |
|---|---|---|---|---|
| **SCORE2-Diabetes** | Europe | 2023 | `score2_diabetes` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/score2/REPORT.md)) |
| **SCORE2-CKD** | Europe | 2022 | `score2_ckd` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/score2_ckd/REPORT.md)) |
| **SMART-REACH** | International | 2018 | `smart_reach` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/smart_reach/REPORT.md)) |
| **DIAL2** | Europe (derived in Sweden) | 2023 | `dial2` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/dial2/REPORT.md)) |
| **CHA2DS2-VASc** | Europe | 2010 | `cha2ds2_vasc` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/af/REPORT.md)) |
| **CHADS2** | United States | 2001 | `chads2` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/af/REPORT.md)) |
| **HAS-BLED** | Europe | 2010 | `has_bled` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/af/REPORT.md)) |

#### Acute chest pain and coronary syndromes

| Score | Region | Year | Function | Evidence |
|---|---|---|---|---|
| **HEART** | Netherlands | 2008 | `heart` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/heart/REPORT.md)) |
| **TIMI** | United States (Boston) | 2000 | `timi` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/timi/REPORT.md)) |
| **EDACS** | Australia & New Zealand | 2014 | `edacs` | ✅ Reference-verified ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/edacs/REPORT.md)) |
| **GRACE** | International, 14 countries | 2003-2004 | `grace_hospital`, `grace_6month` | 📄 Paper-checked ([report](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/grace/REPORT.md)) |

## 🔬 How each score is verified

1. **Transcribed from the source.** Coefficients and tables are taken from the original publication, never from
   secondary sources, and each cites its table or line.
2. **Checked against the paper.** The implementation reproduces the worked examples, printed tables and risk
   charts its paper provides. Errors found in the publications are documented, not silently corrected.
3. **Verified independently.** Where an independent implementation exists, outputs are compared directly,
   across thousands of simulated patients wherever the reference can be run.

The full reports are in [`reference/`](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/reference/README.md).

## 📝 Citing PyCVDRisk

A paper describing the library and its verification is in preparation. Until it is published, please cite the
software using [`CITATION.cff`](https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/CITATION.cff) (GitHub's
"Cite this repository" button gives BibTeX and APA).

## 🤝 Contributing

Contributions from clinicians, epidemiologists and developers are welcome:

- 🐞 **Report a discrepancy.** If an output differs from a paper, calculator or other implementation,
  [open an issue](https://github.com/m-aljasem/PyCVDRisk/issues) with the inputs, both results and the source.
- ➕ **Propose a score.** Suggest a published risk equation with its reference; it is added once its full
  equation or point table is available and can be verified.
- 📚 **Provide a reference.** Authors' code, official calculators and worked examples let a paper-checked score
  become reference-verified.
- 🛠️ **Improve the code.** Pull requests must keep the tests and the verification against the source passing.

Contact: **Mohamad AlJasem**, [mohamad@aljasem.eu.org](mailto:mohamad@aljasem.eu.org)

## ⚖️ License

MIT, except the QRISK2 and QRISK3 modules (`src/cvd_risk/qrisk2`, `src/cvd_risk/qrisk3`), which are derived
from ClinRisk's code and licensed under LGPL-3.0-or-later.

> ⚠️ PyCVDRisk is intended for research and education. It is not a medical device and is not a substitute for
> clinical judgement.
