# GRACE2 (Global Registry of Acute Coronary Events version 2.0)

GRACE2 is a clinical prediction tool for estimating the risk of death from admission to 6 months in patients presenting with acute coronary syndromes (ACS).

## Overview

The GRACE2 risk score was developed from the Global Registry of Acute Coronary Events (GRACE) database to predict mortality in patients with acute coronary syndromes. It provides accurate risk stratification for short-term prognosis following ACS presentation.

GRACE2 has been extensively validated and is widely used in clinical practice to guide management decisions in ACS patients.

## Mathematical Formulation

GRACE2 uses a logistic regression model that predicts 6-month mortality:

```
6-month Mortality Risk = 1 / (1 + exp(-(β₀ + Σ[β × variables])))
```

The score ranges from 1-263 points and is calculated from clinical variables available at presentation.

## Risk Factors

- **Age**: Years
- **Heart Rate**: beats per minute
- **Systolic Blood Pressure**: mmHg
- **Creatinine**: μmol/L or mg/dL
- **Killip Class**: Classification of heart failure severity
- **Cardiac Arrest at Admission**: Yes/no
- **ST-segment Deviation**: Yes/no
- **Elevated Cardiac Enzymes**: Yes/no

## Patient Population

GRACE2 is designed for patients presenting with **acute coronary syndromes (ACS)**, including:
- ST-elevation myocardial infarction (STEMI)
- Non-ST-elevation myocardial infarction (NSTEMI)
- Unstable angina

It should not be used for stable coronary artery disease or primary prevention.

## Killip Classification

| Class | Description |
|-------|-------------|
| I | No heart failure |
| II | S3 gallop, pulmonary rales, or venous hypertension |
| III | Acute pulmonary edema |
| IV | Cardiogenic shock |

## Scoring System

GRACE2 assigns points based on clinical variables:

| Risk Factor | Points |
|-------------|--------|
| Age (per 10 years) | +9-72 |
| Heart Rate (per 10 bpm) | +3-30 |
| Systolic BP (per 10 mmHg) | -11 to +20 |
| Creatinine (per 10 μmol/L) | +1-28 |
| Killip Class II | +21 |
| Killip Class III | +39 |
| Killip Class IV | +64 |
| Cardiac Arrest | +43 |
| ST Deviation | +29 |
| Elevated Enzymes | +15 |

## Risk Categories

- **Low Risk (≤108 points)**: <5% 6-month mortality
- **Moderate Risk (109-140 points)**: 5-15% 6-month mortality
- **High Risk (≥141 points)**: >15% 6-month mortality

## Clinical Applications

- **Low risk**: May be suitable for early discharge or less aggressive management
- **High risk**: Requires intensive monitoring, invasive evaluation, and aggressive therapy
- **Moderate risk**: Individualized management based on other clinical factors

## Reference

Granger CB, Goldberg RJ, Dabbous O, et al. Predictors of hospital mortality in the global registry of acute coronary events. Arch Intern Med. 2003;163(19):2345-2353.

DOI: [10.1001/archinte.163.19.2345](https://doi.org/10.1001/archinte.163.19.2345)

## Implementation Notes

- **ACS-specific model only** - designed for acute coronary syndrome patients
- Predicts 6-month mortality (not long-term risk)
- All variables should be measured at hospital presentation
- Killip class requires clinical assessment
- Particularly useful for triage and management decisions in ACS
- Validated in multiple international cohorts
- Should be used as part of comprehensive ACS evaluation
- Regular reassessment may be needed as clinical status changes
