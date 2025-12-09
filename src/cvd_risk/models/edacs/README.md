# EDACS (Emergency Department Assessment of Chest Pain Score)

EDACS is a clinical decision tool for risk stratification of patients presenting to the emergency department with chest pain. It helps identify patients who are at low risk for major adverse cardiac events (MACE).

## Overview

The Emergency Department Assessment of Chest Pain Score (EDACS) was developed to improve risk stratification of chest pain patients in emergency departments. It combines patient demographics, risk factors, symptoms, and cardiac biomarkers to identify low-risk patients who may be suitable for early discharge and outpatient evaluation.

EDACS has been validated in multiple international cohorts and is designed to be simple to use at the bedside.

## Mathematical Formulation

EDACS uses a logistic regression model that predicts 30-day risk of major adverse cardiac events (MACE):

```
MACE Risk = 1 / (1 + exp(-(β₀ + Σ[β × variables])))
```

## Risk Factors

- **Age**: Years
- **Sex**: Male/Female
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Presence of diabetes (yes/no)
- **Known CAD**: Prior coronary artery disease (yes/no)
- **Chest Pain Characteristics**: Typical vs atypical
- **Troponin Level**: Initial troponin measurement

## Patient Population

EDACS is designed for patients presenting to the emergency department with **chest pain suggestive of acute coronary syndrome (ACS)**. It should not be used for:
- ST-elevation myocardial infarction (STEMI)
- Patients with obvious high-risk features requiring immediate intervention
- Asymptomatic patients

## Scoring System

EDACS assigns points based on risk factors:

| Risk Factor | Points |
|-------------|--------|
| Age ≥ 50 | +2 |
| Male Sex | +1 |
| Smoking | +1 |
| Diabetes | +1 |
| Known CAD | +1 |
| Typical Chest Pain | +1 |
| Troponin ≥ 99th percentile | +2 |

## Risk Categories

- **Very Low Risk (0-9 points)**: <1% 30-day MACE risk
- **Low Risk (10-15 points)**: 1-5% 30-day MACE risk
- **Moderate Risk (16-20 points)**: 5-15% 30-day MACE risk
- **High Risk (21+ points)**: >15% 30-day MACE risk

## Clinical Decision Rules

- **EDACS ≤ 9**: May be suitable for early discharge with outpatient follow-up
- **EDACS ≥ 20**: High risk, requires admission and aggressive management
- **EDACS 10-19**: Intermediate risk, requires further evaluation

## Reference

Than, M., Cullen, L., Aldous, S., et al. (2014). 2-Hour accelerated diagnostic protocol to assess patients with chest pain symptoms using contemporary troponins as the only biomarker. Journal of the American College of Cardiology, 64(3), 256-266.

DOI: [10.1016/j.jacc.2014.05.016](https://doi.org/10.1016/j.jacc.2014.05.016)

## Implementation Notes

- **Emergency department tool only** - designed for acute chest pain evaluation
- Requires initial troponin measurement (high-sensitivity preferred)
- Chest pain characteristics should be assessed by experienced clinician
- Validated for 30-day MACE prediction (not long-term risk)
- Particularly useful for identifying low-risk patients who can avoid admission
- Should be used as part of a comprehensive evaluation protocol
- Regular troponin monitoring may still be needed based on clinical judgment
