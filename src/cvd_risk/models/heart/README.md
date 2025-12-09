# HEART (History, ECG, Age, Risk factors and Troponin) Score

The HEART score is a bedside clinical prediction tool used to stratify risk in patients presenting to the emergency department with chest pain suggestive of acute coronary syndrome (ACS).

## Overview

The HEART score combines five clinical elements (History, ECG, Age, Risk factors, Troponin) into a simple 0-10 point scoring system. It was developed to improve risk stratification of chest pain patients and guide decisions about admission versus discharge.

HEART has been validated in multiple international cohorts and is widely used in emergency departments for chest pain evaluation.

## Mathematical Formulation

HEART uses a point-based scoring system where each component receives 0-2 points based on clinical assessment:

```
HEART Score = History + ECG + Age + Risk Factors + Troponin
```

## Components

### History (0-2 points)
- **0 points**: Non-suspicious pain
- **1 point**: Moderately suspicious pain
- **2 points**: Highly suspicious pain

### ECG (0-2 points)
- **0 points**: Normal ECG
- **1 point**: Non-specific repolarization disturbance
- **2 points**: Significant ST-depression

### Age (0-2 points)
- **0 points**: <45 years
- **1 point**: 45-65 years
- **2 points**: >65 years

### Risk Factors (0-2 points)
- **0 points**: No risk factors
- **1 point**: 1-2 risk factors
- **2 points**: ≥3 risk factors or history of atherosclerotic disease

### Troponin (0-2 points)
- **0 points**: Normal
- **1 point**: 1-3× normal limit
- **2 points**: >3× normal limit

## Patient Population

HEART is designed for patients presenting to the emergency department with **chest pain suggestive of acute coronary syndrome (ACS)**. It should not be used for:
- ST-elevation myocardial infarction (STEMI)
- Obvious high-risk patients requiring immediate intervention
- Asymptomatic patients

## Risk Factors

The risk factors component includes:
- Hypercholesterolemia
- Hypertension
- Family history of CAD
- Diabetes mellitus
- Smoking
- Obesity
- History of atherosclerotic disease

## Risk Categories

- **Low Risk (0-3 points)**: 0-3.7% risk of major adverse cardiac events
- **Moderate Risk (4-6 points)**: 3.7-21.2% risk of major adverse cardiac events
- **High Risk (7-10 points)**: 21.2-72.7% risk of major adverse cardiac events

## Clinical Decision Rules

- **HEART ≤ 3**: Low risk, may be suitable for early discharge with outpatient follow-up
- **HEART ≥ 7**: High risk, requires admission and further evaluation
- **HEART 4-6**: Intermediate risk, requires further testing or observation

## Time Horizon

HEART predicts risk of major adverse cardiac events within **6 weeks** of presentation.

## References

Backus BE, Six AJ, Kelder JC, et al. A prospective validation of the HEART score for chest pain patients at the emergency department. Int J Cardiol. 2013;168(3):2153-2158.

DOI: [10.1016/j.ijcard.2013.01.255](https://doi.org/10.1016/j.ijcard.2013.01.255)

Six AJ, Cullen L, Backus BE, et al. The HEART score for the assessment of patients with chest pain in the emergency department: a multinational validation study. Crit Pathw Cardiol. 2013;12(3):121-126.

DOI: [10.1097/HPC.0b013e31828b327e](https://doi.org/10.1097/HPC.0b013e31828b327e)

## Implementation Notes

- **Emergency department tool only** - designed for acute chest pain evaluation
- Subjective elements (History, ECG interpretation) require clinical expertise
- Requires troponin measurement (preferably high-sensitivity)
- Validated for 6-week MACE prediction
- Particularly useful for identifying low-risk patients who can avoid admission
- Should be used as part of a comprehensive chest pain protocol
- Risk reassessment may be needed if clinical status changes
