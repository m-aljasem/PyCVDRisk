# TIMI (Thrombolysis In Myocardial Infarction) Risk Score for UA/NSTEMI

The TIMI risk score is a clinical prediction tool used to stratify risk in patients with unstable angina (UA) and non-ST elevation myocardial infarction (NSTEMI).

## Overview

The TIMI risk score was developed from the TIMI 11B and ESSENCE trials to predict 14-day and long-term risk of death, myocardial infarction, and urgent revascularization in patients with unstable angina and NSTEMI.

It is simple to calculate at the bedside and helps guide management decisions in acute coronary syndrome patients.

## Mathematical Formulation

TIMI uses a point-based scoring system (0-7 points) that predicts risk of adverse cardiovascular events. Each risk factor present adds 1 point to the score.

## Risk Factors

- **Age ≥65 years**: +1 point
- **≥3 CAD risk factors**: +1 point
- **Known CAD (stenosis ≥50%)**: +1 point
- **ST-segment deviation**: +1 point
- **≥2 anginal events in prior 24 hours**: +1 point
- **Use of aspirin in prior 7 days**: +1 point
- **Elevated cardiac markers**: +1 point

## Patient Population

TIMI is designed for patients presenting with **unstable angina (UA) or non-ST elevation myocardial infarction (NSTEMI)**. It should not be used for:
- ST-elevation myocardial infarction (STEMI)
- Stable angina
- Primary prevention

## CAD Risk Factors

The ≥3 CAD risk factors include:
- Family history of CAD
- Hypertension
- Hypercholesterolemia
- Diabetes
- Current smoking

## Scoring System

| Risk Factor | Points | Description |
|-------------|--------|-------------|
| Age ≥65 | +1 | Elderly patients |
| ≥3 CAD Risk Factors | +1 | Diabetes, smoking, hypertension, etc. |
| Known CAD | +1 | Prior coronary artery disease |
| ST Deviation | +1 | ECG changes |
| ≥2 Angina Episodes | +1 | Frequent recent symptoms |
| Aspirin Use | +1 | Recent aspirin therapy |
| Elevated Markers | +1 | Troponin or CK-MB elevation |

## Risk Categories

- **Very Low Risk (0-1 points)**: <5% risk of adverse events
- **Low Risk (2-3 points)**: 5-13% risk of adverse events
- **Moderate Risk (4 points)**: 13-20% risk of adverse events
- **High Risk (5-7 points)**: 20-40% risk of adverse events

## Clinical Applications

- **Low risk (≤3 points)**: May be suitable for early conservative management
- **High risk (≥4 points)**: Requires aggressive management, likely invasive evaluation
- **Very high risk**: Consider immediate invasive strategy

## Time Horizons

TIMI predicts risk at multiple time points:
- **14-day risk**: Immediate prognosis
- **1-year risk**: Long-term prognosis
- **Long-term risk**: Extended follow-up

## Reference

Antman EM, Cohen M, Bernink PJ, et al. The TIMI risk score for unstable angina/non-ST elevation MI: A method for prognostication and therapeutic decision making. JAMA. 2000;284(7):835-842.

DOI: [10.1001/jama.284.7.835](https://doi.org/10.1001/jama.284.7.835)

## Implementation Notes

- **UA/NSTEMI specific model** - designed for non-ST elevation ACS
- Simple point-based system (no calculations required)
- All variables readily available at presentation
- Validated for short-term (14-day) and long-term prognosis
- Particularly useful for triage decisions in the emergency department
- Should be used as part of comprehensive ACS evaluation
- Risk stratification helps guide antiplatelet and anticoagulant therapy intensity
