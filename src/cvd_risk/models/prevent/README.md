# PREVENT (Predicting Risk of Cardiovascular Disease Events)

PREVENT is the American Heart Association's comprehensive cardiovascular risk prediction model for primary prevention, replacing the previous ASCVD Pooled Cohort Equations. It incorporates additional risk factors and provides more precise risk estimates.

## Overview

PREVENT (Predicting Risk of cardiovascular disease EVENTs) was developed to improve upon the ASCVD risk calculator by incorporating additional risk factors, novel biomarkers, and contemporary data. The model provides both 10-year and 30-year risk estimates.

## Risk Factors

### Traditional Risk Factors
- **Age**: Years
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mg/dL
- **HDL Cholesterol**: mg/dL
- **Smoking Status**: Current smoker

### Additional Risk Factors
- **Diabetes**: Type 1 or Type 2
- **Race/Ethnicity**: For calibration
- **Chronic Kidney Disease**: eGFR and albuminuria
- **Family History**: Premature CVD
- **Inflammatory Biomarkers**: hsCRP, Lp(a)
- **Risk-Enhancing Factors**: Social deprivation, etc.

## Mathematical Formulation

PREVENT uses a sophisticated modeling approach combining:

1. **Logistic regression** for 10-year risk
2. **Survival analysis** for 30-year risk
3. **Calibration factors** for different populations

## Risk Categories

### 10-Year Risk
- **Low Risk**: <5%
- **Borderline Risk**: 5-7.5%
- **Intermediate Risk**: 7.5-20%
- **High Risk**: ≥20%

### 30-Year Risk
- **Low Risk**: <20%
- **Intermediate Risk**: 20-30%
- **High Risk**: ≥30%

## Reference

Khan SS, Coresh J, Pencina MJ, et al. Novel prediction equations for absolute risk assessment of total cardiovascular disease incorporating cardiovascular-kidney-metabolic health: the PREVENT equations. *Circulation*. 2024.

DOI: [10.1161/CIRCULATIONAHA.123.067626](https://doi.org/10.1161/CIRCULATIONAHA.123.067626)

## Implementation Notes

- Comprehensive primary prevention model
- Incorporates CKM (Cardiovascular-Kidney-Metabolic) health
- Includes novel biomarkers and risk enhancers
- Provides both 10-year and 30-year risk estimates
- This is a placeholder - full implementation pending
- Represents current state-of-the-art in CVD risk prediction
