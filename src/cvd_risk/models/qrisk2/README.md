# QRISK2 (QRISK Cardiovascular Risk Algorithm 2)

QRISK2 is the second version of the UK-based cardiovascular disease risk prediction algorithm, published in 2008. It improved upon the original QRISK model with enhanced risk factors and better calibration.

## Overview

QRISK2 was developed to provide more accurate cardiovascular risk prediction for UK populations. It incorporates additional clinical risk factors and uses more recent UK data for calibration. The model has been widely adopted in UK clinical practice.

## Risk Factors

- **Age**: 35-74 years
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Body Mass Index**: kg/m²
- **Family History**: CVD in first-degree relative
- **Townsend Deprivation Score**: Socioeconomic deprivation
- **Atrial Fibrillation**: Presence of AF
- **Rheumatoid Arthritis**: Presence of RA
- **Chronic Kidney Disease**: eGFR <60 mL/min/1.73m²
- **On Statins**: Current statin treatment

## Mathematical Formulation

QRISK2 uses a logistic regression model:

```
Risk = 1 / (1 + exp(-(ln(odds))))
```

Where the log-odds are calculated using multiple risk factors calibrated to UK populations.

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: ≥20%

## Reference

Hippisley-Cox J, Coupland C, Vinogradova Y, et al. Predicting cardiovascular risk in England and Wales: prospective derivation and validation of QRISK2. *BMJ*. 2008;336(7659):1475-82.

DOI: [10.1136/bmj.39609.449676.25](https://doi.org/10.1136/bmj.39609.449676.25)

## Implementation Notes

- UK-specific algorithm calibrated to QRESEARCH database
- Includes Townsend deprivation score for socioeconomic adjustment
- Accounts for various comorbidities (AF, RA, CKD)
- Widely used in UK primary care
- This is a placeholder - full implementation pending
- Risk estimates are for 10-year period
