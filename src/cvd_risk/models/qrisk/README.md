# QRISK (QRISK Cardiovascular Risk Algorithm)

QRISK is the original UK-based cardiovascular disease risk prediction algorithm developed in 2007. It was designed to provide more accurate risk prediction for UK populations compared to existing models.

## Overview

QRISK was developed to address limitations of the Framingham risk equations when applied to UK populations. It incorporates additional risk factors and was calibrated using UK-specific data from the QRESEARCH database.

The original QRISK model includes factors such as age, sex, smoking, blood pressure, cholesterol, BMI, family history, and Townsend deprivation score.

## Risk Factors

- **Age**: 35-74 years
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Body Mass Index**: kg/m²
- **Family History**: CVD in first-degree relative
- **Townsend Deprivation Score**: Measure of socioeconomic deprivation

## Mathematical Formulation

QRISK uses a logistic regression model with multiple risk factors:

```
Risk = 1 / (1 + exp(-(ln(odds))))
```

Where the log-odds incorporate traditional risk factors plus additional UK-specific variables.

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: ≥20%

## Reference

Hippisley-Cox J, Coupland C, Vinogradova Y, et al. Predicting cardiovascular risk in England and Wales: prospective derivation and validation of QRISK2. *BMJ*. 2008;336(7659):1475-82.

(Note: This reference is for QRISK2, but QRISK was the original version)

DOI: [10.1136/bmj.39609.449676.25](https://doi.org/10.1136/bmj.39609.449676.25)

## Implementation Notes

- UK-specific algorithm
- Includes socioeconomic deprivation (Townsend score)
- More comprehensive than Framingham for UK populations
- This is a placeholder - full implementation pending
- Risk estimates are for 10-year period
