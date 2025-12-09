# DIAL2 (DIAbetes Lifetime perspective model)

DIAL2 estimates individual lifetime risk of incident cardiovascular events in adults with Type 2 diabetes. It accounts for the competing risk of non-CVD mortality and provides more accurate risk prediction for diabetic populations.

## Overview

The DIAbetes Lifetime perspective model (DIAL2) was developed specifically for patients with Type 2 diabetes to provide more accurate cardiovascular risk prediction than general population models. It incorporates diabetes-specific risk factors and accounts for competing risks.

DIAL2 was developed using data from multiple European diabetes cohorts and provides lifetime risk estimates rather than fixed time horizons.

## Mathematical Formulation

DIAL2 uses a competing risks model that simultaneously estimates:

1. **CVD risk**: Risk of first cardiovascular event
2. **Non-CVD mortality risk**: Competing risk of death from non-cardiovascular causes

The model uses transformed risk factors and accounts for age interactions.

## Risk Factors

- **Age**: Years (typically 30-80)
- **Sex**: Male/Female
- **Age at Diabetes Onset**: Years
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **HbA1c**: mmol/mol or %
- **eGFR**: mL/min/1.73m² (estimated glomerular filtration rate)
- **Smoking Status**: Current smoker (yes/no)

## Patient Population

DIAL2 is specifically designed for patients with **Type 2 diabetes**. It should not be used for:
- Type 1 diabetes
- General population without diabetes
- Gestational diabetes

## Risk Factor Transformations

| Risk Factor | Transformation | Notes |
|-------------|----------------|-------|
| Systolic BP | `(sbp - 138) / 17` | Standardized |
| eGFR | `(log(egfr) - 4.4) / 0.26` | Log-transformed |
| Total Cholesterol | `(chol - 5.1) / 1.1` | Standardized |
| HDL Cholesterol | `(hdl - 1.3) / 0.4` | Standardized |
| HbA1c | `(hba1c - 55) / 16` | mmol/mol units |
| Age at Onset | `(age_onset - 58) / 12` | Standardized |

## Age Interactions

All risk factors include age interaction terms of the form: `β₁ × variable + β₂ × variable × ((age - 63) / 12)`

## Risk Categories

- **Low Risk**: <10% lifetime risk
- **Moderate Risk**: 10-20% lifetime risk
- **High Risk**: 20-30% lifetime risk
- **Very High Risk**: ≥30% lifetime risk

## Reference

Ostergaard HB, et al. (2023). Estimating individual lifetime risk of incident cardiovascular events in adults with Type 2 diabetes: an update and geographical calibration of the DIAbetes Lifetime perspective model (DIAL2). European Journal of Preventive Cardiology, 30, 61–69.

DOI: [10.1093/eurjpc/zwac232](https://doi.org/10.1093/eurjpc/zwac232)

## Implementation Notes

- **Diabetes-specific model only** - designed exclusively for Type 2 diabetes patients
- Lifetime risk estimates (not fixed time horizon like 10-year risk)
- Accounts for competing risks of non-CVD mortality
- Requires diabetes onset age (critical for accurate prediction)
- Kidney function (eGFR) is an important predictor
- All laboratory values should be recent and standardized
- Particularly useful for long-term risk communication in diabetes management
