# SCORE2-Asia CKD (SCORE2 with Chronic Kidney Disease adjustments)

SCORE2-Asia CKD is the SCORE2 model with Chronic Kidney Disease (CKD) adjustments specifically calibrated for Asian populations. It incorporates eGFR and proteinuria measurements to adjust cardiovascular risk estimates.

## Overview

SCORE2-Asia CKD extends the SCORE2 model to Asian populations with chronic kidney disease. It was developed using large Asian cohort data and includes CKD-specific risk factors that significantly impact cardiovascular risk prediction in this population.

The model provides more accurate risk prediction for Asian patients with CKD compared to general population models.

## Mathematical Formulation

SCORE2-Asia CKD follows the same structure as SCORE2 but includes additional CKD variables:

```
10-year CVD Risk = 1 - exp(-exp(scale₁ + scale₂ × ln(-ln(1 - uncalibrated_risk))))
```

Where the uncalibrated risk includes CKD adjustments.

## Risk Factors

- **Age**: 40-69 years (optimal range)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Presence of diabetes (yes/no)
- **eGFR**: mL/min/1.73m² (estimated glomerular filtration rate)
- **Albuminuria/Proteinuria**: ACR (albumin-creatinine ratio) or dipstick protein

## Patient Population

SCORE2-Asia CKD is specifically designed for **Asian populations with chronic kidney disease**. It should not be used for:
- General population without CKD
- Non-Asian populations
- Patients with end-stage renal disease requiring dialysis

## CKD Risk Factor Transformations

| Risk Factor | Transformation | Notes |
|-------------|----------------|-------|
| eGFR | `(log(eGFR) - 4.4) / 0.26` | Log-transformed |
| Albuminuria | `(log(ACR) - 0.5) / 1.0` | Log-transformed, mg/mmol |
| Proteinuria | Dipstick result (0-4 scale) | Qualitative measure |

## Regions

SCORE2-Asia CKD provides region-specific risk estimates for Asian populations:
- **Low risk**: Certain Asian countries
- **Moderate risk**: Most Asian countries
- **High risk**: High-risk Asian regions
- **Very high risk**: Very high-risk Asian regions

## Risk Categories

- **Low**: <5%
- **Moderate**: 5-10%
- **High**: 10-20%
- **Very High**: ≥20%

## CKD Impact

Chronic kidney disease significantly increases cardiovascular risk:
- **eGFR <60 mL/min/1.73m²**: Substantially increases risk
- **Albuminuria**: Further risk elevation independent of eGFR
- **Combined CKD**: Highest risk category

## Reference

Kunihiro M, et al. (2022). SCORE2-Asia risk prediction algorithms: a recalibration of SCORE2 risk prediction algorithms based on a large cohort study in Asia. European Journal of Preventive Cardiology, 29(18), 2494-2505.

DOI: [10.1093/eurjpc/zwac176](https://doi.org/10.1093/eurjpc/zwac176)

## Implementation Notes

- **Asia + CKD specific model** - designed for Asian populations with chronic kidney disease
- Requires kidney function assessment (eGFR and proteinuria/albuminuria)
- More accurate than general SCORE2 for Asian CKD patients
- CKD variables significantly modify risk prediction
- Risk estimates are for 10-year period
- Particularly important for risk stratification in Asian diabetic and hypertensive populations with CKD
- Regular monitoring of kidney function recommended for risk reassessment
