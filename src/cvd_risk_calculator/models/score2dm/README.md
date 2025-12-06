# SCORE2-DM (SCORE2 for Diabetes Mellitus)

SCORE2-DM is a specialized version of the SCORE2 cardiovascular risk prediction model designed specifically for patients with diabetes mellitus. It provides more accurate risk estimates for diabetic populations.

## Overview

SCORE2-DM was developed to address the higher cardiovascular risk in patients with diabetes. The model adjusts the baseline risk and incorporates diabetes-specific risk factors to provide more accurate predictions for this high-risk population.

## Risk Factors

- **Age**: 40-69 years
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Type 1 or Type 2 diabetes
- **HbA1c**: Glycated hemoglobin (for diabetes management)
- **Duration of Diabetes**: Years since diagnosis

## Mathematical Formulation

SCORE2-DM modifies the standard SCORE2 model with diabetes-specific adjustments:

```
Risk_DM = Risk_SCORE2 × Diabetes_Adjustment_Factor
```

Where the diabetes adjustment factor accounts for:
- Presence of diabetes
- Diabetes duration
- Glycemic control (HbA1c)
- Diabetes-specific complications

## Risk Categories

- **Low Risk**: <5%
- **Moderate Risk**: 5-10%
- **High Risk**: 10-20%
- **Very High Risk**: ≥20%

## Reference

SCORE2 Working Group and ESC Cardiovascular Risk Collaboration. SCORE2-DM risk prediction algorithms: new models to estimate 10-year risk of cardiovascular disease in individuals with type 2 diabetes. *European Heart Journal*. 2023.

DOI: [10.1093/eurheartj/ehad260](https://doi.org/10.1093/eurheartj/ehad260)

## Implementation Notes

- Specifically designed for patients with diabetes
- Higher baseline risk compared to general population
- Accounts for diabetes duration and control
- This is a placeholder - full implementation pending
- Risk estimates are for 10-year period
