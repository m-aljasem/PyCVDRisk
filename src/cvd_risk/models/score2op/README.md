# SCORE2-OP (SCORE2 for Older Persons)

SCORE2-OP is a specialized version of the SCORE2 cardiovascular risk prediction model designed for older adults aged 70-89 years. It extends the age range beyond the standard SCORE2 model.

## Overview

SCORE2-OP was developed to provide accurate cardiovascular risk prediction for older adults, addressing the limitations of SCORE2 which was validated only up to age 69. The model accounts for age-related changes in risk factor relationships.

## Risk Factors

- **Age**: 70-89 years (extended range)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Frailty**: Measures of physical frailty
- **Comorbidities**: Age-related conditions

## Mathematical Formulation

SCORE2-OP extends SCORE2 with age-specific adjustments for older adults:

```
Risk_OP = f(age, sex, SBP, TC, HDL, smoking, frailty_factors)
```

The model accounts for:
- Non-linear age effects in older age groups
- Changing relationships between risk factors and outcomes
- Impact of frailty and multimorbidity
- Different baseline hazards for older populations

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: 20-30%
- **Very High Risk**: ≥30%

## Reference

SCORE2-OP Working Group. SCORE2-OP risk prediction algorithms: estimating incident cardiovascular event risk in older persons. *European Heart Journal*. 2021.

DOI: [10.1093/eurheartj/ehab284](https://doi.org/10.1093/eurheartj/ehab284)

## Implementation Notes

- Designed specifically for adults aged 70-89 years
- Accounts for frailty and multimorbidity
- Extended age range beyond standard SCORE2
- This is a placeholder - full implementation pending
- Risk estimates are for 10-year period
