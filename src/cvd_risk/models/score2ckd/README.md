# SCORE2-CKD (SCORE2 for Chronic Kidney Disease)

SCORE2-CKD is a specialized version of the SCORE2 cardiovascular risk prediction model designed for patients with chronic kidney disease (CKD). It accounts for the elevated cardiovascular risk associated with kidney dysfunction.

## Overview

SCORE2-CKD was developed to provide accurate cardiovascular risk prediction for patients with CKD, who have significantly higher cardiovascular risk than the general population. The model incorporates kidney function measures and adjusts for CKD-specific risk factors.

## Risk Factors

- **Age**: 40-69 years
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **eGFR**: Estimated glomerular filtration rate (mL/min/1.73m²)
- **Albuminuria**: Urine albumin-to-creatinine ratio
- **Kidney Function**: CKD stage/EPI equation

## Mathematical Formulation

SCORE2-CKD modifies SCORE2 with CKD-specific adjustments:

```
Risk_CKD = Risk_SCORE2 × CKD_Adjustment_Factor
```

Where the CKD adjustment factor incorporates:
- eGFR level
- Albuminuria/proteinuria
- CKD stage
- Kidney disease progression

## Risk Categories

- **Low Risk**: <5%
- **Moderate Risk**: 5-10%
- **High Risk**: 10-20%
- **Very High Risk**: ≥20%

## Reference

SCORE2-CKD Working Group. SCORE2-CKD: a cardiovascular risk prediction model for patients with chronic kidney disease. *European Heart Journal*. 2023.

DOI: [10.1093/eurheartj/ehad219](https://doi.org/10.1093/eurheartj/ehad219)

## Implementation Notes

- Specifically designed for patients with CKD
- Incorporates eGFR and albuminuria
- Accounts for higher baseline cardiovascular risk in CKD
- This is a placeholder - full implementation pending
- Risk estimates are for 10-year period
