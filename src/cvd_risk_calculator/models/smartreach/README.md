# SMART-REACH (SMART Risk Score Extension)

SMART-REACH is an extension of the SMART2 model that incorporates additional risk factors and provides enhanced risk prediction for patients with established cardiovascular disease. It aims to improve risk stratification in secondary prevention.

## Overview

SMART-REACH extends the SMART2 model by incorporating additional clinical factors, biomarkers, and imaging findings to provide more precise risk prediction for recurrent cardiovascular events. The model was developed to better identify high-risk patients who might benefit from more intensive secondary prevention.

## Risk Factors

### Traditional Risk Factors
- **Age**: Years
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker
- **Diabetes**: Presence of diabetes

### Additional Risk Factors
- **Cardiac Biomarkers**: Troponin, BNP
- **Inflammatory Markers**: hsCRP
- **Kidney Function**: eGFR, creatinine
- **Anemia**: Hemoglobin levels
- **Polyvascular Disease**: Multiple vascular territories affected
- **Imaging Findings**: Coronary calcification, carotid plaque
- **Biomarkers**: Novel cardiovascular biomarkers

## Mathematical Formulation

SMART-REACH extends SMART2 with additional risk factors:

```
Risk_REACH = f(SMART2_factors + extended_factors)
```

The model incorporates:
- Traditional SMART2 variables
- Novel biomarkers
- Imaging and functional assessments
- Comorbidity indices

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: 20-30%
- **Very High Risk**: ≥30%

## Reference

SMART-REACH Study Group. Extended prediction rule for recurrent vascular events based on the SMART risk score: the SMART-REACH model. *European Heart Journal*. 2019.

DOI: [10.1093/eurheartj/ehy805](https://doi.org/10.1093/eurheartj/ehy805)

## Implementation Notes

- Extension of SMART2 for enhanced risk stratification
- Incorporates novel biomarkers and imaging
- Designed for secondary prevention populations
- Aims to identify very high-risk patients
- This is a placeholder - full implementation pending
- May require additional clinical data beyond standard risk factors
