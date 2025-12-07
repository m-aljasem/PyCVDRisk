# LifeCVD2

LifeCVD2 is a comprehensive cardiovascular disease risk prediction model that incorporates lifetime risk assessment alongside 10-year risk prediction. It provides both short-term and long-term cardiovascular risk estimates.

## Overview

LifeCVD2 extends traditional 10-year risk models by incorporating lifetime risk assessment, providing a more comprehensive view of cardiovascular risk over a patient's expected lifespan. The model combines traditional risk factors with novel biomarkers and risk enhancers.

## Risk Factors

- **Age**: Full adult age range
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current/former/never
- **Diabetes**: Presence and type
- **Family History**: CVD family history
- **Biomarkers**: Novel cardiovascular biomarkers
- **Risk Enhancers**: Additional risk factors

## Mathematical Formulation

LifeCVD2 combines multiple modeling approaches:

1. **10-year risk**: Traditional logistic regression
2. **Lifetime risk**: Age-specific risk accumulation
3. **Integrated risk**: Combined short- and long-term assessment

## Risk Categories

### 10-Year Risk
- **Low Risk**: <5%
- **Borderline Risk**: 5-7.5%
- **Intermediate Risk**: 7.5-20%
- **High Risk**: ≥20%

### Lifetime Risk
- **Low Risk**: <39%
- **Intermediate Risk**: 39-49%
- **High Risk**: ≥50%

## Reference

Lloyd-Jones DM, Braun LT, Ndumele CE, et al. Use of risk assessment tools to guide decision-making in the primary prevention of atherosclerotic cardiovascular disease: A special report from the American Heart Association and American College of Cardiology. *Journal of the American College of Cardiology*. 2019;73(24):3153-3167.

DOI: [10.1016/j.jacc.2019.02.016](https://doi.org/10.1016/j.jacc.2019.02.016)

## Implementation Notes

- Combines 10-year and lifetime risk assessment
- Incorporates novel risk factors and biomarkers
- Provides comprehensive risk communication
- This is a placeholder - full implementation pending
- Suitable for both clinical and research applications
