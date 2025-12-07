# SCORE (Systematic COronary Risk Evaluation)

SCORE is the original European cardiovascular risk prediction model published in 2003 by the European Society of Cardiology. It estimates 10-year risk of fatal cardiovascular disease for European populations.

## Overview

SCORE was developed as the first comprehensive European cardiovascular risk prediction model, replacing region-specific models with a standardized approach. The model predicts 10-year risk of fatal cardiovascular disease (CVD) events including coronary heart disease and stroke deaths.

SCORE provides separate risk charts for high-risk and low-risk European regions, calibrated to local mortality statistics.

## Risk Factors

- **Age**: 40-65 years (optimal range)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)

## Regions

SCORE provides region-specific risk estimates:

- **High-risk regions**: Most Eastern European countries
- **Low-risk regions**: Most Western European countries, including UK, France, Germany, Spain, Italy

## Mathematical Formulation

SCORE uses a risk chart approach with age-specific risk tables based on total cholesterol and systolic blood pressure, stratified by smoking status and sex.

## Risk Categories

- **Low Risk**: <1%
- **Moderate Risk**: 1-5%
- **High Risk**: 5-10%
- **Very High Risk**: ≥10%

## Reference

Conroy RM, Pyörälä K, Fitzgerald AP, et al. Estimation of ten-year risk of fatal cardiovascular disease in Europe: the SCORE project. *European Heart Journal*. 2003;24(11):987-1003.

DOI: [10.1016/S0195-668X(03)00114-3](https://doi.org/10.1016/S0195-668X(03)00114-3)

## Implementation Notes

- Model validated for ages 40-65 years
- Predicts fatal CVD events only (not non-fatal)
- Separate charts for high-risk and low-risk regions
- This is a placeholder - full implementation pending
- Risk estimates are for 10-year period
