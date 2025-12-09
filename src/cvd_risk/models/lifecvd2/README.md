# LifeCVD2

LifeCVD2 is a comprehensive lifetime cardiovascular disease risk prediction model that estimates both 10-year and lifetime CVD risk using competing risk methodology with age-specific baseline hazards and region-specific recalibration.

## Overview

LifeCVD2 provides a comprehensive assessment of cardiovascular risk by estimating both short-term (10-year) and long-term (lifetime) CVD risk. The model uses competing risk methodology, accounting for both CVD and non-CVD mortality, with region-specific recalibration for global applicability.

## Risk Factors

- **Age**: 35-99 years
- **Sex**: Male/Female
- **Region**: Low, Moderate, High, Very High CVD risk regions
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current/never
- **Diabetes**: Presence/absence

## Mathematical Formulation

LifeCVD2 uses a competing risk framework with:

1. **Linear predictors**: Sex-specific coefficients for CVD and non-CVD mortality
2. **Age interactions**: All risk factors interact with age
3. **Baseline hazards**: Age-specific 1-year baseline survival probabilities
4. **Recalibration**: Region-specific Weibull recalibration for global validity
5. **Lifetime accumulation**: Year-by-year risk accumulation using life table methods

## Risk Categories

### 10-Year Risk
- **Low Risk**: <5%
- **Moderate Risk**: 5-10%
- **High Risk**: 10-20%
- **Very High Risk**: ≥20%

### Lifetime Risk
- **Low Risk**: <20%
- **Moderate Risk**: 20-30%
- **High Risk**: 30-40%
- **Very High Risk**: ≥40%

## Additional Outputs

- **CVD-free life expectancy**: Years of life free from CVD
- **Median survival age**: Age at which 50% of individuals have survived CVD-free
- **Lifetime risk**: Cumulative CVD risk from current age to age 100

## Reference

Hageman SHJ, et al. Prediction of individual lifetime cardiovascular risk and potential treatment benefit: development and recalibration of the LIFE-CVD2 model to four European risk regions. *European Journal of Preventive Cardiology*. 2024.

DOI: [10.1093/eurjpc/zwae174](https://academic.oup.com/eurjpc/article-lookup/doi/10.1093/eurjpc/zwae174)

## Implementation Notes

- Valid for ages 35-99 years
- Requires diabetes status as mandatory input
- Uses region-specific recalibration (Low/Moderate/High/Very High)
- Provides both 10-year and lifetime risk assessment
- Suitable for clinical decision-making and research applications
- Based on competing risk methodology with age-specific hazards
