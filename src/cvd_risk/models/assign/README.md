# ASSIGN (ASsessing cardiovascular risk using SIGN guidelines)

ASSIGN is the Scottish Intercollegiate Guidelines Network cardiovascular risk prediction model specifically developed for the Scottish population, with both original (v1.0) and recalibrated (v2.0) versions.

## Overview

The ASSIGN score was developed to provide accurate cardiovascular risk prediction for the Scottish population. It incorporates traditional risk factors plus additional factors like family history of cardiovascular disease and socioeconomic deprivation (measured by the Scottish Index of Multiple Deprivation - SIMD).

ASSIGN provides more accurate risk prediction for the Scottish population compared to other models by accounting for the higher cardiovascular risk burden in Scotland.

## Mathematical Formulation

ASSIGN uses a logistic regression model calibrated to Scottish population data:

```
Risk = 1 - (baseline survival)^exp(Σ[β × transformed variables])
```

The linear predictor includes main effects and interaction terms.

## Risk Factors

- **Age**: 30-74 years
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Presence of diabetes (yes/no)
- **Family History of CVD**: Yes/no
- **Socioeconomic Deprivation**: SIMD score (1-10, where 1 is most deprived)

## Model Versions

ASSIGN has two versions:
- **Version 1.0 (2006)**: Original calibration
- **Version 2.0 (2009)**: Recalibrated for improved accuracy

## Coefficients

### Version 1.0 Coefficients

| Risk Factor | Male (β) | Female (β) |
|-------------|----------|------------|
| Age | 0.05698 | 0.07203 |
| Total Cholesterol | 0.22286 | 0.12720 |
| HDL Cholesterol | -0.53684 | -0.40710 |
| Systolic BP | 0.01183 | 0.01103 |
| Diabetes | 0.81558 | 0.58992 |
| Family History CVD | 0.27500 | 0.27500 |
| Smoking | 0.02005 | 0.02005 |
| SIMD Score | 0.06296 | 0.06296 |

### Version 2.0 Coefficients

| Risk Factor | Male (β) | Female (β) |
|-------------|----------|------------|
| Age | 0.06306 | 0.07947 |
| Total Cholesterol | 0.21395 | 0.11678 |
| HDL Cholesterol | -0.51302 | -0.38781 |
| Systolic BP | 0.01183 | 0.01103 |
| Diabetes | 0.81558 | 0.58992 |
| Family History CVD | 0.27500 | 0.27500 |
| Smoking | 0.02005 | 0.02005 |
| SIMD Score | 0.06296 | 0.06296 |

## Baseline Survival (10-year)

- **Male v1.0**: 0.8831
- **Female v1.0**: 0.9365
- **Male v2.0**: 0.9130
- **Female v2.0**: 0.9666

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: ≥20%

## Reference

SIGN (Scottish Intercollegiate Guidelines Network). (2009). Risk estimation and the prevention of cardiovascular disease: a national clinical guideline. Edinburgh: SIGN. (SIGN publication no. 97)

## Implementation Notes

- **Scotland-specific model** - developed and validated for Scottish population
- Includes socioeconomic deprivation measure (SIMD) unique to ASSIGN
- Family history component adds predictive value
- Both v1.0 and v2.0 versions available for comparison
- Risk estimates are for 10-year period
- All lipid values should be in mmol/L
