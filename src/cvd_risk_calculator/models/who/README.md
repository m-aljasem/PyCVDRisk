# WHO CVD Risk Charts

The World Health Organization (WHO) CVD Risk Charts provide region-specific cardiovascular disease risk estimates for 21 global regions. The model was developed to provide simple, standardized risk prediction tools for use in low- and middle-income countries.

## Overview

The WHO CVD Risk Charts were developed to address the need for standardized cardiovascular risk assessment tools that account for regional differences in CVD burden and risk factor distributions. The model provides 10-year risk estimates for cardiovascular disease based on major risk factors and incorporates region-specific baseline hazards.

The full model includes separate risk charts for 21 WHO regions, each with calibrated risk equations.

## Mathematical Formulation

The WHO model uses a logistic regression approach:

```
Risk = 1 / (1 + exp(-(ln(odds))))
```

Where the log-odds are calculated as:

```
ln(odds) = β₀ + β₁×age + β₂×sex + β₃×TC + β₄×HDL + β₅×SBP + β₆×smoking + β₇×diabetes
```

## Risk Factors

- **Age**: 40-80 years (optimal range)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Presence of diabetes (yes/no)

## WHO Regions

The model includes 21 global regions:

1. **African Region E** (High-income)
2. **African Region D** (Middle-income)
3. **African Region C** (Low-income)
4. **Region of the Americas A** (High-income)
5. **Region of the Americas B** (Middle-income)
6. **Region of the Americas D** (Low-income)
7. **Eastern Mediterranean Region B** (Middle-income)
8. **Eastern Mediterranean Region D** (Low-income)
9. **European Region A** (High-income)
10. **European Region B** (Middle-income)
11. **European Region C** (Low-income)
12. **South-East Asia Region B** (Middle-income)
13. **South-East Asia Region D** (Low-income)
14. **Western Pacific Region A** (High-income)
15. **Western Pacific Region B** (Middle-income)
16. **Western Pacific Region D** (Low-income)

## Coefficients

### Global Coefficients (Simplified Implementation)

| Risk Factor | Coefficient (β) | Notes |
|-------------|-----------------|-------|
| Age | 0.063 | Years |
| Female Sex | -0.088 | Female vs Male (protective) |
| Total Cholesterol | 0.167 | mmol/L |
| HDL Cholesterol | -0.212 | mmol/L (protective) |
| Systolic BP | 0.018 | mmHg |
| Smoking | 0.481 | Current smoker |
| Diabetes | 0.412 | Type 1 or Type 2 |
| Constant | -3.845 | Intercept |

## Region-Specific Adjustments

Each WHO region has specific:
- Baseline survival probabilities
- Calibration factors
- Risk factor coefficients adjusted for local epidemiology

The regional adjustments account for differences in:
- CVD mortality rates
- Risk factor prevalence
- Population characteristics
- Healthcare access

## Baseline Survival Probabilities

Baseline survival varies by region and socioeconomic status. In the simplified implementation:

- **Default**: 0.95 (10-year survival probability)

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: 20-30%
- **Very High Risk**: ≥30%

## Reference

WHO CVD Risk Chart Working Group. World Health Organization cardiovascular disease risk charts: revised models to estimate risk in 21 global regions. *The Lancet Global Health*. 2019;7(10):e1332-e1345.

DOI: [10.1016/S2214-109X(19)30318-3](https://doi.org/10.1016/S2214-109X(19)30318-3)

## Implementation Notes

- Model validated for ages 40-80 years
- Region-specific coefficients available for 21 WHO global regions
- This implementation uses simplified global coefficients
- Designed for use in low- and middle-income countries
- Accounts for regional differences in CVD epidemiology
- Risk estimates are for 10-year period
- All lipid values should be in mmol/L
