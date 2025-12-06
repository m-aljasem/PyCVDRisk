# Globorisk

Globorisk is a global cardiovascular disease risk prediction model that provides country-specific risk estimates accounting for local mortality rates. It was developed using data from 182 countries to create globally applicable risk prediction algorithms.

## Overview

Globorisk addresses the limitation of region-specific risk models by providing country-specific cardiovascular disease risk estimates. The model accounts for local cardiovascular mortality rates and provides risk predictions that are calibrated to each country's baseline risk.

The full Globorisk model includes country-specific coefficients and baseline hazards for 182 countries, making it one of the most comprehensive global risk prediction tools.

## Mathematical Formulation

Globorisk uses a logistic regression model with country-specific coefficients:

```
Risk = 1 / (1 + exp(-(ln(odds))))
```

Where the log-odds are calculated as:

```
ln(odds) = β₀ + β₁×age + β₂×sex + β₃×TC + β₄×HDL + β₅×SBP + β₆×smoking + β₇×diabetes
```

## Risk Factors

- **Age**: 40-74 years (optimal range)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Presence of diabetes (yes/no)

## Coefficients

### Global Coefficients (Simplified Implementation)

| Risk Factor | Coefficient (β) | Notes |
|-------------|-----------------|-------|
| Age | 0.065 | Years |
| Male Sex | 0.241 | Male vs Female |
| Total Cholesterol | 0.184 | mmol/L |
| HDL Cholesterol | -0.231 | mmol/L (protective) |
| Systolic BP | 0.020 | mmHg |
| Smoking | 0.512 | Current smoker |
| Diabetes | 0.445 | Type 1 or Type 2 |
| Constant | -4.623 | Intercept |

## Country-Specific Adjustments

The full Globorisk model includes country-specific coefficients for 182 countries. Each country has its own:

- Baseline survival probability
- Regression coefficients for risk factors
- Calibration factors

Country-specific coefficients are derived from:
- Local mortality statistics
- Regional cohort studies
- WHO mortality data
- National health surveys

## Baseline Survival Probabilities

The baseline survival probability varies by country and reflects the local cardiovascular mortality rate. In the simplified implementation:

- **Global Average**: 0.92 (10-year survival probability)

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: 20-30%
- **Very High Risk**: ≥30%

## Reference

Ueda P, Woodward M, Lu Y, et al. Laboratory-based and office-based risk scores and charts to predict 10-year risk of cardiovascular disease in 182 countries: a pooled analysis of prospective cohorts and health surveys. *The Lancet Diabetes & Endocrinology*. 2017;5(3):196-213.

DOI: [10.1016/S2213-8587(17)30015-0](https://doi.org/10.1016/S2213-8587(17)30015-0)

## Implementation Notes

- Model validated for ages 40-74 years
- Country-specific coefficients available for 182 countries
- This implementation uses global average coefficients
- Full model requires country-specific calibration for optimal accuracy
- Accounts for local CVD mortality rates in risk estimation
- Risk estimates are for 10-year period
- All lipid values should be in mmol/L
