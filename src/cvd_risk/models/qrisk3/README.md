# QRISK3 (QRISK Cardiovascular Disease Risk Algorithm)

QRISK3 is a cardiovascular disease risk prediction algorithm developed in the UK that estimates 10-year risk of cardiovascular disease. It incorporates additional risk factors compared to earlier QRISK versions, including ethnicity, BMI, and family history.

## Overview

QRISK3 was developed to provide more accurate risk prediction for UK populations by including additional risk factors and accounting for ethnic differences in cardiovascular disease risk. The model predicts the 10-year risk of developing cardiovascular disease (CVD) including coronary heart disease and stroke.

## Mathematical Formulation

QRISK3 uses a logistic regression model:

```
Risk = 1 / (1 + exp(-(ln(odds))))
```

Where the log-odds are calculated as:

```
ln(odds) = β₀ + β₁×age + β₂×age² + β₃×TC + β₄×HDL + β₅×SBP + β₆×smoking +
           β₇×diabetes + β₈×BMI + β₉×family_history + β₁₀×ethnicity_adjustment +
           sex-specific adjustments
```

## Risk Factors

- **Age**: 25-84 years (optimal range)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Presence of diabetes (yes/no)
- **BMI**: Body mass index (kg/m²)
- **Family History**: CVD in first-degree relative before age 60 (yes/no)
- **Ethnicity**: White, South Asian, Black, Chinese, Mixed, Other

## Coefficients

### Base Coefficients (Applied to All Individuals)

| Risk Factor | Coefficient (β) | Notes |
|-------------|-----------------|-------|
| Age | 0.0799 | Linear age term |
| Age² | -0.0005 | Quadratic age term |
| Total Cholesterol | 0.1909 | mmol/L |
| HDL Cholesterol | -0.2624 | mmol/L (protective) |
| Systolic BP | 0.0195 | mmHg |
| Smoking | 0.8807 | Current smoker |
| Diabetes | 1.2226 | Type 1 or Type 2 |
| BMI | 0.0211 | kg/m² |
| Family History | 0.5113 | First-degree relative <60 |
| Constant | -24.675 | Intercept |

### Ethnicity Adjustments (Added to Base Score)

| Ethnicity | Coefficient (β) |
|-----------|-----------------|
| White | 0 (reference) |
| South Asian | 1.3256 |
| Black | 0.2437 |
| Chinese | -0.4158 |
| Mixed | 0.3847 |
| Other | 0.1985 |

### Sex-Specific Adjustments

#### Female Adjustments
- Age adjustment: +0.1767
- Smoking × Age interaction: -0.0125

#### Male Adjustments
- Age adjustment: +0.1538
- Smoking × Age interaction: -0.0105

## Baseline Survival Probabilities (10-year)

- **Male**: 0.9186
- **Female**: 0.9653

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: ≥20%

## Reference

Hippisley-Cox J, Coupland C, Brindle P. Development and validation of QRISK3 risk prediction algorithms to estimate future risk of cardiovascular disease: prospective cohort study. *BMJ*. 2017;357:j2099.

DOI: [10.1136/bmj.j2099](https://doi.org/10.1136/bmj.j2099)

## Implementation Notes

- Model validated for ages 25-84 years
- Includes ethnicity-specific adjustments for more accurate risk prediction
- Requires additional factors (BMI, family history) for optimal accuracy
- UK-based algorithm but widely applicable to other populations
- Risk estimates are for 10-year period
- All lipid values should be in mmol/L
- BMI is an important risk factor that should be included when available
