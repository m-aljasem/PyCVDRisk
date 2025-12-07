# ASCVD (Atherosclerotic Cardiovascular Disease) Pooled Cohort Equations

The ASCVD Pooled Cohort Equations estimate 10-year risk of atherosclerotic cardiovascular disease for primary prevention. This model was developed by the American College of Cardiology and American Heart Association (ACC/AHA) and published in 2013.

## Overview

The ASCVD model predicts the 10-year risk of developing atherosclerotic cardiovascular disease (ASCVD), which includes:
- Coronary heart disease (CHD)
- Stroke
- Peripheral artery disease

The model uses traditional risk factors and provides sex-specific equations for risk prediction.

## Mathematical Formulation

The ASCVD risk is calculated using a logistic regression model:

```
Risk = 1 / (1 + exp(-(ln(odds))))
```

Where the log-odds are calculated as:

```
ln(odds) = β₀ + β₁×ln(age) + β₂×ln(age)² + β₃×ln(TC) + β₄×ln(age)×ln(TC) +
           β₅×ln(HDL) + β₆×ln(age)×ln(HDL) + β₇×ln(SBP) + β₈×ln(age)×ln(SBP) +
           β₉×smoking + β₁₀×smoking×ln(age)
```

For individuals on antihypertensive treatment, the SBP term is replaced with a treatment-specific coefficient.

## Risk Factors

- **Age**: 40-79 years (optimal range)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mg/dL
- **HDL Cholesterol**: mg/dL
- **Smoking Status**: Current smoker (yes/no)
- **Antihypertensive Treatment**: On treatment (yes/no)

## Coefficients

### Male (White/Non-African American) Coefficients

| Term | Coefficient (β) | Description |
|------|-----------------|-------------|
| Constant | -23.9802 | Intercept |
| ln(Age) | 3.06117 | Age (natural log) |
| ln(Age)² | -1.12370 | Age squared (natural log) |
| ln(Total Cholesterol) | 1.93303 | Total cholesterol (natural log) |
| ln(Age) × ln(Total Cholesterol) | -0.52551 | Age-cholesterol interaction |
| ln(HDL Cholesterol) | -1.47906 | HDL cholesterol (natural log) |
| ln(Age) × ln(HDL Cholesterol) | 0.31583 | Age-HDL interaction |
| ln(SBP) × ln(Age) | 0.93338 | Age-SBP interaction (untreated) |
| ln(SBP) × ln(Age) × Treatment | 1.99881 | Age-SBP-treatment interaction |
| Smoking | 0.65451 | Current smoking |
| Smoking × ln(Age) | -0.57763 | Smoking-age interaction |

### Female (White/Non-African American) Coefficients

| Term | Coefficient (β) | Description |
|------|-----------------|-------------|
| Constant | -26.1931 | Intercept |
| ln(Age) | 2.32888 | Age (natural log) |
| ln(Age)² | -1.12370 | Age squared (natural log) |
| ln(Total Cholesterol) | 1.20904 | Total cholesterol (natural log) |
| ln(Age) × ln(Total Cholesterol) | -0.30784 | Age-cholesterol interaction |
| ln(HDL Cholesterol) | -0.70833 | HDL cholesterol (natural log) |
| ln(Age) × ln(HDL Cholesterol) | 0.39163 | Age-HDL interaction |
| ln(SBP) × ln(Age) | 1.34065 | Age-SBP interaction (untreated) |
| ln(SBP) × ln(Age) × Treatment | 2.00168 | Age-SBP-treatment interaction |
| Smoking | 0.52873 | Current smoking |
| Smoking × ln(Age) | -0.48660 | Smoking-age interaction |

## Baseline Survival Probabilities

- **Male**: 0.91436 (10-year survival probability)
- **Female**: 0.96652 (10-year survival probability)

## Risk Categories

- **Low Risk**: <5%
- **Borderline Risk**: 5-7.4%
- **Intermediate Risk**: 7.5-19.9%
- **High Risk**: ≥20%

## Reference

Goff DC Jr, Lloyd-Jones DM, Bennett G, et al. 2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk: A Report of the American College of Cardiology/American Heart Association Task Force on Practice Guidelines. *Circulation*. 2014;129(25 Suppl 2):S49-73.

DOI: [10.1161/01.cir.0000437741.48606.98](https://doi.org/10.1161/01.cir.0000437741.48606.98)

## Implementation Notes

- Model validated for ages 40-79 years
- This implementation uses white/non-African American coefficients
- Full model includes race-specific equations for African American populations
- Risk estimates are for 10-year period
- Antihypertensive treatment status affects SBP coefficient
- All lipid values should be in mg/dL (not mmol/L)
