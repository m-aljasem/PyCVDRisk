# SMART2 (Second Manifestations of ARTerial disease)

SMART2 is a risk prediction model specifically designed for patients with established cardiovascular disease. It estimates the risk of recurrent cardiovascular events (myocardial infarction, stroke, or vascular death) in secondary prevention populations.

## Overview

SMART2 (Second Manifestations of ARTerial disease) was developed to provide accurate risk prediction for patients who have already experienced a cardiovascular event. Unlike primary prevention models, SMART2 predicts the risk of recurrent events in patients with established CVD.

The model was developed using data from the SMART (Second Manifestations of ARTerial disease) cohort study, which followed patients with established arterial disease.

## Mathematical Formulation

SMART2 uses a logistic regression model:

```
Risk = 1 / (1 + exp(-(ln(odds))))
```

Where the log-odds are calculated as:

```
ln(odds) = β₀ + β₁×age + β₂×sex + β₃×SBP + β₄×HDL + β₅×TC + β₆×smoking + β₇×diabetes
```

## Risk Factors

- **Age**: Years (typically 40-80)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Presence of diabetes (yes/no)

## Patient Population

SMART2 is specifically designed for patients with **established cardiovascular disease**, including:

- Coronary artery disease
- Cerebrovascular disease
- Peripheral artery disease
- Abdominal aortic aneurysm

## Coefficients

| Risk Factor | Coefficient (β) | Notes |
|-------------|-----------------|-------|
| Age | 0.028 | Years |
| Male Sex | 0.261 | Male vs Female |
| Systolic BP | 0.008 | mmHg |
| HDL Cholesterol | -0.458 | mmol/L (protective) |
| Total Cholesterol | 0.152 | mmol/L |
| Smoking | 0.371 | Current smoker |
| Diabetes | 0.456 | Type 1 or Type 2 |
| Constant | -5.432 | Intercept |

## Baseline Survival Probability

- **10-year survival**: 0.85 (approximation)

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: 20-30%
- **Very High Risk**: ≥30%

## Reference

Dorresteijn JA, Visseren FL, Wassink AM, et al. Development and validation of a prediction rule for recurrent vascular events and death in patients with established cardiovascular disease. *European Heart Journal*. 2014;35(29):1925-31.

DOI: [10.1093/eurheartj/euh286](https://doi.org/10.1093/eurheartj/euh286)

## Implementation Notes

- **Secondary prevention model only** - designed for patients with established CVD
- Risk estimates are for recurrent events, not primary prevention
- Model validated in secondary prevention populations
- All lipid values should be in mmol/L
- Risk estimates are for 10-year period
- Particularly useful for treatment intensification decisions in secondary prevention
