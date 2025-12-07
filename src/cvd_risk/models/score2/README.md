# SCORE2 (Systematic COronary Risk Evaluation 2)

SCORE2 is the updated European cardiovascular risk prediction model published in 2021 by the ESC Cardiovascular Risk Collaboration. It estimates 10-year risk of fatal and non-fatal cardiovascular disease for European populations.

## Overview

SCORE2 replaces the original SCORE model and provides more accurate risk predictions by:
- Including both fatal and non-fatal cardiovascular events
- Using region-specific baseline hazards
- Incorporating contemporary European cohort data
- Providing separate equations for men and women

## Mathematical Formulation

The SCORE2 model calculates 10-year risk of CVD using transformed risk factors and sex-specific coefficients, followed by region-specific calibration.

### Risk Calculation Steps

1. **Transform risk factors** according to the formulas below
2. **Calculate uncalibrated risk**: `10-year risk = 1 - (baseline survival)^exp(x)`
   where `x = Σ[β × (transformed variables)]`
3. **Apply calibration**: `Calibrated 10-year risk (%) = [1 - exp(-exp(scale₁ + scale₂ × ln(-ln(1 - 10-year risk))))] × 100`

### Risk Factor Transformations

## Risk Factors

- **Age**: 40-69 years (optimal range)
- **Sex**: Male/Female
- **Systolic Blood Pressure**: mmHg
- **Total Cholesterol**: mmol/L
- **HDL Cholesterol**: mmol/L
- **Smoking Status**: Current smoker (yes/no)

## Regions

SCORE2 provides region-specific risk estimates:

- **Low risk**: Denmark, Norway, Sweden
- **Moderate risk**: Austria, Belgium, Cyprus, Finland, France, Germany, Greece, Iceland, Ireland, Italy, Luxembourg, Malta, Netherlands, Portugal, Slovenia, Spain, Switzerland
- **High risk**: Albania, Bulgaria, Czech Republic, Estonia, Hungary, Latvia, Lithuania, Montenegro, Poland, Romania, Serbia, Slovakia
- **Very high risk**: High-risk countries

## Coefficients and Transformations

### Risk Factor Transformations
| Risk Factor | Transformation | Variable |
|-------------|----------------|----------|
| Age (years) | `cage = (age - 60) / 5` | cage |
| Smoking | `current = 1, other = 0` | smoking |
| SBP (mm Hg) | `csbp = (sbp - 120) / 20` | csbp |
| Total Cholesterol (mmol/L) | `ctchol = tchol - 6` | ctchol |
| HDL Cholesterol (mmol/L) | `chdl = (hdl - 1.3) / 0.5` | chdl |

### Sex-Specific Coefficients
| Risk Factor | Transformation | Male (β) | Female (β) |
|-------------|----------------|----------|------------|
| Age | cage | 0.3742 | 0.4648 |
| Smoking | smoking | 0.6012 | 0.7744 |
| SBP | csbp | 0.2777 | 0.3131 |
| Total Cholesterol | ctchol | 0.1458 | 0.1002 |
| HDL Cholesterol | chdl | -0.2698 | -0.2606 |
| Smoking × Age | smoking × cage | -0.0755 | -0.1088 |
| SBP × Age | csbp × cage | -0.0255 | -0.0277 |
| Total Cholesterol × Age | ctchol × cage | -0.0281 | -0.0226 |
| HDL Cholesterol × Age | chdl × cage | 0.0426 | 0.0613 |

### Baseline Survival (10-year)
- **Male**: 0.9605
- **Female**: 0.9776

## Region-Specific Calibration Scales

The model uses region and sex-specific calibration scales to adjust the uncalibrated risk:

| Region | Male Scale₁ | Male Scale₂ | Female Scale₁ | Female Scale₂ |
|--------|-------------|-------------|---------------|---------------|
| Low | -0.5699 | 0.7476 | -0.7380 | 0.7019 |
| Moderate | -0.1565 | 0.8009 | -0.3143 | 0.7701 |
| High | 0.3207 | 0.9360 | 0.5710 | 0.9369 |
| Very High | 0.5836 | 0.8294 | 0.9412 | 0.8329 |

### Risk Regions Definition
- **Low**: Denmark, Norway, Sweden
- **Moderate**: Austria, Belgium, Cyprus, Finland, France, Germany, Greece, Iceland, Ireland, Italy, Luxembourg, Malta, Netherlands, Portugal, Slovenia, Spain, Switzerland
- **High**: Albania, Bulgaria, Czech Republic, Estonia, Hungary, Latvia, Lithuania, Montenegro, Poland, Romania, Serbia, Slovakia
- **Very High**: High-risk countries

## Risk Categories

- **Low**: <5%
- **Moderate**: 5-10%
- **High**: 10-20%
- **Very High**: ≥20%

## Reference

SCORE2 Working Group and ESC Cardiovascular Risk Collaboration. (2021). SCORE2 risk prediction algorithms: new models to estimate 10-year risk of cardiovascular disease in Europe. *European Heart Journal*, 42(25), 2439-2454.

DOI: [10.1093/eurheartj/ehab309](https://doi.org/10.1093/eurheartj/ehab309)

## Implementation Notes

- Model validated for ages 40-69 years; extrapolation outside this range may have reduced accuracy
- Region selection is critical; use appropriate region based on country/population
- The coefficients shown above are the exact values from the SCORE2 publication
- Non-HDL cholesterol can be approximated as total cholesterol - HDL
- Risk predictions are for 10-year period
- Current implementation may use approximations; refer to publication for exact calculations
