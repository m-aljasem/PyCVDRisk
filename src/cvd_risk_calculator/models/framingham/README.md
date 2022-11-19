# Framingham Risk Score

The Framingham Risk Score is one of the most widely used cardiovascular risk prediction models, developed from the Framingham Heart Study. It estimates 10-year risk of cardiovascular disease using a point-based system with traditional risk factors.

## Overview

The Framingham Risk Score predicts the 10-year risk of developing coronary heart disease (CHD) or cardiovascular disease (CVD) based on major risk factors identified from the Framingham Heart Study cohort.

The model uses a simplified point-based system that assigns points to different risk factor categories, then converts the total points to a risk percentage based on age and sex.

## Risk Factors

- **Age**: 30-79 years (optimal range)
- **Sex**: Male/Female
- **Total Cholesterol**: mg/dL
- **HDL Cholesterol**: mg/dL
- **Systolic Blood Pressure**: mmHg
- **Smoking Status**: Current smoker (yes/no)
- **Diabetes**: Presence of diabetes (for some variants)

## Point System

### Age Points (Male)

| Age Range | Points |
|-----------|--------|
| 30-34 | 0 |
| 35-39 | 2 |
| 40-44 | 5 |
| 45-49 | 6 |
| 50-54 | 8 |
| 55-59 | 10 |
| 60-64 | 11 |
| 65-69 | 12 |
| 70-74 | 14 |

### Age Points (Female)

| Age Range | Points |
|-----------|--------|
| 30-34 | 0 |
| 35-39 | 2 |
| 40-44 | 4 |
| 45-49 | 5 |
| 50-54 | 7 |
| 55-59 | 8 |
| 60-64 | 9 |
| 65-69 | 10 |
| 70-74 | 11 |

### Total Cholesterol Points (Male)

| Cholesterol (mg/dL) | Points (by Age) |
|-------------------|----------------|
| <160 | -2, 0, 1* |
| 160-199 | 0, 1, 2 |
| 200-239 | 1, 3, 4 |
| 240-279 | 2, 4, 5 |
| ≥280 | 3, 5, 6 |

*Points vary by age group: <50, 50-59, ≥60 years

### Total Cholesterol Points (Female)

| Cholesterol (mg/dL) | Points (by Age) |
|-------------------|----------------|
| <160 | -2, 0, 1* |
| 160-199 | 0, 1, 3 |
| 200-239 | 1, 3, 4 |
| 240-279 | 3, 4, 5 |
| ≥280 | 4, 6, 7 |

*Points vary by age group: <50, 50-59, ≥60 years

### HDL Cholesterol Points (Male)

| HDL (mg/dL) | Points |
|-------------|--------|
| <35 | 5 |
| 35-44 | 2 |
| 45-49 | 1 |
| 50-59 | 0 |
| ≥60 | -2 |

### HDL Cholesterol Points (Female)

| HDL (mg/dL) | Points |
|-------------|--------|
| <35 | 7 |
| 35-44 | 4 |
| 45-49 | 2 |
| 50-59 | 1 |
| ≥60 | -1 |

### Blood Pressure Points (Male)

| BP Category | Points (Untreated/Treated) |
|-------------|---------------------------|
| Optimal (<120/<80) | 0/0 |
| Normal (120-129/<80) | 0/1 |
| High Normal (130-139/80-89) | 1/2 |
| Stage 1 (140-159/90-99) | 2/3 |

### Blood Pressure Points (Female)

| BP Category | Points (Untreated/Treated) |
|-------------|---------------------------|
| Optimal (<120/<80) | 0/0 |
| Normal (120-129/<80) | 0/1 |
| High Normal (130-139/80-89) | 1/2 |
| Stage 1 (140-159/90-99) | 3/4 |

## Risk Calculation

The total points are calculated by summing points from all risk factors. The 10-year risk percentage is then looked up from age- and sex-specific tables.

### 10-Year Risk by Points and Age Group (Male)

| Points | 30-39 | 40-49 | 50-59 | 60-69 | 70-79 |
|--------|--------|--------|--------|--------|--------|
| 0 | 1 | 2 | 3 | 5 | 8 |
| 1 | 1 | 2 | 3 | 5 | 8 |
| 2 | 1 | 3 | 4 | 6 | 9 |
| 3 | 1 | 3 | 5 | 7 | 10 |
| 4 | 1 | 4 | 6 | 8 | 11 |
| 5 | 2 | 5 | 7 | 9 | 12 |
| 6 | 2 | 6 | 8 | 10 | 13 |
| 7 | 3 | 8 | 10 | 11 | 14 |
| 8 | 4 | 10 | 12 | 13 | 15 |
| 9 | 5 | 12 | 14 | 15 | 17 |
| 10 | 6 | 14 | 16 | 17 | 18 |
| 11 | 8 | 16 | 19 | 20 | 20 |
| 12 | 10 | 18 | 22 | 24 | 22 |
| 13 | 12 | 20 | 25 | 27 | 24 |
| 14 | 16 | 25 | 29 | 29 | 27 |
| 15 | 20 | 30 | 35 | 33 | 29 |
| 16 | 25 | 35 | 40 | 38 | 32 |

### 10-Year Risk by Points and Age Group (Female)

| Points | 30-39 | 40-49 | 50-59 | 60-69 | 70-79 |
|--------|--------|--------|--------|--------|--------|
| 0 | 1 | 2 | 3 | 5 | 8 |
| 1 | 1 | 2 | 3 | 5 | 8 |
| 2 | 1 | 2 | 4 | 6 | 9 |
| 3 | 1 | 3 | 5 | 7 | 10 |
| 4 | 2 | 4 | 6 | 8 | 11 |
| 5 | 2 | 5 | 7 | 9 | 12 |
| 6 | 3 | 6 | 8 | 10 | 13 |
| 7 | 4 | 8 | 10 | 12 | 15 |
| 8 | 5 | 10 | 12 | 14 | 16 |
| 9 | 6 | 12 | 14 | 16 | 18 |
| 10 | 8 | 15 | 17 | 19 | 20 |
| 11 | 11 | 18 | 21 | 22 | 22 |
| 12 | 14 | 22 | 25 | 26 | 24 |
| 13 | 17 | 27 | 30 | 30 | 27 |
| 14 | 22 | 32 | 35 | 34 | 29 |
| 15 | 27 | 37 | 40 | 38 | 32 |
| 16 | 30 | 40 | 43 | 42 | 34 |

## Risk Categories

- **Low Risk**: <10%
- **Moderate Risk**: 10-20%
- **High Risk**: >20%

## Reference

Wilson PW, D'Agostino RB, Levy D, Belanger AM, Silbershatz H, Kannel WB. Prediction of coronary heart disease using risk factor categories. *Circulation*. 1998;97(18):1837-47.

DOI: [10.1161/01.CIR.97.18.1837](https://doi.org/10.1161/01.CIR.97.18.1837)

## Implementation Notes

- Model validated for ages 30-79 years
- This is a simplified implementation using the point-based system
- The full Framingham model includes multiple variants (hard CHD, all CVD, stroke-specific, etc.)
- Risk estimates are for 10-year period
- All lipid values should be in mg/dL
- Blood pressure treatment status affects point assignment
