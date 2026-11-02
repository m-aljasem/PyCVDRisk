# PREDICT (New Zealand): verification report

Generated 2026-11-01 by `reference/predict/report.py`.

Standard implementation: the VIEW team's R package PredictRiskScores 0.9 (University of Auckland; vendored
unmodified in `reference/vendor/PredictRiskScores`, commit in `COMMIT`). Papers: Pylypchuk 2018 (Lancet,
`source/pylypchuk2018.pdf`) and Selak 2019 (Ann Intern Med, `source/selak2019.pdf`).

## 1. Coefficients

Every number in each R function (coefficients, centring values, ACR shift, baseline survival, mean linear
predictor), parsed from the source text, against `coefficients.py`:

| Model | R function | Numbers | Identical |
|---|---|---:|---|
| cvd | `NoPriorCVDRisk` | 48 | yes |
| cvd_bmi | `NoPriorCVDRisk_BMI` | 60 | yes |
| cvd_policy | `NoPriorCVDRisk_Policy` | 40 | yes |
| t2d | `PriorT2DRisk` | 62 | yes |
| bleed | `NoPriorCVDBleedRisk` | 50 | yes |
| post_cvd | `PostCVDRisk` | 38 | yes |
| post_acs | `PostACSRisk` | 40 | yes |

Printed tables against `coefficients.py`: 49/49 rows identical to the last printed digit.

| Source | Row | Printed (women, men) | Package | |
|---|---|---|---|---|
| Pylypchuk 2018 T3 | Age | (0.0756412, 0.0675532) | (0.0756412, 0.0675532) | ok |
| Pylypchuk 2018 T3 | Māori | (0.3910183, 0.2899054) | (0.3910183, 0.2899054) | ok |
| Pylypchuk 2018 T3 | Pacific | (0.2010224, 0.1774195) | (0.2010224, 0.1774195) | ok |
| Pylypchuk 2018 T3 | Indian | (0.1183427, 0.2902049) | (0.1183427, 0.2902049) | ok |
| Pylypchuk 2018 T3 | Chinese or other Asian | (-0.28551, -0.3975687) | (-0.28551, -0.3975687) | ok |
| Pylypchuk 2018 T3 | NZDep quintile | (0.1080795, 0.0794903) | (0.1080795, 0.0794903) | ok |
| Pylypchuk 2018 T3 | Ex-smoker | (0.087476, 0.0753246) | (0.087476, 0.0753246) | ok |
| Pylypchuk 2018 T3 | Smoker | (0.6226384, 0.5058041) | (0.6226384, 0.5058041) | ok |
| Pylypchuk 2018 T3 | cardiovascular disease | (0.0445534, 0.1326587) | (0.0445534, 0.1326587) | ok |
| Pylypchuk 2018 T3 | Atrial fibrillation | (0.8927126, 0.5880131) | (0.8927126, 0.5880131) | ok |
| Pylypchuk 2018 T3 | Diabetes | (0.5447632, 0.5597023) | (0.5447632, 0.5597023) | ok |
| Pylypchuk 2018 T3 | SBP per 1 mm Hg | (0.0136606, 0.0163778) | (0.0136606, 0.0163778) | ok |
| Pylypchuk 2018 T3 | TC/HDL | (0.1226753, 0.1283758) | (0.1226753, 0.1283758) | ok |
| Pylypchuk 2018 T3 | OBPLM* | (0.339925, 0.2947634) | (0.339925, 0.2947634) | ok |
| Pylypchuk 2018 T3 | OLLM | (-0.0593798, -0.0537314) | (-0.0593798, -0.0537314) | ok |
| Pylypchuk 2018 T3 | OATM | (0.1172496, 0.0934141) | (0.1172496, 0.0934141) | ok |
| Pylypchuk 2018 T3 | Age × diabetes | (-0.0222549, -0.020235) | (-0.0222549, -0.020235) | ok |
| Pylypchuk 2018 T3 | Age × SBP | (-0.0004425, -0.0004184) | (-0.0004425, -0.0004184) | ok |
| Pylypchuk 2018 T3 | OBPLM × SBP | (-0.004313, -0.0053077) | (-0.004313, -0.0053077) | ok |
| Pylypchuk 2018 T3 | mean Age | (56.13665, 51.79953) | (56.13665, 51.79953) | ok |
| Pylypchuk 2018 T3 | mean NZDep quintle | (2.990826, 2.972793) | (2.990826, 2.972793) | ok |
| Pylypchuk 2018 T3 | mean SBP | (129.0173, 129.1095) | (129.0173, 129.1095) | ok |
| Pylypchuk 2018 T3 | mean TC/HDL | (3.726268, 4.38906) | (3.726268, 4.38906) | ok |
| Pylypchuk 2018 T3 | baseline survival | (0.983169213058, 0.974755526232) | (0.983169213058, 0.974755526232) | ok |
| Selak 2019 T4 | Age, per year | (0.03502806, 0.03538036) | (0.03502806, 0.03538036) | ok |
| Selak 2019 T4 | ori | (0.311582316, 0.40955001) | (0.311582316, 0.40955001) | ok |
| Selak 2019 T4 | Paciﬁc | (0.291826502, 0.52687151) | (0.291826502, 0.52687151) | ok |
| Selak 2019 T4 | Indian | (-0.17017867, -0.01815411) | (-0.17017867, -0.01815411) | ok |
| Selak 2019 T4 | Chinese or other Asian | (0.044890076, 0.37798865) | (0.044890076, 0.37798865) | ok |
| Selak 2019 T4 | NZDep quintile, per 1 quintile | (0.098736992, 0.09305327) | (0.098736992, 0.09305327) | ok |
| Selak 2019 T4 | Former smoker | (0.144844011, 0.15536803) | (0.144844011, 0.15536803) | ok |
| Selak 2019 T4 | Current smoker | (0.495240401, 0.38226181) | (0.495240401, 0.38226181) | ok |
| Selak 2019 T4 | Family history of premature CVD | (0.055185249, 0.05028066) | (0.055185249, 0.05028066) | ok |
| Selak 2019 T4 | Diabetes | (0.182633821, 0.17500777) | (0.182633821, 0.17500777) | ok |
| Selak 2019 T4 | Cancer | (0.299418027, 0.56636099) | (0.299418027, 0.56636099) | ok |
| Selak 2019 T4 | Prior bleeding event | (1.157219678, 1.14121551) | (1.157219678, 1.14121551) | ok |
| Selak 2019 T4 | Peptic ulcer disease (nonbleeding) | (0.426358755, 0.22227113) | (0.426358755, 0.22227113) | ok |
| Selak 2019 T4 | Alcohol-related condition | (0.95065995, 0.67405759) | (0.95065995, 0.67405759) | ok |
| Selak 2019 T4 | Chronic liver disease or pancreatitis | (0.979437007, 0.77588662) | (0.979437007, 0.77588662) | ok |
| Selak 2019 T4 | SBP, per mm Hg | (0.004991576, 0.00373758) | (0.004991576, 0.00373758) | ok |
| Selak 2019 T4 | Ratio of total-HDL cholesterol, per 1 unit | (0.001878851, -0.05009861) | (0.001878851, -0.05009861) | ok |
| Selak 2019 T4 | Blood pressure-lowering | (0.140874933, 0.20741834) | (0.140874933, 0.20741834) | ok |
| Selak 2019 T4 | Lipid-lowering | (0.010545182, -0.04636764) | (0.010545182, -0.04636764) | ok |
| Selak 2019 T4 | Peptic ulcer disease | (0.370528961, 0.36282612) | (0.370528961, 0.36282612) | ok |
| Selak 2019 T4 | Nonsteroidal anti-inﬂammatory | (0.10655804, 0.17279428) | (0.10655804, 0.17279428) | ok |
| Selak 2019 T4 | Corticosteroid | (0.328347624, 0.35261644) | (0.328347624, 0.35261644) | ok |
| Selak 2019 T4 | Selective serotonin reuptake inhibitor | (0.16495507, 0.2928215) | (0.16495507, 0.2928215) | ok |
| Selak 2019 T4 | Baseline survival at 5 y | (0.98902929, 0.9886172) | (0.98902929, 0.9886172) | ok |
| Selak 2019 T4 | Mean prognostic index‡ | (3.262378, 2.787439) | (3.262378, 2.787439) | ok |

## 2. Worked examples

| Example | Package | Printed | |
|---|---:|---:|---|
| Pylypchuk 2018 panel: European woman 55, diabetes, ex-smoker, NZDep 3, SBP 135, TC/HDL 5, on BP drugs | 0.051072713 | 0.0511 | ok |
| Selak 2019 Table 4: European man 65, diabetes, former smoker, NZDep 3, SBP 130, TC/HDL 5, BP and lipid drugs | 0.019038994 | 0.019038994 | ok |

## 3. Comparison with the authors' R code

`run_reference.R` runs the seven R functions unmodified on `fixtures/<model>_patients.csv` (3,000 patients
each, drawn where the R code behaves as documented: whole-number ages, BMI, creatinine, HbA1c and days;
lower-case sex) with `dp = 12`.

| Model | Patients | Max abs difference | |
|---|---:|---:|---|
| cvd | not run | | |
| cvd_bmi | not run | | |
| cvd_policy | not run | | |
| t2d | not run | | |
| bleed | not run | | |
| post_cvd | not run | | |
| post_acs | not run | | |

**Not run yet.** Executing the downloaded R code needs the owner's permission (see
`reference/README.md`). Until then the evidence for the T2D, BMI, policy and post-event models is the
parsed coefficients only; the 2018 and bleeding models also rest on their papers.

## 4. Defects in the R code (found by reading it)

| Where | R behaviour | This package |
|---|---|---|
| All primary-prevention models | `values$age[which(values$age > 79)] <- 80` clamps ages above 79 to 80, not 79; ages below 30 become 30. The input check only accepts whole-number ages 18-110. | NaN outside 30-79 (real-valued ages accepted). |
| NoPriorCVDRisk_BMI | `bmi %in% 25:29.9` (and 30:34.9, 35:39.9) is membership in the integers 25..29 etc., so a BMI of 27.3 falls in no band and is scored as the 18.5-25 reference. | Bands [25, 30), [30, 35), [35, 40). |
| PostCVDRisk, PostACSRisk | BMI `%in% 20:24`, `30:34`, `35:39`; HbA1c `>= 40 & <= 64`; creatinine `>= 100 & <= 149`; SBP `>= 120 & <= 139`, `>= 140 & <= 159`; days `>= 182 & <= 365`: non-integer values between bands (BMI 22.3, HbA1c 64.5, creatinine 149.5, SBP 139.5) get the reference category. | Gap-free bands [20, 25), [40, 65), [100, 150), [120, 140), [140, 160), [182, 366) days. |
| PostCVDRisk, PostACSRisk | `male = +(input$sex %in% ok.male)` has no `tolower`, so sex "M" or "Male" is scored as female (the f/m indices elsewhere use `tolower`). | Case-insensitive sex. |
| PriorT2DRisk | `log((acr + 0.01) / 1000)` uses the function argument `acr`, not the column: called with `dat` it evaluates a variable named `acr` in the caller's environment, and the whole vector is assigned to each sex's subset, so mixed-sex batches get other rows' ACR. | ACR taken per row. |
| All | Missing binary inputs (NA) are scored as 0 by default (`allow.na = TRUE`). | NaN. |
| Package metadata | DESCRIPTION says GPL-2; the repository's LICENSE file is GPL-3. | - |

## 5. Not checked

- Pylypchuk 2021 (diabetes model): the Zotero attachment is a different article, so the paper's tables were
  not compared. The VARIANZ policy model (Mehta 2018) and the post-CVD / post-ACS models (Poppe) have no
  paper in the library; for these the R package is the only source.

