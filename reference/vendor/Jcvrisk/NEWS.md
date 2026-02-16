# Jcvrisk 0.1.3 (2025-10-29)

- Add 8 new functions regarding JALS score.
  -`jals_stroke_wECG()` : 5-year risk calculation for Stroke based on the JALS study with electrocardiogram model
  -`jals_stroke_woECG()`: 5-year risk calculation for Stroke based on the JALS study without electrocardiogram model
  -`jals_ami_wECG()`: 5-year risk calculation for acute myocardial infarction (AMI) based on the JALS study with electrocardiogram model
  -`jals_ami_woECG()`: 5-year risk calculation for acute myocardial infarction (AMI) based on the JALS study without electrocardiogram model
  -`jals_composit_wECG()`: 5-year risk calculation for composite outcome (stroke + acute myocardial infarction) based on the JALS study with electrocardiogram model
  -`jals_composit_woECG()`: 5-year risk calculation for composite outcome (stroke + acute myocardial infarction) based on the JALS study without electrocardiogram model
  -`jals_cvd_wECG()`: 5-year risk calculation for cardiovascular disease event based on the JALS study with electrocardiogram model
  -`jals_cvd_woECG()`: 5-year risk calculation for cardiovascular disease event based on the JALS study without electrocardiogram model

- Unify ambiguous sex calling
  - The definition of sex was changed from sex[male == 1, female == 2] to male[male ==1, female ==0].


# Jcvrisk 0.1.2 (2025-04-20)

- Fixed a calculation error in the `suita_wECG()` and `suita_ECG()` function for estimated 10-year risk values.
  - Risk calculation about sex was corrected from [male=0, female =1] to [male=1, female =2].


# Jcvrisk 0.1.1 (2025-04-18)

- Fixed a calculation error in the `suita_wECG()` function for estimated 10-year risk values.
  - For individuals with SCORE = 30–35: the risk estimate was corrected from 14% to 15%.
  - For individuals with SCORE ≥ 36: the risk estimate was corrected from 25% to 26%.


# Jcvrisk 0.1.0 (2024-10-22)

* Initial CRAN submission.
