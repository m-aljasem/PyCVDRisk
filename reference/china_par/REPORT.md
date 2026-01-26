# China-PAR (2016): verification report

Evidence level A + C: the official calculator (displayed values) and the paper's Supplemental Table 1 and Table 2.

## Sources

- **Paper**: Yang X, Li J, Hu D, et al. Predicting the 10-year risks of atherosclerotic cardiovascular disease in
  Chinese population: the China-PAR project. Circulation 2016;134:1430-40. doi:10.1161/CIRCULATIONAHA.116.022367
  (`source/yang2016_circulation.pdf`, accepted manuscript with supplement; sha256 95013287...b8c8). Supplemental
  Table 1 prints the sex-specific coefficients (2 decimals), the mean coefficient x value (140.68 men, 117.26 women)
  and the 10-year baseline survival (0.97, 0.99); Supplemental Table 2 the formula and the worked example (men, 60).
- **Official calculator**: National Center for Cardiovascular Diseases / Fuwai Hospital,
  https://www.cvdrisk.com.cn/ASCVD/Eval. It computes on the server (no formula in the page) and shows the 10-year
  risk, the risk under ideal risk factors (one decimal) and, below age 60, a lifetime risk from a separate model.
- Also in `source/`: Yang 2016 (Chronic Dis Transl Med; risk categories < 5%, 5-10%, >= 10%) and Li 2019 (Biomed
  Environ Sci; applies the equations with S10 = 0.97 / 0.99). Neither prints more digits.
- **Unofficial calculator**: appcardio.com/china-par-cardiovascular-risk-calculator (JavaScript)
  uses the printed 2-decimal coefficients, S10 and means as they stand.

## Model

risk = 1 - exp(-exp(intercept + LP)); LP = coefficients x (ln age, ln SBP treated or untreated, ln TC, ln HDL,
ln waist, smoker, diabetes, north, urban, family history, ln age x ln SBP, ln age x smoker, ln age x family
history); women have no urban, family-history or ln age x smoker terms. intercept = ln(-ln S10) - mean.

## The printed coefficients do not reproduce the calculator

At face value Supplemental Table 1 gives the paper's own example (man, 60) as 11.9% (Supplemental Table 2 prints
11.0%, the calculator shows 10.8%), and reproduces 15 of the calculator's 550 shown values, missing by up to 3.4 points in men and 12.7 in
women (`fit_calculator.py`). With 2 decimals on coefficients multiplying ln-values near 4 and their products near
20, and S10 to 2 decimals, the printed table cannot be used as it stands.

## Coefficients recovered from the calculator (evidence A)

`query_calculator.py` submitted seeded synthetic patients through the form (age 30-80, both sexes, every flag,
continuous inputs across the form ranges); `fixtures/calculator_reference.csv` stores what the page showed.

- Seed 0 (400 patients: 203 men, 197 women) was used for recovery. On the scale ln(-ln(1 - risk)) the model is
  linear, and each shown value bounds the linear predictor to an interval (shown +- 0.05 points). A linear program
  (`fit_calculator.py`) finds that coefficients within 0.005 of every printed value, with one free intercept per sex,
  satisfy all 400 intervals; `fixtures/recovered_coefficients.json` is the point farthest from the interval walls.
  Several sit on the +- 0.005 bound: this is one consistent set, not the developers' unpublished full-precision
  values (`fixtures/recovered_coefficients_seed0.json`).
- Seed 1 (150 patients: 78 men, 72 women) was queried afterwards and held out. The seed-0 set gives 144/150 shown
  values; the other 6 (ages 61-80) are one decimal off, each unrounded value within 0.015 point of the rounding edge.
- The package's `COEF` is the same recovery on all 550 patients (`fixtures/recovered_coefficients.json`, still within
  0.005 of every printed coefficient): 550/550 shown values equal. Expect about 4% of new patients to land one
  decimal off at a rounding edge (the held-out rate).
- The 8 profiles of Table 2: the calculator equals ours 8/8 (`fixtures/table2.csv`).

`tests/china_par/test_china_par.py` checks all of this.

## Findings

1. **Printed precision.** Supplemental Table 1 (2-decimal coefficients, S10 0.97 / 0.99) does not reproduce the
   official calculator: up to 12.7 points off in women. The appcardio.com calculator uses it as printed and inherits
   the error.
2. **Table 2 differs from the calculator** in 5 of 8 values: men 50, 60, 70 years 5.4, 11.0, 19.6% (calculator
   5.3, 10.8, 19.3); women 60, 70 years 10.1, 17.1% (calculator 10.2, 17.2).
3. **Worked example, women.** Supplemental Table 1's example (stated: with diabetes) enters diabetes = 0 in the
   women's column; its sum (119.22) gives 6.9% with the printed values, against the 10.1% printed beside it.
4. **Not implemented.** The lifetime risk the calculator shows below age 60 (a separate China-PAR model, not in these
   sources) and the "ideal risk factor" value.

Calculator queried.
