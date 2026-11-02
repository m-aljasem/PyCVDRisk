# REGICOR: verification report

Generated 2026-11-01 by `reference/regicor/report.py`.

Evidence level C: Marrugat et al., J Epidemiol Community Health 2003 (`source/marrugat2003.pdf`),
Table 1 (coefficients), Table 2 (Framingham and Gerona means, 10-year event rates) and the
Appendix (formula, one worked example for a man). No independent implementation was available.

## 1. Transcription

39/39 rows of Tables 1-2, rebuilt from `coefficients.py`, are found in the PDF text.

## 2. Bands

Each band at both ends, against a base person aged 50 (TC 180, HDL 55 mg/dL, BP 125/82). The
log-hazard change is recovered from the two risks. SBP bands are tested with DBP 70, DBP bands
with SBP 110 (so the other reading is in the lowest category).

| Sex | Input | Band | Log-hazard change at low / high | Table 1 difference | |
|---|---|---|---|---:|---|
| male | tc | 100-159.99 | -0.380 / -0.380 | -0.38 | ok |
| male | tc | 160-199.99 | +0.000 / +0.000 | +0.00 | ok |
| male | tc | 200-239.99 | +0.570 / +0.570 | +0.57 | ok |
| male | tc | 240-279.99 | +0.740 / +0.740 | +0.74 | ok |
| male | tc | 280-400 | +0.830 / +0.830 | +0.83 | ok |
| male | hdl | 20-34.99 | +0.610 / +0.610 | +0.61 | ok |
| male | hdl | 35-44.99 | +0.370 / +0.370 | +0.37 | ok |
| male | hdl | 45-49.99 | +0.000 / +0.000 | +0.00 | ok |
| male | hdl | 50-59.99 | +0.000 / +0.000 | +0.00 | ok |
| male | hdl | 60-100 | -0.460 / -0.460 | -0.46 | ok |
| male | sbp | 100-119.99 | +0.090 / +0.090 | +0.09 | ok |
| male | sbp | 120-129.99 | +0.000 / +0.000 | +0.00 | ok |
| male | sbp | 130-139.99 | +0.420 / +0.420 | +0.42 | ok |
| male | sbp | 140-159.99 | +0.660 / +0.660 | +0.66 | ok |
| male | sbp | 160-220 | +0.900 / +0.900 | +0.90 | ok |
| male | dbp | 60-79.99 | +0.090 / +0.090 | +0.09 | ok |
| male | dbp | 80-84.99 | +0.000 / +0.000 | +0.00 | ok |
| male | dbp | 85-89.99 | +0.420 / +0.420 | +0.42 | ok |
| male | dbp | 90-99.99 | +0.660 / +0.660 | +0.66 | ok |
| male | dbp | 100-120 | +0.900 / +0.900 | +0.90 | ok |
| female | tc | 100-159.99 | -0.210 / -0.210 | -0.21 | ok |
| female | tc | 160-199.99 | +0.000 / +0.000 | +0.00 | ok |
| female | tc | 200-239.99 | +0.440 / +0.440 | +0.44 | ok |
| female | tc | 240-279.99 | +0.560 / +0.560 | +0.56 | ok |
| female | tc | 280-400 | +0.890 / +0.890 | +0.89 | ok |
| female | hdl | 20-34.99 | +0.730 / +0.730 | +0.73 | ok |
| female | hdl | 35-44.99 | +0.600 / +0.600 | +0.60 | ok |
| female | hdl | 45-49.99 | +0.600 / +0.600 | +0.60 | ok |
| female | hdl | 50-59.99 | +0.000 / +0.000 | +0.00 | ok |
| female | hdl | 60-100 | -0.540 / -0.540 | -0.54 | ok |
| female | sbp | 100-119.99 | -0.740 / -0.740 | -0.74 | ok |
| female | sbp | 120-129.99 | +0.000 / +0.000 | +0.00 | ok |
| female | sbp | 130-139.99 | -0.370 / -0.370 | -0.37 | ok |
| female | sbp | 140-159.99 | +0.220 / +0.220 | +0.22 | ok |
| female | sbp | 160-220 | +0.610 / +0.610 | +0.61 | ok |
| female | dbp | 60-79.99 | -0.740 / -0.740 | -0.74 | ok |
| female | dbp | 80-84.99 | +0.000 / +0.000 | +0.00 | ok |
| female | dbp | 85-89.99 | -0.370 / -0.370 | -0.37 | ok |
| female | dbp | 90-99.99 | +0.220 / +0.220 | +0.22 | ok |
| female | dbp | 100-120 | +0.610 / +0.610 | +0.61 | ok |

40/40 bands give the Table 1 coefficient at both ends.

## 3. Appendix worked example

Man aged 50, diabetic, non-smoker, total cholesterol 210, HDL 52 mg/dL, high-normal blood pressure.

| Step | Framingham | Gerona |
|---|---:|---:|
| sum(b*X), printed 4.02 | 4.0200 | 4.0200 |
| Terms of sum(b*Xmean) written out in the Appendix match Tables 1-2 | yes (14) | yes (14) |
| sum(b*Xmean), printed | 3.84 | 4.32 |
| sum(b*Xmean), those terms added up | 3.6661 | 4.1208 |
| Risk from the printed sums (printed 0.068 / 0.047) | 0.0679 | 0.0260 |
| Same with the sign of the exponent flipped | 0.0478 | 0.0470 |
| Risk from this package | 0.0802 | 0.0317 |
| Age coefficient that would give the printed sum | 0.0536 | 0.0536 |

The Appendix's written-out terms are exactly Table 1 x Table 2, but they add up to 3.67 and 4.12, not the printed 3.84 and 4.32. Both printed sums are
reproduced if the age term uses a coefficient of about 0.0536 instead of the printed 0.05 (the
same value fits both sums), while sum(b*X) = 4.02 uses 0.05. So the example probably mixes an
unrounded age coefficient (means) with the rounded one (patient).

A second slip in the Gerona step: the paper prints exp(4.02-4.32) = 0.741 (correct), then
1 - 0.965^0.741 = 0.047. In fact 1 - 0.965^0.741 = 0.026; 0.047 is 1 - 0.965^exp(+0.30), the
exponent with its sign flipped. The paper's remark that recalibration gives "approximately two
thirds" of the Framingham risk (0.047/0.068) rests on that slip. The Framingham step (0.068) is
consistent with its printed sums.

This package uses Table 1 and Table 2 throughout and gives 0.080 and 0.032.
Owner action: ask the authors for the unrounded coefficients (or check Wilson et al.,
Circulation 1998;97:1837-47, which the function comes from).

## 4. Other notes

- Women, Gerona: Table 2 gives a mean age of 54.2 and a mean squared age of 2931.7, below
  54.2^2 = 2937.6; a mean of squares cannot be below the square of the mean. The
  effect on the linear predictor is 0.0059. The package uses the printed value.
- Women's age coefficients are printed as 0.17 and -0.001 only, and the paper has no example for women.
- Table 1 writes the top bands as ">280" and "S>160"/"D>100"; Table 2 writes ">=280" and ">=160"/">=100".
  The package uses >=, so no value falls between bands.
- v1 of this package used continuous ln-coefficients and a constant that do not appear in this paper.

