# SMART2: verification report

Model: SMART2 (Hageman et al., Eur Heart J 2022;43:1715-27, doi:10.1093/eurheartj/ehac056), 10-year risk of
recurrent myocardial infarction, stroke or vascular death in people aged 40-80 with established atherosclerotic
disease, recalibrated to 4 European and 5 other regions. `cvd_risk.smart2(...)`.

Evidence: level A, the official U-Prevent calculator (u-prevent.com), two runs entered through its web form by the
owner with a browser assistant. Sources: Supplementary Table 2 (coefficients, yearly baseline risks,
expected/observed ratios, mean linear predictor) and the article's Table 2 (hazard ratios and their units).

## What the supplement does not say

Supplementary Table 2 used as printed gives implausible risks (98% for an ordinary 65-year-old man with CAD).
Two corrections reproduce the calculator:

1. **SBP is per 10 mmHg.** The article's Table 2 gives the hazard ratio per 10 mmHg; the supplement's beta
   (0.018913154) carries no unit. With SBP/10 every one-change patient below moves as the calculator does.
2. **A constant of 0.211 is missing.** After (1), the calculator's linear predictor is higher than ours by the same
   amount for every patient. The values consistent with all 30 round-1 results at display precision lie between
   0.2104 and 0.2111. This range contains -ln(0.81) = 0.21072, the model's fixed antithrombotic offset, which the
   package uses (`_LP_MEAN = -0.0463729 - 0.21072103`). The likely reading is that the printed mean linear
   predictor was computed with the offset, but this is inferred, not stated by the authors.

## Checks (`tests/smart2/test_smart2.py`)

1. A hand calculation from Supplementary Table 2 (with 1 and 2) equals the package to 1e-12.
2. Risk rises with the horizon (1-10 years) and falls with antithrombotic use; invalid rows give NaN.
3. **U-Prevent round 1: 30/30 patients reproduced** at 0.1% (`fixtures/uprevent_results.jsonl`): a base patient,
   18 one-change variants (every predictor, the very-high-risk region), 11 random patients. Patient 21's total
   cholesterol was raised to the calculator's minimum (2.5 mmol/L); the test uses the value entered.
4. **U-Prevent round 2: 20/20 comparable patients reproduced** (`fixtures/uprevent_results2.jsonl`): HbA1c 33-108
   in two diabetic profiles (no effect on the risk; HbA1c is not a SMART2 predictor), and every input at the
   calculator's limits (age 40/80, eGFR 30/120, hsCRP 0.1/15, 30 years since the event, SBP 95/200,
   total cholesterol 8.0 with HDL 0.6). Patient 100 repeats round-1 patient 0 (26.1% both times).

`uprevent.py` regenerates the patient sets and the instructions given to the browser assistant
(`fixtures/uprevent_prompt*.md`).

## Not covered

- U-Prevent's "dual antiplatelet therapy" and "aspirin + low-dose DOAC" (round-2 patients 120-121: 17.4% and 17.0%
  against 21.7% with aspirin) add treatment effects beyond SMART2's single antithrombotic offset (implied extra
  hazard ratios about 0.78 and 0.76). The package models aspirin or equivalent only.
- The calculator offers only the four European regions and only the 10-year horizon, so the Asia, Australia,
  Japan, North America and Latin America ratios and the 1-9-year baselines are checked against the supplement
  only.
- Round 1 used LDL = non-HDL - 0.1 and HbA1c = 53 as placeholders where the calculator required them; round 2
  shows neither changes the 10-year risk.
