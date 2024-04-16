# WHO CVD risk charts (2019 revision) — status

**Blocked: needs data from the authors (methodology step 4).**

Source paper: WHO CVD Risk Chart Working Group. World Health Organization
cardiovascular disease risk charts: revised models to estimate risk in 21
global regions. Lancet Glob Health 2019;7:e1332-45.
doi:10.1016/S2214-109X(19)30318-3 (open access, CC BY; PMC7025029).
Supplementary appendix saved in `source/` (downloaded via the Europe PMC
supplementary-files API; sha256
92b52f66f2d7f4f3cb207d7dbedb14975c758a9e97f09e7982f5d4294ec9cf80).

What the publication gives:
- Log hazard ratios for the lab and non-lab models, MI and stroke, by sex
  (appendix Table 1.6), with centring (age 60, SBP 120, cholesterol 6, BMI 25).
- Baseline survival (main paper Table 2; worked example in appendix Table 4.1).
- The recalibration method (appendix pp. 40-41): per country/region, sex and
  endpoint, an intercept and slope from regressing GBD-2017 expected risk on
  model-predicted risk across 5-year age groups; MI and stroke combined as
  1 - (1 - p_MI)(1 - p_stroke).

What it does not give: the recalibration factors themselves (21 regions x 2
sexes x 2 endpoints x 2 models = 168 intercept/slope pairs), and the GBD
incidence and NCD-RisC risk-factor inputs only appear as figures (1.7-1.16).
So exact risks cannot be reproduced from the publication; only the published
charts (risk bands) can be checked, which proves category agreement, not
numbers.

Note: v1's WHO "validation" used `whoishRisk`, which is the 2007 WHO/ISH model,
not this one.

