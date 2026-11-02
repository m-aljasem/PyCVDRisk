# WHO/ISH risk prediction charts (WHO 2007): verification report

Generated 2026-11-01 by `reference/who_ish/report.py`.

## 1. Source

WHO, *Prevention of cardiovascular disease: guidelines for assessment and management of
cardiovascular risk* (2007), saved as `source/who2007_guideline.pdf`. There is no equation:
Annex 5 says the charts tabulate mean 10-year risks from a simulated cohort of 1,000,000 people
per age and sex group and subregion, built from Comparative Risk Assessment data. The
guideline prints two sample charts: Annex 3 (SEAR-D, with cholesterol) and Annex 4 (EMR-B,
without cholesterol). The charts for the other 12 subregions were published separately
(WHO/ISH pocket guidelines). We could not download them because the WHO IRIS server timed out.
Their cell values come from whoishRisk 1.0 (Collins et al., R, GPL-3), whose table
(`source/whoishRisk/WHO_ISH_Scores.csv`) covers all 14 subregions. `make_table.py` builds the
package table from it and replaces every printed cell with the value read from the guideline.

Cell selection: the guideline's age rule ("if age is 50-59 select 50, 60-69 select 60 etc",
p. 17), applied from 40 to 79. The guideline gives no rule for SBP or cholesterol; we use
whoishRisk's (SBP <140 -> 120, 140-159 -> 140, ...; cholesterol <4.5 -> 4, 4.5-5.4 -> 5, ...).

## 2. The guideline's charts (evidence level A)

Each cell of Annexes 3 and 4 was read from a ghostscript render (200 dpi) by matching the colour
at the cell centre to the five key colours. The worst distance to the nearest key colour is 8.3
(on the 0-441 RGB scale); the key colours are at least 90 apart.

| Comparison | Cells | Agree |
|---|---:|---:|
| `who_ish` vs guideline (SEAR-D 640 + EMR-B 128) | 768 | 768 |
| whoishRisk vs guideline | 768 | 765 |

whoishRisk differs from the printed chart in these cells (all in SEAR-D):

| Cell | whoishRisk | Guideline (used) |
|---|---|---|
| SEAR_D: no diabetes, women, non-smoker, age 70, SBP 120, chol 5 | <10% | 10% to <20% |
| SEAR_D: no diabetes, women, non-smoker, age 70, SBP 120, chol 6 | <10% | 10% to <20% |
| SEAR_D: no diabetes, women, smoker, age 70, SBP 120, chol 8 | 30% to <40% | >=40% |

## 3. whoishRisk (evidence level B)

Every one of the 10752 cells (14 subregions x 768) was looked up through `who_ish`.
10749 agree with whoishRisk's table. The 3 that differ are the
guideline corrections in section 2.

The lookup logic was then checked against a line-by-line transliteration of whoishRisk's
R function (`WHO_ISH_Risk`, its `ifelse` binning chains and table match). The R code itself was not run.
The test set has 20000 random patients aged 40-79 (seed 2007); 30% have SBP and cholesterol on or just
below a cut point, and 25% use the no-cholesterol chart: 19989/20000 agree,
11 differ only through the corrected cells, and 0 differ otherwise.

## 4. Internal consistency of the table

A higher risk factor should not give a lower band. Across the 14 subregions, these cells break
that rule. None is in a printed chart, so they cannot be checked against the source:

| Factor | Subregion | Cell | Band | Band one step up |
|---|---|---|---|---|
| diabetes | AFR_D | no diabetes, men, smoker, age 40, SBP 140, chol 8 | 30% to <40% | 20% to <30% |
| smoker | AFR_D | diabetes, men, non-smoker, age 40, SBP 140, chol 8 | 30% to <40% | 20% to <30% |
| smoker | AFR_E | diabetes, women, non-smoker, age 60, SBP 120, chol 8 | 20% to <30% | 10% to <20% |
| smoker | AMR_A | diabetes, women, non-smoker, age 60, SBP 120, chol 5 | 10% to <20% | <10% |
| smoker | AMR_A | diabetes, women, non-smoker, age 60, SBP 120, chol 7 | 20% to <30% | 10% to <20% |
| smoker | EUR_C | diabetes, women, non-smoker, age 60, SBP 120, chol 6 | 10% to <20% | <10% |
| smoker | EUR_C | diabetes, women, non-smoker, age 60, SBP 140, chol 4 | 10% to <20% | <10% |
| smoker | EUR_C | diabetes, women, non-smoker, age 60, SBP 140, chol 6 | 20% to <30% | 10% to <20% |
| smoker | EUR_C | diabetes, women, non-smoker, age 60, SBP 160, chol 4 | 20% to <30% | 10% to <20% |
| smoker | EUR_C | diabetes, women, non-smoker, age 60, SBP 160, chol 6 | 30% to <40% | 20% to <30% |

They are kept as published by whoishRisk. The simulated means need not be monotone, so these
may be in the official charts too. Checking them needs the 12 regional charts.

## 5. Limitations

- 12 of 14 subregions rest on whoishRisk (level B) until the official regional charts are obtained.
- The SBP and cholesterol cut points are whoishRisk's convention; the guideline states none.
