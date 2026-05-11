# Hisayama ASCVD risk model (2022): verification report

Evidence level A + B + C. A: the Japan Atherosclerosis Society's official 2022 app (browser JS, run unmodified) gives
the same shown risk, risk category and reference risk as the package on 3,920/3,920 patients; it uses the Fig. 1 points
for age, sex, SBP, HDL-C, LDL-C and smoking and the paper's points-to-risk formula, so those are level A. The diabetes,
proteinuria and exercise points, the full equation and the non-HDL-C model are not in the app and stay at C (+ B:
Jcvrisk on every patient; ecg.watson.jp on 79/80 chart cells).

## Sources

- **Paper**: Honda T, Chen S, Hata J, Yoshida D, Hirakawa Y, Furuta Y, Shibata M, Sakata S, Kitazono T, Ninomiya T.
  Development and validation of a risk prediction model for atherosclerotic cardiovascular disease in Japanese
  adults: the Hisayama Study. J Atheroscler Thromb 2022;29(3):345-61. doi:10.5551/jat.61960 (`source/honda2022.pdf`,
  with supplement; sha256 4211287b...8344). Table 2: coefficients (3 decimals); the equation with S10 = 0.9696
  (competing-risk adjusted, Supplemental Table 1) and constant 6.7963; Fig. 1: point chart and risk table;
  Supplemental Table 3: how the points were derived; Supplemental Table 4: the non-HDL-C model (constant 6.9895).
- **R package Jcvrisk 0.1.3** (CRAN 2025-10-29, Okumiyama H, Fujii R; MIT): `hisayama()` implements the point score
  and its formula. Vendored unmodified in `reference/vendor/Jcvrisk` (matches the package's MD5 list).
- **Web calculator**: ecg.watson.jp/medical/ASCVD ("ASCVD Risk Calculator for Japanese patients (Hisayama score)"; `source/ecg_watson_ascvd.html`, sha256 d4246e41...; printed copy `source/ecg_watson_calculator_page.pdf`).
  It implements the JAS 2022 guideline chart: no proteinuria or exercise items, impaired glucose tolerance 1 point
  in place of diabetes 3, ages 40-79.

- **JAS official app (level A)**: "動脈硬化性疾患発症予測・脂質管理目標設定アプリ（医師・医療従事者向け）", the web
  version of the JAS 2022 risk / lipid-target app (App Store id1229590106, Google Play baraabaru.qzlab.com.kandomyakuapp2),
  linked from https://www.j-athero.org/jp/general/ge_tool2/ and served from
  https://www.j-athero.org/general/gl2022app/workers/ (Copyright JAS 2022). It computes in the browser:
  `mainQuestions.js` (`calsSuitaScore2`, `createResult2`; sha256 93d8c048...0419) sums the points and looks the total
  up in a 30-row table of 10-year risk and category; `result.js` shows the risk text, category, LDL-C / non-HDL-C
  targets and the ratio to the lowest-risk person of the same age and sex. All 15 HTML/JS files are in
  `source/jas2022_app/` (sha256 in `sha256sum source/jas2022_app/*`). The older public tool
  j-athero.org/general/ge_app/general2/ is still live and still says it follows the 2017 guideline (Suita-based).

## Model

`hisayama`: p = 1 - 0.9696^exp(sum(beta x) - 6.7963), age, sex, SBP, diabetes, HDL-C, LDL-C (or non-HDL-C, the
alternative model), proteinuria, current smoking, no regular exercise. `hisayama_points` + `hisayama_points_risk`:
Fig. 1 points (0-27 without age) and p = 1 - 0.9696^exp((score + age points) x 0.144 - 2.4767), age points 0, 5,
11, 16, 20 by decade (80-84).

## Checks (`tests/hisayama/test_hisayama.py`)

- Points = weight / 0.144, rounded, for all 18 rows of Supplemental Table 3.
- 2.4767 = 6.7963 - 4.3196, and 4.3196 is sum(beta x reference value) from that table's weights (within 0.002).
- Both equation constants are within 0.05 of sum(beta x mean) from Table 1 (3-decimal betas).
- Jcvrisk `hisayama()` on 2,702 seeded patients aged 40-84: equal to 1e-12.
- ecg.watson.jp: its 20 x 4 table equals the formula at score + 2 (the page has no exercise item, so every patient
  carries the 2 points for no regular exercise), shown to 1 decimal with a floor of 1.0, in 79/80 cells.

- **JAS official app** (`run_jas2022_app.js` -> `fixtures/jas2022_app_reference.json`; test
  `test_jas2022_official_app`): every point cell (4 age decades x 2 sexes x 3 HDL x 4 LDL x 5 SBP x 2 glucose x 2
  smoking = 1,920, at band-boundary values 119/120, 129/130, 139/140, 159/160, 39/40, 59/60) plus 2,000 seeded
  patients with raw values (1,000 entering TC + TG, the app's Friedewald route, parseInt truncation). The app shows
  points nowhere; it shows the 10-year risk to 1 decimal, the category (low < 2%, intermediate 2-< 10%, high >= 10%)
  and the ratio to the same-age, same-sex reference risk. Package: Fig. 1 points + 1 for impaired glucose tolerance,
  put through `hisayama_points_risk(score + 2, age)`, shown to 1 decimal ("1.0% or less" at total <= 7). Shown risk
  3,920/3,920, category 3,920/3,920 (1,082 low, 1,931 intermediate, 907 high), reference risk 3,920/3,920, LDL-C
  used 3,920/3,920; all 29 reachable table rows hit (totals 0-35). The app's table equals the formula at total + 2,
  rounded to 1 decimal, in 30/30 rows (row 7 is "1.0% or less", formula 0.94%).

## Findings

1. **Fig. 1's risk table does not follow the paper's own formula.** Its values form one sequence in score + age
   points, as the formula implies, but they rise faster (on the log(-log) scale about 0.155 per point, not 0.144):
   15.7% against 11.9% at score 27, ages 40-49; 49.7% against 37.0% at score 16, ages 80-84. The package follows
   the stated formula, as do Jcvrisk and the JAS 2022 chart (via the web calculator); Fig. 1's table is not used.
2. **Web calculator, one cell**: score 19, ages 70-79 shows 41.1%; the formula and the JAS official app give 41.4%
   (third-party transcription error).
3. **The JAS 2022 chart is a different score** (confirmed by the JAS app's code): no proteinuria or exercise item,
   impaired glucose tolerance (fasting glucose 110-125 mg/dL or OGTT 2-h 140-199) 1 point; diabetes, CKD and PAD
   route to the high-risk category before the score; prior CHD or atherothrombotic stroke route to secondary
   prevention; ages 40-79. Its risk table is the paper's formula with the 2 no-exercise points always added
   (classification: guideline adaptation, not an error). Not implemented as a separate function; the test shows it
   is `hisayama_points(..., diabetes=0, proteinuria=0, exercise=1) + igt` through `hisayama_points_risk`.
4. **JAS app, unreachable rows**: the table has rows for totals 36 (46.1%) and >= 37 ("50% or more"), but the
   maximum total is 35 (16 + 7 + 2 + 3 + 4 + 1 + 2). Harmless.
5. **JAS app, input handling**: with TC entered the form clears LDL-C and uses Friedewald LDL-C = TC - HDL-C - TG/5
   (truncated), whatever the TG level; the non-HDL-C it computes is passed on but not scored.
