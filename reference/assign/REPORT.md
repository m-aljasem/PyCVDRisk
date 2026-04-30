# ASSIGN: verification report

Models: ASSIGN 1.0 (Woodward, Brindle, Tunstall-Pedoe; Heart 2007;93:172) and ASSIGN 2.0 (Welsh, Kimenai,
Woodward; Heart 2025;111:557), a sex-specific Cox model with eight inputs (age, total and HDL cholesterol, SBP,
diabetes, family history, cigarettes per day, SIMD score / 10). `cvd_risk.assign(..., version=1|2)`.

Evidence for version 2: level A, the official ASSIGN v2.0 calculator (Right Decision Service, NHS Scotland,
rightdecisions.scot.nhs.uk/assign-v20), plus level C, the printed equations in `source/`. Evidence for version 1:
level A, the developers' ASSIGN offline calculator 1.5.1 (March 2014) from assign-score.com, recovered from the
Wayback Machine (1,000/1,000 seeded patients, max difference 5.7e-14 percentage points), plus level C, the 2007
appendix (`woodward2007_appendix.pdf`). The live calculators run only version 2.

The calculator computes in the browser. Its `executeAlgorithm` method was copied verbatim from the page's
`app.js` (`source/rightdecisions_executeAlgorithm.js`). `run_calculator.py` typed 200 seeded patients
into the real form, with SIMD entered directly rather than by postcode, and saved each displayed score to
`fixtures/calculator_reference.jsonl`. The page shows 1 decimal below 10 and a whole number from 10.

## Version 1: the developers' calculators on the Wayback Machine

CDX search of `assign-score.com/*` found these calculator artefacts, all fetched with `id_` URLs into
`source/wayback/`:

| File | Capture | What it is | sha256 |
|---|---|---|---|
| `assignscore-offline-version-1-5-1.zip` | 20160404100837, `http://www.assign-score.com/downloads/assignscore-offline-version-1-5-1.zip` | Offline calculator 1.5.1, files dated March 2014: a static HTML page whose JavaScript computes the score | 9fd35e7eaa6a319256f53a86a6e9935640ae659b5b4cc1673cb531ad47ac3ae6 |
| `offline_1_5_1/` | from the zip | The 7 files used (`includes/js/assign-offline.js`, `betas.js`, `means.js`, `ranges.js`, `usual-ranges.js`, `main/index.html`, `assign-score.html`); 937 postcode-to-SIMD 2012 tables and images not kept | in the zip |
| `HeartAppxUpdate2013.pdf` | 20140123220234, `/downloads/HeartAppxUpdate2013.pdf` | The 2007 Appendix 1 "modified to include rheumatoid arthritis Oct 2013 onwards" | 13df520708361fefcf86f18b4786caf3d863b20d2e352963fd6d4aa40a137c35 |
| `calculator_20131220.js`, `calculator_20160404.js` | `/includes/js/calculator.js` | Online calculator UI script; the score came from a server-side PHP endpoint (`/includes/php/preview-calculator.ajax.php`, POST) that the archive does not hold. The two captures differ only in two UI lines | 9cf4700b58f22a970cf0baf5cfd887cd51bf2fff2ab7d1b84686124eb6aeae88, cae1bc09d19b015cbc622625afc35adc5b1f099353f711cc610a87388687a0b3 |
| `calculator-hashtables_20131220.js` | 20131220143821, `/includes/js/calculator-hashtables.js` | Online calculator's betas (version 1, women's family history 0.49159), range checks and fill-in means | 6ec51bd99d9ffc031c3c30db7b363d052d0095b6eba83bdc70248f955d742003 |
| `assign-score_executeAlgorithm_20251204.js` | 20251204101422, `https://www.assign-score.com/calculators/assign/dist/js/app.js` | The 2025 redesigned site's calculator: `executeAlgorithm` cut from the 4.4 MB bundle (bundle sha256 61de2726870fc7ac2ac12ac34e70d64fcec305e176f900453df86148aeaad440, not kept). Version 2.0, character-for-character the Right Decisions method | c1e6b0b81bd440ba4533cfc44f3c28f3b701bd7794c0dda0f915103e49c84613 |

The earlier pages (`calculate.asp` 2007, `/estimate-the-risk/` 2008-2010) are server-side forms with no formula in
the archived HTML; no other offline version, spreadsheet or executable is in the index.

What version 1.5 is. The offline page says "Version 1.5.1 March 2014 incorporates SIMD 2012, rheumatoid arthritis and
updated population means, algorithm unchanged". Reading the code confirms this:

- `betas.js` holds the 16 version-1 betas and the 16 version-1 centring means (named `beta_coefficients`), and
  `calculate()` uses the 2007 survival values 0.8831 / 0.9365. All equal the package's version 1.
- Rheumatoid arthritis (from October 2013): `if (arthritis > 0) cpd = cpd + 10`, the rule in
  `HeartAppxUpdate2013.pdf`. This is an input recoding, not a new coefficient.
- "Updated population means" are `means.js`, the age- and sex-band values filled in when the user clicks "use mean"
  for a missing cholesterol, HDL, SBP or cigarettes value. They do not enter the equation.
- SIMD 2012 is the postcode lookup table; the score uses whatever SIMD value it returns.
- The displayed score is `Math.round` of the risk (a whole number), with a 20% high-risk message.

So version 1.5 is version 1.0 run on SIMD 2012 with one extra input rule. It is evidence for version 1.0, not a
separate model. The package has no rheumatoid arthritis argument; a user adds 10 to `cigarettes_per_day`.

The run. `run_offline_v151.js` cuts out, verbatim, `betas.js`, the `accMul`/`accAdd`/`accDiv` decimal helpers and
the body of `calculate()` from the arthritis rule to the rounded score, and runs them in Node with the form values as
strings, as jQuery passes them. SIMD is set directly instead of by postcode. 1,000 seeded patients (mulberry32, seed
151; 500 women, 200 with rheumatoid arthritis, ages 30-74, SIMD 0.5-89 with 2 decimals) are saved to
`fixtures/offline_v151_reference.jsonl`. Results: **1,000/1,000 unrounded scores within 5.7e-14** of
`assign(..., version=1)` (cigarettes + 10 for arthritis) and 1,000/1,000 displayed integers equal; scores 0.6-99.9.
Ignoring the arthritis rule would change the displayed score for 190 of the 200 arthritis patients.

## Checks (`tests/assign/test_assign.py`)

1. Every beta, mean and survival value is found in the PDF text: version 1 in the 2007 appendix, version 2 means
   and survival in the 2025 supplement.
2. The calculator's 32 constants (8 betas, 8 means, survival, per sex) equal the package's version-2 constants.
3. Official calculator run: **200/200 displayed scores reproduced exactly** (package risk rounded as the page
   rounds). 100 women, 75 smokers, 42 with diabetes; scores 0.4-76, 116 shown with a decimal (below 10).
4. A hand calculation (man, version 1) and the version-2 mean-person case agree to 1e-12.
5. Input handling: age outside 30-74, unknown sex, missing SIMD and non-binary diabetes give NaN.
6. Offline calculator 1.5.1: its 32 betas and centring means equal the package's version-1 constants, its survival
   values are 0.8831 / 0.9365, and it adds 10 cigarettes for rheumatoid arthritis.
7. Offline calculator run: 1,000/1,000 scores within 1e-12 (observed 5.7e-14) and 1,000/1,000 displayed integers.

## Printed discrepancies

| Item | 2007 appendix | 2025 supplement / Right Decisions text | Official calculator | Used |
|---|---|---|---|---|
| Men's total-cholesterol beta | 0.22286 (in `asm` and `Lbar`) | 0.22286 in `asm`, **0.2286** in `lbarm` and `lbarm2` | 0.22286 in both | 0.22286 |
| Women's family-history beta | 0.49159 | 0.492 | 0.492 | 0.49159 (v1), 0.492 (v2) |

| Cigarettes term in L | once | `HeartAppxUpdate2013.pdf` prints `bcpd*cpd+bcpd*cpd` | offline 1.5.1: once | once |

The 2013 appendix update's doubled cigarettes term is a typo: the 2007 appendix and the developers' offline code
have it once.

The calculator settles the first item: 0.2286 is a typo in the printed equations only. Read literally, it would lower
men's hazard by 3.5% (version 1) and 3.1% (version 2).

## Calculator behaviour the package does not copy

- It accepts ages 25-90 and warns outside 30-74; the package returns NaN outside 30-74.
- It fills blank SBP, cholesterol, HDL and cigarettes with age- and sex-specific averages, and a blank SIMD or
  postcode with a median (16.23); the package returns NaN for missing inputs.
- It accepts SIMD 1-89, SBP 80-250, total cholesterol 2-12.5, HDL 0.3-3.5 (below total cholesterol) and 0-100
  cigarettes; the package does not apply these limits.

Offline calculator 1.5.1 (version 1), not copied: it accepts ages 25-90 and warns outside 30-74; a missing or
unmatched postcode sets SIMD to 20, while its own message says "Default median value 15.89 used" (15.89 is only the
initial value of `simdScore`, replaced on every Calculate), an inconsistency in the tool; it fills missing values with the `means.js`
band means.

## Limits

- Age range 30-74 (the 2007 cohort and the calculator's "usual" range); the 2025 derivation used ages 40-69.
- SIMD is the score on its 0-100 scale. Version 1 was derived on SIMD 2004 and version 2 on SIMD 2020; a score from
  another release is not equivalent.
- The 2025 paper calls the outcome ASCVD (hospitalisation, revascularisation or cardiovascular death); the 2007
  outcome was CHD, stroke and TIA, fatal or not.
- The Right Decisions page lists groups to treat as high risk without scoring (CKD, familial hypercholesterolaemia,
  diabetes over 40); the function does not apply that rule, and neither does it apply the 10% (version 2) or 20%
  (version 1) threshold.
