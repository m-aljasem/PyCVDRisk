# CUORE Project individual score (Italy): verification report

Function: `cvd_risk.cuore.cuore`. Checks: `tests/cuore/test_cuore.py`. Evidence level A + C.

## 1. Sources

- **Paper (evidence C).** Giampaoli S, Palmieri L, Donfrancesco C, et al. Cardiovascular risk assessment in Italy:
  the CUORE Project risk score and risk chart. Ital J Public Health 2007;4(2):102-9 (open access;
  `source/giampaoli2007.pdf`, sha256 d32c7faa...ea40c7). Table 1 gives the cohort means and SDs, Table 2 the Cox
  coefficients (three decimals), G (the linear predictor at the means), the 10-year baseline survival and the
  risk equation, 1 - S(t)^exp(sum(beta x) - G), with uncentred inputs. The model was first described in
  Palmieri L et al. Ann Ist Super Sanita 2004;40(4):393-9.
- **Official calculator (evidence A).** The Istituto Superiore di Sanita's online individual score
  (www.cuore.iss.it/valutazione/calc-rischio, served from progettocuore.medisoft.it/CRCOnline). It runs on the
  server; only its displayed result (one decimal, "8,9% nei prossimi 10 anni") can be compared.

## 2. Official calculator

`query_calculator.py` sent 300 seeded patients (both sexes, ages 35-69, every tenth at the edges of the stated
ranges, a third with two different SBP readings) (displayed risks 0.1% to 67.6%); the answers are in
`fixtures/calculator_reference.csv`.

**300/300 displayed values equal** the package's risk rounded to one decimal, with the coefficients, G and S0 of
Table 2 as printed.

The form asks for two SBP readings. The calculator uses their mean rounded to a whole mmHg, half to even (.NET's
default `Math.Round`): 150 and 135 give 142, 115 and 112 give 114. Of the 96 patients with unequal readings,
the plain mean misses 20; the rounded mean misses none. The package takes one SBP; to reproduce the
calculator, pass `np.round((sbp1 + sbp2) / 2)`.

## 3. Checks against the paper

- **Hazard ratios.** Table 2's HR for a 1 SD higher level (Table 1 SDs) or yes vs no equals exp(beta x SD) or
  exp(beta) within the rounding of beta and HR for 13 of the 14 entries (finding 2).
- **G.** sum(beta x Table 1 mean) is 6.62 for men (printed G 6.583) and 6.23 for women (6.016), both within the
  rounding of the three-decimal coefficients (bounds 0.23 and 0.24); a weak check.

## 4. Findings

1. **SBP is rounded in the calculator** (section 2); not stated on the page or in the paper.
2. **Table 2, men's HDL cholesterol: HR 1.60 (adverse) and 0.60 (favourable)**, where beta -0.013 and SD 14.0
   give 1.20 and 0.83. 0.60 is not 1/1.60 either. The calculator agrees with beta -0.013, so the HRs are misprints.
3. **Ranges.** The ISS states the score for ages 35-69, SBP 90-200 mmHg, total cholesterol 130-320 mg/dL and HDL
   20-100 mg/dL; the package gives NaN outside them.
