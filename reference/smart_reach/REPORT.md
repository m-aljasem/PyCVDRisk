# SMART-REACH — verification report

Generated 2026-11-01 by `reference/smart_reach/report.py`.

## 1. Sources

Kaasenbrood et al., J Am Heart Assoc 2018;7:e009217, saved in `source/`: the paper, its
supplement (Data S1; Tables S2-S3 print the model) and the authors' calculator spreadsheet
(Data S2), which holds the coefficients to 7 significant digits. `extract_calculator.py` dumps
the calculator's model formulas to `fixtures/calculator_formulas.txt`.

`check_coefficients.py`: 24 coefficients, 2 region shifts, 45 baseline rows checked; mismatches: none

- The coefficients are the calculator's. Each supplement value is the calculator's rounded to
  4 decimals, except the CVD three-locations term: printed 0.2896, calculator 0.2895387.

## 2. Agreement with the authors' calculator (evidence level A)

The calculator was recalculated by LibreOffice Calc with its formulas unmodified
(`run_reference.py`): each patient's codes go into the engine sheet's input cells, the cells
the front sheet's drop-down controls normally fill. LibreOffice writes 15 significant digits.

```
reference: kaasenbrood2018_calculator.xls (authors' calculator, Data S2), unmodified formulas
engine: LibreOffice 24.2.7.2 420(Build:2)
generated: 2026-11-01
```

600 seeded synthetic patients: every whole age 45-89, both sexes, all combinations of
risk factors, disease locations and the three regions.

| Output | Rows | Agreement |
|---|---|---|
| 10-year risk (ages 45-80) | 468 | max rel diff 3.1e-15; NaN on the same rows: True |
| Life expectancy, normal life tables | 547 | max rel diff 8.9e-16; '>90' on the same rows: True |
| Life expectancy, all rows, calculator's own median rule | 600 | max rel diff 3.5e-14 |

## 3. A defect in the calculator's life expectancy

The two models were fitted separately, so for high-risk people the yearly CVD risk H and
non-CVD death risk I can sum to more than 1 at older ages. Event-free survival
G(t+1) = G(t)(1 - H - I) then turns negative. The calculator finds the median age by
counting the table rows whose G is not in (0, 0.5) (`COUNTIF(O28:O73, 0)`), so negative
rows late in the table are counted as if survival were still above 0.5 and the median is
misplaced even when it was reached decades earlier.

- 53/600 synthetic patients (8.8%) have such a table; for every one of
  them, and no other, our life expectancy differs from the calculator's. In 19 of them the
  calculator's life expectancy is below the person's current age.
- `smart_reach` returns the first crossing of 0.5 instead. Applying the calculator's rule to
  our tables reproduces its output on all 600 rows (third row of the table above), which
  confirms the diagnosis. 547/600 rows agree directly.
- The 10-year risk follows the calculator's arithmetic; 7 of 468 patients have a
  negative G within their 10-year window (ages [75, 76, 80]).

## 4. Mutation check

Every non-zero coefficient multiplied by 1 + 1e-6: 24/24 change the 10-year risks
beyond the 1e-12 tolerance.

## 5. Scope

- Outputs: 10-year risk of MI, stroke or CVD death (ages 45-80) and life expectancy free of
  these events (ages 45-89). The calculator's treatment-effect module (statins, PCSK9
  inhibitors, blood pressure lowering, antithrombotics, canakinumab) is not implemented.
- Age in whole years, as in the calculator's table. Regions: the calculator's three
  (Western Europe = reference, North America, the Netherlands).
- A row with no disease location (CAD, cerebrovascular disease, PAD) is NaN: the model is
  for people with established atherosclerotic disease (the calculator does compute it).
