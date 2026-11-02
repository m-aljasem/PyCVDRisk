# CHADS2, CHA2DS2-VASc, HAS-BLED: verification report

Generated 2026-11-01 by `reference/af/report.py`.

Evidence level C: each paper's own definition, checked rule by rule. No independent
implementation is used: the scores are sums of 0/1 items, so the risk is a wrong rule, not
arithmetic. The quoted definitions are matched against the PDF text by the script.

**chads2** (gage2001.pdf): "adding 1 point each for any of the following—recent CHF, hyper- tension, age 75 years or older, and DM—and 2 points for a history of stroke or TIA" (found in PDF)

| Inputs (others 0, age 50, male) | Score | Expected |
|---|---:|---:|
| {'heart_failure': 1} | 1 | 1 |
| {'hypertension': 1} | 1 | 1 |
| {'age': 74.9} | 0 | 0 |
| {'age': 75} | 1 | 1 |
| {'diabetes': 1} | 1 | 1 |
| {'stroke_tia': 1} | 2 | 2 |
| {'heart_failure': 1, 'hypertension': 1, 'age': 80, 'diabetes': 1, 'stroke_tia': 1} | 6 | 6 |

**chads2** (gage2001.pdf): "an 82-year-old (+1) patient who had hypertension (+1) and a prior stroke (+2) would have a CHADS2 score of 4" (found in PDF)

| Inputs (others 0, age 50, male) | Score | Expected |
|---|---:|---:|
| {'age': 82, 'hypertension': 1, 'stroke_tia': 1} | 4 | 4 |

**cha2ds2_vasc** (lip2010.pdf): "C ongestive heart failure/LV dysfunction 1 H ypertension 1 A ge >= 75 y 2 D iabetes mellitus 1 S troke/TIA/TE 2" (found in PDF)

| Inputs (others 0, age 50, male) | Score | Expected |
|---|---:|---:|
| {'heart_failure': 1} | 1 | 1 |
| {'hypertension': 1} | 1 | 1 |
| {'age': 75} | 2 | 2 |
| {'diabetes': 1} | 1 | 1 |
| {'stroke_tia_te': 1} | 2 | 2 |
| {'vascular_disease': 1} | 1 | 1 |
| {'age': 64.9} | 0 | 0 |
| {'age': 65} | 1 | 1 |
| {'age': 74.9} | 1 | 1 |
| {'sex': 'female'} | 1 | 1 |
| {'age': 80, 'sex': 'female', 'heart_failure': 1, 'hypertension': 1, 'diabetes': 1, 'stroke_tia_te': 1, 'vascular_disease': 1} | 9 | 9 |

**cha2ds2_vasc** (lip2010.pdf): "A ge 65-74 y 1 S ex c ategory (ie female gender) 1" (found in PDF)


**has_bled** (pisters2010.pdf): "A Abnormal renal and liver function (1 point each) 1 or 2 S Stroke 1 B Bleeding 1 L Labile INRs 1 E Elderly 1 D Drugs or alcohol (1 point each) 1 or 2" (found in PDF)

| Inputs (others 0, age 50, male) | Score | Expected |
|---|---:|---:|
| {'hypertension': 1} | 1 | 1 |
| {'abnormal_renal': 1} | 1 | 1 |
| {'abnormal_liver': 1} | 1 | 1 |
| {'abnormal_renal': 1, 'abnormal_liver': 1} | 2 | 2 |
| {'stroke': 1} | 1 | 1 |
| {'bleeding': 1} | 1 | 1 |
| {'labile_inr': 1} | 1 | 1 |
| {'age': 65} | 0 | 0 |
| {'age': 65.1} | 1 | 1 |
| {'drugs': 1} | 1 | 1 |
| {'alcohol': 1} | 1 | 1 |
| {'drugs': 1, 'alcohol': 1} | 2 | 2 |
| {'hypertension': 1, 'abnormal_renal': 1, 'abnormal_liver': 1, 'stroke': 1, 'bleeding': 1, 'labile_inr': 1, 'drugs': 1, 'alcohol': 1, 'age': 70} | 9 | 9 |

**has_bled** (pisters2010.pdf): "elderly ( . 65 years)" (found in PDF)


32/32 cases give the published score; 6/6 quotes found in the PDFs.

Notes: HAS-BLED's "elderly" is printed as ">65 years" (age 65 scores 0). CHADS2's "recent" CHF
and the HAS-BLED item definitions (renal: dialysis, transplant or creatinine >= 200 umol/L; alcohol
>= 8 units/week) are the caller's judgement; the functions take flags.
