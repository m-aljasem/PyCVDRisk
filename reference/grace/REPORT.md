# GRACE (in-hospital and 6-month post-discharge): verification report

Generated 2026-11-01 by `reference/grace/report.py`.

Evidence level C: the published nomograms. No independent implementation exists for them:
RiskScorescvd's `GRACE()` is a different, continuous (interpolated) version with other weights
(Killip 0/15/29/44), so it is not a reference for either paper.

## 1. In-hospital death (Granger 2003, Figure 4)

| Input | Band (low-high input) | Points at low / high | Printed |  |
|---|---|---|---:|---|
| killip | 1-1 | 0 / 0 | 0 | printed: "I 0 II 20 III 39 IV 59" found |
| killip | 2-2 | 20 / 20 | 20 |
| killip | 3-3 | 39 / 39 | 39 |
| killip | 4-4 | 59 / 59 | 59 |
| sbp | 0-80 | 58 / 58 | 58 | printed: "≤80 58 80-99 53 100-119 43 120-139 34 140-159 24 160-199 10 ≥200 0" found |
| sbp | 80.1-99.9 | 53 / 53 | 53 |
| sbp | 100-119.9 | 43 / 43 | 43 |
| sbp | 120-139.9 | 34 / 34 | 34 |
| sbp | 140-159.9 | 24 / 24 | 24 |
| sbp | 160-199.9 | 10 / 10 | 10 |
| sbp | 200-300 | 0 / 0 | 0 |
| heart_rate | 0-50 | 0 / 0 | 0 | printed: "≤50 0 50-69 3 70-89 9 90-109 15 110-149 24 150-199 38 ≥200 46" found |
| heart_rate | 50.1-69.9 | 3 / 3 | 3 |
| heart_rate | 70-89.9 | 9 / 9 | 9 |
| heart_rate | 90-109.9 | 15 / 15 | 15 |
| heart_rate | 110-149.9 | 24 / 24 | 24 |
| heart_rate | 150-199.9 | 38 / 38 | 38 |
| heart_rate | 200-300 | 46 / 46 | 46 |
| age | 18-30 | 0 / 0 | 0 | printed: "≤30 0 30-39 8 40-49 25 50-59 41 60-69 58 70-79 75 80-89 91 ≥90 100" found |
| age | 30.1-39.9 | 8 / 8 | 8 |
| age | 40-49.9 | 25 / 25 | 25 |
| age | 50-59.9 | 41 / 41 | 41 |
| age | 60-69.9 | 58 / 58 | 58 |
| age | 70-79.9 | 75 / 75 | 75 |
| age | 80-89.9 | 91 / 91 | 91 |
| age | 90-110 | 100 / 100 | 100 |
| creatinine | 0-0.39 | 1 / 1 | 1 | printed: "0-0.39 1 0.40-0.79 4 0.80-1.19 7 1.20-1.59 10 1.60-1.99 13 2.00-3.99 21 >4.0 28" found |
| creatinine | 0.4-0.79 | 4 / 4 | 4 |
| creatinine | 0.8-1.19 | 7 / 7 | 7 |
| creatinine | 1.2-1.59 | 10 / 10 | 10 |
| creatinine | 1.6-1.99 | 13 / 13 | 13 |
| creatinine | 2-3.99 | 21 / 21 | 21 |
| creatinine | 4.01-9 | 28 / 28 | 28 |
| cardiac_arrest | 0-0 | 0 / 0 | 0 | printed: "Cardiac Arrest at Admission 39" found |
| cardiac_arrest | 1-1 | 39 / 39 | 39 |
| st_deviation | 0-0 | 0 / 0 | 0 | printed: "ST-Segment Deviation 28" found |
| st_deviation | 1-1 | 28 / 28 | 28 |
| elevated_enzymes | 0-0 | 0 / 0 | 0 | printed: "Elevated Cardiac Enzyme Levels 14" found |
| elevated_enzymes | 1-1 | 14 / 14 | 14 |

39/39 bands give the printed points at both ends; 8/8 printed rows found in the PDF.

### Worked examples

| Example (Figure 4) | Printed | This implementation |
|---|---:|---:|
| Killip II, SBP 100, HR 100, age 65, creatinine 1, ST deviation, enzymes | 196 | 185 |
| Killip I, SBP 80, HR 60, age 55, creatinine 0.4 | 103 | 106 |

The examples are quoted from the PDF (2/2 found). They do not follow the printed table:
example 1 gives SBP 100 53 points (table: 100-119 = 43) and its terms add to 195, not 196;
example 2 gives creatinine 0.4 1 point (table: 0.40-0.79 = 4). Both examples put a value on a
band's printed lower edge into the band below. With that reading they give 195 and 103.
We follow the table, whose 0.40 is explicit; this is a note for the authors.

## 2. 6-month post-discharge death (Eagle 2004, Figure 2)

| Input | Band (low-high input) | Points at low / high | Printed |  |
|---|---|---|---:|---|
| age | 18-29.9 | 0 / 0 | 0 | printed: "≤29 30–39 40–49 50–59 60–69 70–79 80–89 ≥90 0 0 18 36 55 73 91 100" found |
| age | 30-39.9 | 0 / 0 | 0 |
| age | 40-49.9 | 18 / 18 | 18 |
| age | 50-59.9 | 36 / 36 | 36 |
| age | 60-69.9 | 55 / 55 | 55 |
| age | 70-79.9 | 73 / 73 | 73 |
| age | 80-89.9 | 91 / 91 | 91 |
| age | 90-110 | 100 / 100 | 100 |
| chf_history | 0-0 | 0 / 0 | 0 | printed: "Heart Failure 24" found |
| chf_history | 1-1 | 24 / 24 | 24 |
| mi_history | 0-0 | 0 / 0 | 0 | printed: "Myocardial Infarction 12" found |
| mi_history | 1-1 | 12 / 12 | 12 |
| heart_rate | 0-49.9 | 0 / 0 | 0 | printed: "≤49.9 50–69.9 70–89.9 90–109.9 110–149.9 150–199.9 ≥200 0 3 9 14 23 35 43" found |
| heart_rate | 50-69.9 | 3 / 3 | 3 |
| heart_rate | 70-89.9 | 9 / 9 | 9 |
| heart_rate | 90-109.9 | 14 / 14 | 14 |
| heart_rate | 110-149.9 | 23 / 23 | 23 |
| heart_rate | 150-199.9 | 35 / 35 | 35 |
| heart_rate | 200-300 | 43 / 43 | 43 |
| sbp | 0-79.9 | 24 / 24 | 24 | printed: "≤79.9 80–99.9 100–119.9 120–139.9 140–159.9 160–199.9 ≥200 24 22 18 14 10 4 0" found |
| sbp | 80-99.9 | 22 / 22 | 22 |
| sbp | 100-119.9 | 18 / 18 | 18 |
| sbp | 120-139.9 | 14 / 14 | 14 |
| sbp | 140-159.9 | 10 / 10 | 10 |
| sbp | 160-199.9 | 4 / 4 | 4 |
| sbp | 200-300 | 0 / 0 | 0 |
| st_depression | 0-0 | 0 / 0 | 0 | printed: "11ST-Segment Depression" found |
| st_depression | 1-1 | 11 / 11 | 11 |
| creatinine | 0-0.39 | 1 / 1 | 1 | printed: "0–0.39 0.4–0.79 0.8–1.19 1.2–1.59 1.6–1.99 2–3.99 ≥4 1 3 5 7 9 15 20" found |
| creatinine | 0.4-0.79 | 3 / 3 | 3 |
| creatinine | 0.8-1.19 | 5 / 5 | 5 |
| creatinine | 1.2-1.59 | 7 / 7 | 7 |
| creatinine | 1.6-1.99 | 9 / 9 | 9 |
| creatinine | 2-3.99 | 15 / 15 | 15 |
| creatinine | 4-9 | 20 / 20 | 20 |
| elevated_enzymes | 0-0 | 0 / 0 | 0 | printed: "15Elevated Cardiac Enzymes" found |
| elevated_enzymes | 1-1 | 15 / 15 | 15 |
| in_hospital_pci | 0-0 | 14 / 14 | 14 | printed: "Coronary Invervention 14" found |
| in_hospital_pci | 1-1 | 0 / 0 | 0 |

39/39 bands give the printed points at both ends; 9/9 printed rows found in the PDF.
Figure 2 also prints a stray "1" under SBP >=200; it belongs to no row and is ignored.
The paper gives no worked example.
