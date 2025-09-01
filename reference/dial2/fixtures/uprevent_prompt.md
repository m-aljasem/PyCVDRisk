# Task: read DIAL2 results from U-Prevent for 20 patients

On the U-Prevent DIAL2 calculator page, for each patient below:

1. Enter exactly these values. Units: SBP mmHg; total and HDL cholesterol mmol/L; HbA1c mmol/mol;
   eGFR mL/min/1.73 m2; age_at_diagnosis = age when type 2 diabetes was diagnosed; region = the
   ESC risk region (pick a country of that region, e.g. low: Netherlands, moderate: Germany).
   Leave every other field (BMI, albuminuria, insulin, retinopathy, treatment goals) empty or at
   its default; write down any field that is required and not listed.
2. Copy the results exactly as displayed, with the shown decimals: 10-year CVD risk, lifetime CVD
   risk, and CVD-free life expectancy.
3. Output ONE line of JSON per patient, nothing else:
   {"id": 0, "risk10": "12%", "lifetime_risk": "45%", "life_expectancy": "81", "country": "Netherlands", "notes": ""}
   Keep values as the displayed strings. Put anything unusual (a field you had to fill, a warning,
   a value out of range) in "notes".

Patients (one JSON object per line):

{"id": 0, "age": 58, "sex": "male", "smoker": true, "sbp": 116, "total_chol": 4.0, "hdl_chol": 1.3, "hba1c": 53, "egfr": 87, "age_at_diagnosis": 57, "region": "low"}
{"id": 1, "age": 61, "sex": "female", "smoker": false, "sbp": 126, "total_chol": 7.1, "hdl_chol": 1.0, "hba1c": 67, "egfr": 57, "age_at_diagnosis": 53, "region": "low"}
{"id": 2, "age": 53, "sex": "male", "smoker": true, "sbp": 170, "total_chol": 6.0, "hdl_chol": 1.1, "hba1c": 49, "egfr": 97, "age_at_diagnosis": 47, "region": "moderate"}
{"id": 3, "age": 61, "sex": "female", "smoker": false, "sbp": 144, "total_chol": 7.3, "hdl_chol": 1.7, "hba1c": 56, "egfr": 87, "age_at_diagnosis": 55, "region": "moderate"}
{"id": 4, "age": 47, "sex": "male", "smoker": false, "sbp": 177, "total_chol": 4.3, "hdl_chol": 1.9, "hba1c": 42, "egfr": 92, "age_at_diagnosis": 38, "region": "low"}
{"id": 5, "age": 72, "sex": "female", "smoker": true, "sbp": 160, "total_chol": 7.0, "hdl_chol": 1.9, "hba1c": 81, "egfr": 93, "age_at_diagnosis": 62, "region": "low"}
{"id": 6, "age": 48, "sex": "male", "smoker": false, "sbp": 171, "total_chol": 4.8, "hdl_chol": 1.8, "hba1c": 47, "egfr": 101, "age_at_diagnosis": 39, "region": "moderate"}
{"id": 7, "age": 54, "sex": "female", "smoker": true, "sbp": 117, "total_chol": 4.5, "hdl_chol": 1.8, "hba1c": 77, "egfr": 74, "age_at_diagnosis": 49, "region": "moderate"}
{"id": 8, "age": 54, "sex": "male", "smoker": true, "sbp": 127, "total_chol": 4.1, "hdl_chol": 1.8, "hba1c": 60, "egfr": 96, "age_at_diagnosis": 45, "region": "low"}
{"id": 9, "age": 57, "sex": "female", "smoker": true, "sbp": 118, "total_chol": 5.7, "hdl_chol": 1.3, "hba1c": 71, "egfr": 104, "age_at_diagnosis": 44, "region": "low"}
{"id": 10, "age": 75, "sex": "male", "smoker": false, "sbp": 112, "total_chol": 5.7, "hdl_chol": 1.3, "hba1c": 45, "egfr": 89, "age_at_diagnosis": 68, "region": "moderate"}
{"id": 11, "age": 40, "sex": "female", "smoker": false, "sbp": 140, "total_chol": 7.4, "hdl_chol": 1.0, "hba1c": 66, "egfr": 63, "age_at_diagnosis": 34, "region": "moderate"}
{"id": 12, "age": 61, "sex": "male", "smoker": false, "sbp": 142, "total_chol": 5.5, "hdl_chol": 1.1, "hba1c": 73, "egfr": 88, "age_at_diagnosis": 53, "region": "low"}
{"id": 13, "age": 51, "sex": "female", "smoker": false, "sbp": 125, "total_chol": 4.5, "hdl_chol": 1.9, "hba1c": 82, "egfr": 76, "age_at_diagnosis": 46, "region": "low"}
{"id": 14, "age": 62, "sex": "male", "smoker": false, "sbp": 121, "total_chol": 3.9, "hdl_chol": 1.9, "hba1c": 74, "egfr": 98, "age_at_diagnosis": 49, "region": "moderate"}
{"id": 15, "age": 77, "sex": "female", "smoker": false, "sbp": 135, "total_chol": 4.8, "hdl_chol": 1.0, "hba1c": 71, "egfr": 67, "age_at_diagnosis": 64, "region": "moderate"}
{"id": 16, "age": 78, "sex": "male", "smoker": false, "sbp": 171, "total_chol": 4.4, "hdl_chol": 1.6, "hba1c": 85, "egfr": 68, "age_at_diagnosis": 69, "region": "low"}
{"id": 17, "age": 63, "sex": "female", "smoker": false, "sbp": 121, "total_chol": 7.1, "hdl_chol": 1.1, "hba1c": 61, "egfr": 86, "age_at_diagnosis": 58, "region": "low"}
{"id": 18, "age": 54, "sex": "male", "smoker": true, "sbp": 152, "total_chol": 5.6, "hdl_chol": 1.5, "hba1c": 63, "egfr": 97, "age_at_diagnosis": 43, "region": "moderate"}
{"id": 19, "age": 43, "sex": "female", "smoker": false, "sbp": 146, "total_chol": 3.6, "hdl_chol": 0.9, "hba1c": 72, "egfr": 100, "age_at_diagnosis": 40, "region": "moderate"}

When done, paste all output lines into reference/dial2/fixtures/uprevent_results.jsonl.
