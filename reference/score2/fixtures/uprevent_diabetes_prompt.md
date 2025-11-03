# Task: read SCORE2-Diabetes results from U-Prevent for 32 patients

Open the SCORE2-Diabetes calculator on u-prevent.com (10-year cardiovascular risk for people with type 2
diabetes). For each patient below:

1. Before the first patient, output ONE JSON line listing every input field the calculator shows, with its
   unit, allowed range and whether it is required: {"fields": [...]}.
2. Enter exactly the patient's values. Every patient has type 2 diabetes. Units: age in years; SBP mmHg;
   total_chol, hdl_chol and ldl_chol in mmol/L (ldl_chol is only there in case the calculator requires it);
   HbA1c in mmol/mol; eGFR in mL/min/1.73 m2; age_at_diabetes_diagnosis in years. smoker 1 = current smoker.
   region: low / moderate / high / very_high = the calculator's Low / Moderate / High / Very high risk region.
   Leave everything else off or at its default (no medication, no treatment goals).
3. Do NOT change a value to get it accepted. If the calculator rejects one, record "risk10": null and quote
   the error in "notes". If a required field is not given here, stop and ask me; never fill a placeholder.
4. Copy every result exactly as displayed, with its shown decimals (10-year risk and anything else shown).
5. Output ONE line of JSON per patient, nothing else:
   {"id": 0, "risk10": "12.3%", "other": {"name": "value"}, "notes": ""}

Patients (one JSON object per line):

{"id": 0, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 1, "ldl_chol": 3.8, "age": 40, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 2, "ldl_chol": 3.8, "age": 69, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 3, "ldl_chol": 3.8, "age": 55, "sex": "female", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 4, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 1, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 5, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 110, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 6, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 180, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 7, "ldl_chol": 5.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 7.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 8, "ldl_chol": 3.2, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.8, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 9, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 35, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 10, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 55, "hba1c": 55, "egfr": 85, "region": "moderate"}
{"id": 11, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 40, "egfr": 85, "region": "moderate"}
{"id": 12, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 90, "egfr": 85, "region": "moderate"}
{"id": 13, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 45, "region": "moderate"}
{"id": 14, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 110, "region": "moderate"}
{"id": 15, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "low"}
{"id": 16, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "high"}
{"id": 17, "ldl_chol": 3.8, "age": 55, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 50, "hba1c": 55, "egfr": 85, "region": "very_high"}
{"id": 18, "ldl_chol": 2.8, "age": 54, "sex": "male", "smoker": 1, "sbp": 108, "total_chol": 4.4, "hdl_chol": 1.1, "age_at_diabetes_diagnosis": 50, "hba1c": 79, "egfr": 44, "region": "low"}
{"id": 19, "ldl_chol": 2.0, "age": 56, "sex": "female", "smoker": 0, "sbp": 138, "total_chol": 4.2, "hdl_chol": 1.7, "age_at_diabetes_diagnosis": 42, "hba1c": 51, "egfr": 78, "region": "moderate"}
{"id": 20, "ldl_chol": 2.4, "age": 49, "sex": "male", "smoker": 0, "sbp": 177, "total_chol": 3.9, "hdl_chol": 1.0, "age_at_diabetes_diagnosis": 47, "hba1c": 88, "egfr": 68, "region": "high"}
{"id": 21, "ldl_chol": 3.8, "age": 56, "sex": "female", "smoker": 0, "sbp": 169, "total_chol": 5.6, "hdl_chol": 1.3, "age_at_diabetes_diagnosis": 41, "hba1c": 79, "egfr": 72, "region": "very_high"}
{"id": 22, "ldl_chol": 4.3, "age": 45, "sex": "male", "smoker": 1, "sbp": 186, "total_chol": 6.2, "hdl_chol": 1.4, "age_at_diabetes_diagnosis": 45, "hba1c": 83, "egfr": 85, "region": "low"}
{"id": 23, "ldl_chol": 4.2, "age": 64, "sex": "female", "smoker": 0, "sbp": 181, "total_chol": 5.7, "hdl_chol": 1.0, "age_at_diabetes_diagnosis": 51, "hba1c": 84, "egfr": 87, "region": "moderate"}
{"id": 24, "ldl_chol": 4.1, "age": 46, "sex": "male", "smoker": 0, "sbp": 178, "total_chol": 5.9, "hdl_chol": 1.3, "age_at_diabetes_diagnosis": 44, "hba1c": 92, "egfr": 86, "region": "high"}
{"id": 25, "ldl_chol": 4.1, "age": 51, "sex": "female", "smoker": 0, "sbp": 120, "total_chol": 5.7, "hdl_chol": 1.1, "age_at_diabetes_diagnosis": 47, "hba1c": 66, "egfr": 66, "region": "very_high"}
{"id": 26, "ldl_chol": 4.1, "age": 51, "sex": "male", "smoker": 1, "sbp": 122, "total_chol": 5.6, "hdl_chol": 1.0, "age_at_diabetes_diagnosis": 45, "hba1c": 87, "egfr": 83, "region": "low"}
{"id": 27, "ldl_chol": 2.8, "age": 52, "sex": "female", "smoker": 0, "sbp": 131, "total_chol": 4.2, "hdl_chol": 0.9, "age_at_diabetes_diagnosis": 44, "hba1c": 94, "egfr": 103, "region": "moderate"}
{"id": 28, "ldl_chol": 2.7, "age": 66, "sex": "male", "smoker": 0, "sbp": 102, "total_chol": 5.2, "hdl_chol": 2.0, "age_at_diabetes_diagnosis": 65, "hba1c": 80, "egfr": 76, "region": "high"}
{"id": 29, "ldl_chol": 2.1, "age": 40, "sex": "female", "smoker": 0, "sbp": 110, "total_chol": 3.8, "hdl_chol": 1.2, "age_at_diabetes_diagnosis": 25, "hba1c": 57, "egfr": 70, "region": "very_high"}
{"id": 30, "ldl_chol": 2.2, "age": 56, "sex": "male", "smoker": 0, "sbp": 141, "total_chol": 4.0, "hdl_chol": 1.3, "age_at_diabetes_diagnosis": 45, "hba1c": 79, "egfr": 78, "region": "low"}
{"id": 31, "ldl_chol": 4.3, "age": 48, "sex": "female", "smoker": 0, "sbp": 107, "total_chol": 6.3, "hdl_chol": 1.5, "age_at_diabetes_diagnosis": 45, "hba1c": 69, "egfr": 67, "region": "moderate"}

When done, save all output lines to reference/score2/fixtures/uprevent_diabetes_results.jsonl.
