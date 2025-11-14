# Task: read PREVENT results from the AHA online calculator for 34 patients

Open the American Heart Association PREVENT online calculator:
https://professional.heart.org/en/guidelines-and-statements/prevent-calculator
For each patient below:

1. Before the first patient, output ONE JSON line listing every input field the calculator shows, with its
   unit, allowed range and whether it is required, and every result it displays: {"fields": [...], "results": [...]}.
2. Enter exactly the patient's values. Units: age in years; sex; SBP mmHg; bp_treated 1 = on blood-pressure
   medication; total_chol and hdl_chol in mg/dL; statin 1 = on a statin; diabetes 1 = has diabetes;
   smoker 1 = current smoker; eGFR in mL/min/1.73 m2 (if the calculator asks for creatinine instead, stop
   and tell me); bmi in kg/m2 (if it asks for height and weight instead, stop and tell me).
   uacr (urine albumin-to-creatinine ratio, mg/g) and hba1c (%) are optional: enter them only when they are
   not null, and leave them empty when null. Leave the ZIP code (social deprivation) empty for every patient.
3. Do NOT change a value to get it accepted. If the calculator rejects one, record the patient with empty
   results and quote the error in "notes". If a required field is not given here, stop and ask me; never fill
   a placeholder.
4. Copy every result exactly as displayed, with its shown decimals: 10-year and 30-year risk of total CVD,
   ASCVD and heart failure, and anything else shown (e.g. PREVENT-Age). If 30-year risk is not shown for a
   patient (e.g. above an age limit), use null and say so in "notes".
5. Output ONE line of JSON per patient, nothing else:
   {"id": 0, "cvd10": "5.1%", "ascvd10": "3.2%", "hf10": "2.0%", "cvd30": "20.4%", "ascvd30": "12.1%", "hf30": "11.0%", "other": {}, "notes": ""}

Patients (one JSON object per line):

{"id": 0, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 1, "age": 30, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 2, "age": 79, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 3, "age": 55, "sex": "male", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 4, "age": 55, "sex": "female", "sbp": 100, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 5, "age": 55, "sex": "female", "sbp": 180, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 6, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 1, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 7, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 280, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 8, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 80, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 9, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 1, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 10, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 1, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 11, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 1, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 12, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 45, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 13, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 120, "bmi": 28.0, "uacr": null, "hba1c": null}
{"id": 14, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 22.0, "uacr": null, "hba1c": null}
{"id": 15, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 38.0, "uacr": null, "hba1c": null}
{"id": 16, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": 30.0, "hba1c": null}
{"id": 17, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": 300.0, "hba1c": null}
{"id": 18, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": 5.5}
{"id": 19, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": 8.0}
{"id": 20, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 1, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": null, "hba1c": 8.0}
{"id": 21, "age": 55, "sex": "female", "sbp": 135, "bp_treated": 0, "total_chol": 200, "hdl_chol": 50, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 85, "bmi": 28.0, "uacr": 30.0, "hba1c": 6.5}
{"id": 22, "age": 34, "sex": "male", "sbp": 116, "bp_treated": 1, "total_chol": 295, "hdl_chol": 81, "statin": 0, "diabetes": 0, "smoker": 1, "egfr": 126, "bmi": 22.7, "uacr": null, "hba1c": null}
{"id": 23, "age": 33, "sex": "female", "sbp": 121, "bp_treated": 0, "total_chol": 169, "hdl_chol": 81, "statin": 1, "diabetes": 0, "smoker": 0, "egfr": 98, "bmi": 19.1, "uacr": 16.3, "hba1c": 9.9}
{"id": 24, "age": 55, "sex": "male", "sbp": 155, "bp_treated": 1, "total_chol": 162, "hdl_chol": 39, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 110, "bmi": 23.4, "uacr": 5.2, "hba1c": null}
{"id": 25, "age": 73, "sex": "female", "sbp": 139, "bp_treated": 1, "total_chol": 185, "hdl_chol": 35, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 37, "bmi": 37.2, "uacr": null, "hba1c": 6.4}
{"id": 26, "age": 48, "sex": "male", "sbp": 144, "bp_treated": 0, "total_chol": 223, "hdl_chol": 93, "statin": 0, "diabetes": 0, "smoker": 1, "egfr": 65, "bmi": 22.8, "uacr": 228.8, "hba1c": null}
{"id": 27, "age": 42, "sex": "female", "sbp": 187, "bp_treated": 0, "total_chol": 203, "hdl_chol": 31, "statin": 0, "diabetes": 0, "smoker": 1, "egfr": 68, "bmi": 19.5, "uacr": 65.2, "hba1c": 6.7}
{"id": 28, "age": 79, "sex": "male", "sbp": 122, "bp_treated": 1, "total_chol": 237, "hdl_chol": 65, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 97, "bmi": 29.4, "uacr": null, "hba1c": null}
{"id": 29, "age": 40, "sex": "female", "sbp": 151, "bp_treated": 0, "total_chol": 211, "hdl_chol": 95, "statin": 1, "diabetes": 0, "smoker": 0, "egfr": 134, "bmi": 29.6, "uacr": 36.3, "hba1c": 5.2}
{"id": 30, "age": 76, "sex": "male", "sbp": 100, "bp_treated": 1, "total_chol": 148, "hdl_chol": 47, "statin": 0, "diabetes": 1, "smoker": 0, "egfr": 80, "bmi": 33.7, "uacr": 15.5, "hba1c": null}
{"id": 31, "age": 30, "sex": "female", "sbp": 163, "bp_treated": 0, "total_chol": 165, "hdl_chol": 64, "statin": 0, "diabetes": 0, "smoker": 0, "egfr": 72, "bmi": 24.3, "uacr": null, "hba1c": 7.2}
{"id": 32, "age": 62, "sex": "male", "sbp": 165, "bp_treated": 1, "total_chol": 209, "hdl_chol": 44, "statin": 1, "diabetes": 1, "smoker": 1, "egfr": 124, "bmi": 23.0, "uacr": 90.7, "hba1c": null}
{"id": 33, "age": 57, "sex": "female", "sbp": 96, "bp_treated": 0, "total_chol": 305, "hdl_chol": 95, "statin": 0, "diabetes": 1, "smoker": 1, "egfr": 39, "bmi": 26.7, "uacr": 3.5, "hba1c": 6.4}

When done, save all output lines to reference/prevent/fixtures/aha_results.jsonl.
