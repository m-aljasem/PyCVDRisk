# Task: read SMART2 results from U-Prevent for 22 more patients (round 2)

Same U-Prevent SMART2 calculator as round 1. For each patient below:

1. Enter exactly the patient's values, with the same units and meanings as round 1: SBP mmHg;
   total_chol, hdl_chol and ldl_chol in mmol/L (enter all three as given; non_hdl_chol is for
   reference only); eGFR mL/min/1.73 m2; hsCRP mg/L; years_since_diagnosis = years since the
   first cardiovascular event; cad / cerebrovascular / pad / aaa = 1 means that disease is present.
   hba1c is in mmol/mol: enter it when diabetes = 1 (it is null when diabetes = 0).
   Region: europe_low / europe_moderate / europe_high / europe_very_high = the calculator's
   Low / Moderate / High / Very high risk region.
   antithrombotic: 0 = "No antithrombotic", 1 = "Aspirin or equivalent", "dual_antiplatelet" =
   "Dual antiplatelet therapy", "aspirin_plus_low_dose_doac" = "Aspirin + low dose DOAC".
   Leave everything else off or at its default (smoking only as given; Mediterranean diet off; step
   count empty; no statin, ezetimibe, bempedoic acid, PCSK9-inhibitor, GLP1-RA, SGLT2-inhibitor,
   colchicine or icosapent-ethyl).
2. Several patients sit exactly at an input limit on purpose (age 40 and 80, eGFR 30, hsCRP 0.1 and
   15, 30 years since the event, SBP 200, total cholesterol 8.0, HDL 0.6, HbA1c 33 and 108). Do NOT
   change a value to get it accepted. If the calculator rejects a value, record that patient with
   "risk10": null and quote the error message in "notes", then move on.
3. Do not fill in any placeholder. If a field is required that is not given here, stop and ask me.
4. Copy every result exactly as displayed, with its shown decimals.
5. Output ONE line of JSON per patient, nothing else:
   {"id": 100, "risk10": "26.1%", "other": {"reduction_due_to_treatment": "0.0%"}, "notes": ""}
   Keep values as the displayed strings; put anything unusual in "notes".

Patients (one JSON object per line):

{"id": 100, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 101, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 33, "age": 60, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 102, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 42, "age": 60, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 103, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 53, "age": 60, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 104, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 64, "age": 60, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 105, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 75, "age": 60, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 106, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 90, "age": 60, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 107, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 108, "age": 60, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 108, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 42, "age": 72, "sex": "female", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 55, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 1, "aaa": 0, "antithrombotic": 0, "region": "europe_high"}
{"id": 109, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": 86, "age": 72, "sex": "female", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 55, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 1, "aaa": 0, "antithrombotic": 0, "region": "europe_high"}
{"id": 110, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 40, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 111, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 80, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 112, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 30, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 113, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 120, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 114, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 0.1, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 115, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 15.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 116, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 30, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 117, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 95, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 118, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 200, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 119, "non_hdl_chol": 7.4, "ldl_chol": 6.9, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 8.0, "hdl_chol": 0.6, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 120, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": "dual_antiplatelet", "region": "europe_moderate"}
{"id": 121, "non_hdl_chol": 3.8, "ldl_chol": 3.3, "hba1c": null, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": "aspirin_plus_low_dose_doac", "region": "europe_moderate"}

When done, save all output lines to reference/smart2/fixtures/uprevent_results2.jsonl.
