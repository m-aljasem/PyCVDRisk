# Task: read SMART2 results from U-Prevent for 30 patients

On the U-Prevent SMART2 calculator (secondary prevention, patients with established atherosclerotic
disease; u-prevent.com), for each patient below:

1. Before the first patient, list every input field the calculator shows, with its unit and whether
   it is required, as ONE JSON line: {"fields": [...]}. Note which risk regions or countries it offers.
2. Enter exactly the patient's values. Units: SBP mmHg; cholesterol mmol/L (enter total and HDL if
   the calculator asks for those; non_hdl_chol = total_chol - hdl_chol is given for reference only);
   eGFR mL/min/1.73 m2 (if it asks for creatinine instead, tell me and stop); hsCRP mg/L (if it asks
   for mg/dL, convert and say so); years_since_diagnosis = years since the first manifestation of
   cardiovascular disease. cad / cerebrovascular / pad / aaa = 1 means that disease is present
   (coronary artery disease / cerebrovascular disease / peripheral artery disease / abdominal aortic
   aneurysm). antithrombotic = 1 means aspirin or an equivalent antithrombotic drug is taken.
   Region: europe_low / europe_moderate / europe_high / europe_very_high = the ESC risk region
   (pick a country of that region, e.g. low: Netherlands, moderate: Germany, high: Poland,
   very_high: Ukraine; use the region itself if the calculator offers regions).
   Leave every field not listed here empty or at its default and write down any that is required.
3. Copy every result exactly as displayed, with its shown decimals (10-year risk, and any other
   number the calculator shows, such as life expectancy, lifetime risk or treatment benefit).
4. Output ONE line of JSON per patient, nothing else:
   {"id": 0, "risk10": "20%", "other": {"name": "value"}, "country": "Germany", "notes": ""}
   Keep values as the displayed strings. Put anything unusual (a field you had to fill, a warning,
   a value out of range, a value you had to convert) in "notes".

Patients (one JSON object per line):

{"id": 0, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 1, "non_hdl_chol": 3.8, "age": 70, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 2, "non_hdl_chol": 3.8, "age": 45, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 3, "non_hdl_chol": 3.8, "age": 60, "sex": "female", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 4, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 1, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 5, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 6, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 160, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 7, "non_hdl_chol": 5.3, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 6.5, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 8, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 50, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 9, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 100, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 10, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 8.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 11, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 0.5, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 12, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 10, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 13, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 0, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 14, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 0, "cerebrovascular": 1, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 15, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 1, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 16, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 1, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 17, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 1, "region": "europe_moderate"}
{"id": 18, "non_hdl_chol": 3.8, "age": 60, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 130, "total_chol": 5.0, "hdl_chol": 1.2, "egfr": 80, "hscrp": 2.0, "years_since_diagnosis": 3, "cad": 1, "cerebrovascular": 0, "pad": 0, "aaa": 0, "antithrombotic": 0, "region": "europe_very_high"}
{"id": 19, "non_hdl_chol": 5.0, "age": 67, "sex": "male", "smoker": 1, "diabetes": 1, "sbp": 117, "total_chol": 6.7, "hdl_chol": 1.7, "egfr": 67, "hscrp": 11.6, "years_since_diagnosis": 0, "cad": 1, "cerebrovascular": 1, "pad": 0, "aaa": 1, "antithrombotic": 1, "region": "europe_low"}
{"id": 20, "non_hdl_chol": 1.7, "age": 43, "sex": "female", "smoker": 0, "diabetes": 1, "sbp": 175, "total_chol": 3.4, "hdl_chol": 1.7, "egfr": 70, "hscrp": 7.9, "years_since_diagnosis": 12, "cad": 1, "cerebrovascular": 1, "pad": 0, "aaa": 0, "antithrombotic": 1, "region": "europe_moderate"}
{"id": 21, "non_hdl_chol": 1.5, "age": 49, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 111, "total_chol": 2.4, "hdl_chol": 0.9, "egfr": 78, "hscrp": 3.7, "years_since_diagnosis": 2, "cad": 1, "cerebrovascular": 1, "pad": 1, "aaa": 1, "antithrombotic": 1, "region": "europe_high"}
{"id": 22, "non_hdl_chol": 3.0, "age": 49, "sex": "female", "smoker": 0, "diabetes": 0, "sbp": 152, "total_chol": 4.9, "hdl_chol": 1.9, "egfr": 82, "hscrp": 8.4, "years_since_diagnosis": 0, "cad": 1, "cerebrovascular": 1, "pad": 0, "aaa": 1, "antithrombotic": 1, "region": "europe_very_high"}
{"id": 23, "non_hdl_chol": 3.1, "age": 64, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 171, "total_chol": 4.0, "hdl_chol": 0.9, "egfr": 82, "hscrp": 0.8, "years_since_diagnosis": 18, "cad": 1, "cerebrovascular": 1, "pad": 1, "aaa": 0, "antithrombotic": 1, "region": "europe_low"}
{"id": 24, "non_hdl_chol": 1.5, "age": 74, "sex": "female", "smoker": 0, "diabetes": 1, "sbp": 115, "total_chol": 2.9, "hdl_chol": 1.4, "egfr": 95, "hscrp": 6.8, "years_since_diagnosis": 5, "cad": 0, "cerebrovascular": 0, "pad": 1, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 25, "non_hdl_chol": 3.4, "age": 76, "sex": "male", "smoker": 0, "diabetes": 1, "sbp": 177, "total_chol": 5.0, "hdl_chol": 1.6, "egfr": 105, "hscrp": 8.0, "years_since_diagnosis": 9, "cad": 1, "cerebrovascular": 0, "pad": 1, "aaa": 1, "antithrombotic": 0, "region": "europe_high"}
{"id": 26, "non_hdl_chol": 3.4, "age": 78, "sex": "female", "smoker": 0, "diabetes": 0, "sbp": 159, "total_chol": 5.2, "hdl_chol": 1.8, "egfr": 71, "hscrp": 9.1, "years_since_diagnosis": 8, "cad": 0, "cerebrovascular": 1, "pad": 1, "aaa": 0, "antithrombotic": 0, "region": "europe_very_high"}
{"id": 27, "non_hdl_chol": 3.6, "age": 71, "sex": "male", "smoker": 0, "diabetes": 0, "sbp": 178, "total_chol": 4.8, "hdl_chol": 1.2, "egfr": 43, "hscrp": 7.2, "years_since_diagnosis": 4, "cad": 1, "cerebrovascular": 0, "pad": 1, "aaa": 0, "antithrombotic": 1, "region": "europe_low"}
{"id": 28, "non_hdl_chol": 3.2, "age": 61, "sex": "female", "smoker": 0, "diabetes": 0, "sbp": 165, "total_chol": 5.2, "hdl_chol": 2.0, "egfr": 87, "hscrp": 1.3, "years_since_diagnosis": 10, "cad": 1, "cerebrovascular": 0, "pad": 1, "aaa": 0, "antithrombotic": 0, "region": "europe_moderate"}
{"id": 29, "non_hdl_chol": 3.4, "age": 44, "sex": "male", "smoker": 1, "diabetes": 0, "sbp": 158, "total_chol": 4.6, "hdl_chol": 1.2, "egfr": 49, "hscrp": 8.4, "years_since_diagnosis": 9, "cad": 0, "cerebrovascular": 1, "pad": 1, "aaa": 1, "antithrombotic": 1, "region": "europe_high"}

When done, paste all output lines into reference/smart2/fixtures/uprevent_results.jsonl.
