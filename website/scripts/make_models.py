"""Write website/data/models.json from the v2 package. Run from repo root:
PYTHONPATH=src venv/bin/python website/scripts/make_models.py

Each model's example is executed here, so the website shows real calls and their real output.
"""

import json
from pathlib import Path

import numpy as np

import cvd_risk

PM = "https://pubmed.ncbi.nlm.nih.gov/?term="
DOI = "https://doi.org/"
# slug, name, full name, year, region, photo key, group, outcome, paper, verification, evidence, example
M = [
    ("score2", "SCORE2", "Systematic COronary Risk Evaluation 2", "2021", "Europe", "score2_map", "primary",
     "10-year fatal and non-fatal CVD, ages 40-69, four ESC risk regions", DOI + "10.1093/eurheartj/ehab309",
     "1,532/1,536 cells of the ESC 2021 charts within printed precision; R RiskScorescvd; the paper's worked example.", "A, B, C",
     'score2(age=[50, 62], sex=["male", "female"], smoker=[1, 0], sbp=[140, 130],\n       total_chol=[5.5, 6.1], hdl_chol=[1.3, 1.6], diabetes=0, region="low")'),
    ("score2-op", "SCORE2-OP", "SCORE2 Older Persons", "2021", "Europe", "elderly", "primary",
     "10-year fatal and non-fatal CVD, ages 70-89, adjusted for competing risk", DOI + "10.1093/eurheartj/ehab312",
     "1,024/1,024 cells of the ESC 2021 charts within printed precision; R RiskScorescvd.", "A, B",
     'score2_op(age=75, sex="female", smoker=0, sbp=150, total_chol=5.8,\n          hdl_chol=1.5, diabetes=0, region="moderate")'),
    ("score2-diabetes", "SCORE2-Diabetes", "SCORE2-Diabetes", "2023", "Europe", "glucose", "conditions",
     "10-year CVD risk in type 2 diabetes, ages 40-69", DOI + "10.1093/eurheartj/ehad260",
     "Official U-Prevent calculator (21/22 displayed values); R RiskScorescvd; 7/8 printed worked examples (1 misprint).", "A, B, C",
     'score2_diabetes(age=60, sex="male", smoker=0, sbp=140, total_chol=5.5, hdl_chol=1.2,\n                diabetes=1, age_at_diabetes_diagnosis=52, hba1c=58, egfr=80, region="moderate")'),
    ("score2-ckd", "SCORE2-CKD", "SCORE2 for chronic kidney disease", "2022", "Europe", "kidney", "conditions",
     "10-year CVD risk with eGFR (and optionally albuminuria) add-ons", DOI + "10.1093/eurjpc/zwac176",
     "Developers' CKD-PC online calculator (71 patients x 4 regions); the supplement's worked example; R RiskScorescvd.", "A, C, B",
     'score2_ckd(age=60, sex="male", smoker=0, sbp=140, total_chol=5.5, hdl_chol=1.3,\n           diabetes=0, egfr=45, region="moderate", acr=30, addon="acr")'),
    ("score2-asia-pacific", "SCORE2 Asia-Pacific", "SCORE2 Asia-Pacific", "2024", "Asia-Pacific", "asia_pacific", "primary",
     "10-year fatal and non-fatal CVD, ages 40-69, four Asia-Pacific risk regions", DOI + "10.1093/eurheartj/ehae609",
     "1,535/1,536 cells of the developers' charts (1 print slip); R RiskScorescvd.", "A, B",
     'score2_asia_pacific(age=55, sex="male", smoker=1, sbp=140, total_chol=5.5,\n                    hdl_chol=1.2, diabetes=0, region="high")'),
    ("score", "SCORE", "Systematic COronary Risk Evaluation (2003)", "2003", "Europe", "score_map", "primary",
     "10-year fatal CVD, total-cholesterol model, high- and low-risk regions", DOI + "10.1016/S0195-668X(03)00114-3",
     "All 800 cells of the official charts (798 within printed precision; 2 chart anomalies documented).", "A",
     'score(age=60, sex="male", total_chol=6.0, sbp=150, smoker=1, region="high")'),
    ("globorisk", "Globorisk", "Globorisk (laboratory, office and fatal)", "2015-2017", "Global, 182 countries", "global", "primary",
     "10-year CVD risk recalibrated to each country and calendar year", DOI + "10.1016/S2213-8587(15)00081-9",
     "Verified against the authors' R package 1.0.2.", "A",
     'globorisk(age=55, sex="male", sbp=140, smoker=1, country="MEX", year=2020,\n          total_chol=5.5, diabetes=0)'),
    ("who2019", "WHO 2019", "WHO CVD risk charts (2019)", "2019", "Global, 21 regions / 190 countries", "who_flag", "primary",
     "10-year fatal and non-fatal CVD, laboratory and non-laboratory versions", DOI + "10.1016/S2214-109X(19)30318-3",
     "Every cell of the 42 official WHO charts and R WHORiskCalculator.", "A, B",
     'who2019(age=55, sex="female", sbp=145, smoker=0, total_chol=6.0, diabetes=1,\n        country="EGY")'),
    ("prevent", "PREVENT", "AHA Predicting Risk of CVD EVENTs", "2024", "United States", "us_flag", "primary",
     "10- and 30-year total CVD, ASCVD, heart failure, CHD and stroke; base, UACR, HbA1c, SDI and full models",
     DOI + "10.1161/CIRCULATIONAHA.123.067626",
     "Official AHA online calculator (189/189 displayed risks, 34 patients); the authors' worked examples; R preventr.", "A, C, B",
     'prevent(age=50, sex="female", sbp=160, bp_treated=1, total_chol=5.2, hdl_chol=1.4,\n        statin=0, diabetes=1, smoker=0, egfr=90, bmi=35, years=10, outcome="total_cvd")'),
    ("pce", "Pooled Cohort Equations", "ACC/AHA Pooled Cohort Equations", "2013", "United States", "generations", "primary",
     "10-year hard ASCVD, ages 40-79, white and African American", DOI + "10.1161/01.cir.0000437741.48606.98",
     "ACC CVD Risk Estimator+ code run on 20,000 patients; the guideline's Table A (46/46); R CVrisk and PooledCohort.", "A, C, B",
     'pce(age=55, sex="female", race="white", total_chol=5.5, hdl_chol=1.3, sbp=120,\n    bp_treated=0, smoker=0, diabetes=0)'),
    ("framingham", "Framingham", "Framingham general CVD (2008)", "2008", "Framingham, Massachusetts", "framingham_sign", "primary",
     "10-year general CVD, laboratory and BMI versions", DOI + "10.1161/CIRCULATIONAHA.107.699579",
     "The Framingham Heart Study's own calculator spreadsheets, the paper's worked examples (32/32) and R CVrisk.", "A, C, B",
     'framingham(age=61, sex="female", sbp=124, bp_treated=0, smoker=1, diabetes=0,\n           total_chol=4.65, hdl_chol=1.22)'),
    ("reynolds", "Reynolds", "Reynolds Risk Score (women 2007, men 2008)", "2007-2008", "United States (Boston)", "blood_sample", "primary",
     "10-year CVD risk with hsCRP and parental history", DOI + "10.1001/jama.297.6.611",
     "Implemented from the printed formulas (19/19 terms), checked against the papers' 17 worked examples.", "C",
     'reynolds(age=60, sex="female", sbp=135, total_chol=5.7, hdl_chol=1.3, hscrp=3.0,\n         smoker=0, parental_history=1)'),
    ("qrisk3", "QRISK3", "QRISK3-2017", "2017", "United Kingdom", "uk_flag", "primary",
     "10-year CVD risk with ethnicity, deprivation and clinical conditions", DOI + "10.1136/bmj.j2099",
     "ClinRisk's own C code and R QRISK3 (LGPL-3.0 module).", "A, B",
     'qrisk3(age=64, sex="female", ethnicity="indian", smoking="non", bmi=27, chol_hdl_ratio=4.0,\n       sbp=140, sbp_sd=8, townsend=0)'),
    ("qrisk2", "QRISK2", "QRISK2-2014", "2014", "United Kingdom", "nhs", "primary",
     "10-year CVD risk with ethnicity, deprivation and clinical conditions", DOI + "10.1136/bmj.39609.449676.25",
     "ClinRisk's own C code, including its 30,000-patient validation (2.9e-13) (LGPL-3.0 module).", "A",
     'qrisk2(age=64, sex="female", ethnicity="indian", smoking="non", bmi=27, chol_hdl_ratio=4.0, sbp=140)'),
    ("finrisk", "FINRISK", "FINRISKI 2.0 calculator", "2020", "Finland", "finland_flag", "primary",
     "10-year CHD, stroke and combined risk, ages 30-74", "https://www.laakarilehti.fi/tieteessa/alkuperaistutkimukset/sepelvaltimotaudin-ja-aivohalvauksen-riskin-arviointi-finriski-2.0-laskurilla/en",
     "Official Finnish calculator: 560 values, max relative difference 3e-16 (40 more floored by its display); Table 1 (4/4 equations).", "A, C",
     'finrisk(age=50, sex="male", smoker=1, total_chol=6.0, hdl_chol=1.3, sbp=140,\n        diabetes=0, parent_mi=1, parent_stroke=0, outcome="cvd")'),
    ("adrs", "ADRS", "Abu Dhabi Risk Study", "2024", "United Arab Emirates", "burj_al_arab", "primary",
     "10-year ASCVD, hard CAD and stroke, ages 18+", DOI + "10.1161/JAHA.124.035930",
     "Implemented from Table 2 and the supplement's SPSS syntax (32/32 terms agree); no worked example or calculator exists.", "C",
     'adrs(age=50, sex="male", smoker=1, sbp=135, dbp=85, tc_hdl=4.5, diabetes=0,\n     hypertension=1, bp_treated=1, vitamin_d=25, egfr=95, outcome="ascvd")'),
    ("procam", "PROCAM", "PROCAM scores (2007)", "2007", "Germany", "germany", "primary",
     "10-year major coronary event category (men and women 20-75) and ischaemic stroke/TIA risk (35-65)",
     DOI + "10.1111/j.1365-2362.2007.01888.x",
     "Tables 2-4 and 6-7 checked against the PDF (50/50 point bands, 95/95 risk-table rows, 39/39 stroke risks); 1 printed overlap documented.", "C",
     'procam(age=50, sex="male", ldl_chol=4.4, hdl_chol=1.0, triglycerides=2.0, sbp=140,\n       smoker=1, diabetes=0, family_history=0)'),
    ("regicor", "REGICOR", "Framingham recalibrated to Girona (REGICOR)", "2003", "Spain", "spain_flag", "primary",
     "10-year coronary events, ages 35-74", DOI + "10.1016/s0300-8932(03)76861-4",
     "Official REGICOR charts (men 198/200 cells within 0.01 points; women within 1 point); Wilson 1998 Table 6 verbatim; both worked examples.", "A, C",
     'regicor(age=55, sex="male", total_chol=6.0, hdl_chol=1.15, sbp=145, dbp=90,\n        diabetes=0, smoker=1)'),
    ("predict", "PREDICT", "PREDICT (New Zealand), 7 equations", "2018-2022", "New Zealand", "nz_flag", "primary",
     "5-year CVD (primary prevention, BMI, policy, type 2 diabetes), major bleeding, after CVD and after ACS",
     DOI + "10.1016/S0140-6736(18)30664-0",
     "The authors' R code run unmodified (21,000 patients, max difference 5e-13); 49/49 printed table rows and both worked examples.", "A, C",
     'predict_cvd(age=55, sex="female", ethnicity="maori", nzdep=4, smoking="ex", diabetes=0,\n            af=0, family_history=1, sbp=140, tc_hdl=4.5, bp_meds=1, lipid_meds=0,\n            antithrombotic=0)'),
    ("smart-reach", "SMART-REACH", "SMART-REACH", "2018", "International", "utrecht", "conditions",
     "10-year recurrent CVD risk and CVD-free life expectancy in established CVD", DOI + "10.1161/JAHA.118.009217",
     "The authors' calculator spreadsheet (600 patients, 3e-15); a life-expectancy defect in it documented.", "A",
     'smart_reach(age=65, sex="male", smoker=0, diabetes=1, sbp=135, total_chol=4.5,\n            creatinine=90, cad=1, cerebrovascular=0, pad=0)'),
    ("life-cvd2", "LIFE-CVD2", "LIFE-CVD2", "2024", "Europe", "hourglass", "primary",
     "Lifetime risk to 80, CVD-free life expectancy and treatment benefit", PM + "Hageman+LIFE-CVD2+Eur+J+Prev+Cardiol+2024",
     "The authors' 16 published charts (7,167/7,168 cells within printed precision).", "A",
     'life_cvd2(age=50, sex="female", smoker=1, sbp=150, total_chol=6.0, hdl_chol=1.4,\n          region="moderate", outcome="lifetime_risk")'),
    ("dial2", "DIAL2", "DIAL2 (type 2 diabetes)", "2023", "Europe (derived in Sweden)", "insulin", "conditions",
     "10-year and lifetime CVD risk and CVD-free life expectancy in type 2 diabetes", PM + "Ostergaard+DIAL2+Eur+J+Prev+Cardiol+2023",
     "Official U-Prevent calculator (58/60 displayed values, 20 patients; 2 mis-copied) and the paper's two worked examples.", "A, C",
     'dial2(age=60, sex="male", smoker=0, sbp=140, total_chol=5.0, hdl_chol=1.1, hba1c=58,\n      egfr=80, age_at_diagnosis=55, region="moderate", outcome="lifetime_risk")'),
    ("cha2ds2-vasc", "CHA2DS2-VASc", "CHA2DS2-VASc (stroke in AF)", "2010", "Europe", "stroke_ct", "conditions",
     "Stroke risk points in atrial fibrillation (0-9)", DOI + "10.1378/chest.09-1584",
     "Implemented from the paper's point table; published rules checked.", "C",
     'cha2ds2_vasc(age=72, sex="female", heart_failure=0, hypertension=1, diabetes=0,\n             stroke_tia_te=0, vascular_disease=0)'),
    ("chads2", "CHADS2", "CHADS2 (stroke in AF)", "2001", "United States", "ecg", "conditions",
     "Stroke risk points in atrial fibrillation (0-6)", DOI + "10.1001/jama.285.22.2864",
     "Implemented from the paper's point table; Gage's worked example reproduced.", "C",
     'chads2(age=76, heart_failure=1, hypertension=1, diabetes=0, stroke_tia=0)'),
    ("has-bled", "HAS-BLED", "HAS-BLED (bleeding on anticoagulation)", "2010", "Europe", "blood_bag", "conditions",
     "Major bleeding risk points on anticoagulation (0-9)", DOI + "10.1378/chest.10-0134",
     "Implemented from the paper's point table; published rules checked.", "C",
     'has_bled(age=70, hypertension=1, abnormal_renal=0, abnormal_liver=0, stroke=0,\n         bleeding=0, labile_inr=1, drugs=1, alcohol=0)'),
    ("heart", "HEART", "HEART score (chest pain)", "2008", "Netherlands", "chest_pain", "acute",
     "Major adverse cardiac events in chest pain (0-10)", PM + "HEART+score+chest+pain+Six+Backus+2008",
     "Published point rules (18/18); R RiskScorescvd (5,000 patients identical).", "C, B",
     'heart(history=1, ecg=1, age=58, risk_factors=2, atherosclerosis=0, troponin_ratio=0.5)'),
    ("timi", "TIMI", "TIMI risk score for UA/NSTEMI", "2000", "United States (Boston)", "heart_attack", "acute",
     "14-day adverse events in UA/NSTEMI (0-7)", DOI + "10.1001/jama.284.7.835",
     "Published point rules (11/11); R RiskScorescvd (5,000 patients identical).", "C, B",
     'timi(age=68, risk_factors=3, coronary_stenosis=1, st_deviation=1, anginal_events_24h=2,\n     aspirin_7d=1, elevated_markers=0)'),
    ("edacs", "EDACS", "Emergency Department Assessment of Chest pain Score", "2014", "Australia & New Zealand", "emergency", "acute",
     "Low-risk chest pain identification in the emergency department", PM + "Than+EDACS+Emerg+Med+Australas+2014",
     "Published point rules (29/29); R RiskScorescvd (5,000 patients identical).", "C, B",
     'edacs(age=52, sex="male", known_cad=0, risk_factors=2, diaphoresis=0,\n      radiates_arm_shoulder=1, pleuritic=0, palpation=0)'),
    ("grace", "GRACE", "GRACE in-hospital and 6-month mortality", "2003-2004", "International, 14 countries", "ambulance", "acute",
     "In-hospital (Granger 2003) and 6-month post-discharge (Eagle 2004) mortality points in ACS",
     DOI + "10.1001/archinte.163.19.2345",
     "Published nomograms (78/78 bands at both ends); the 6-month score equals the developers' archived GRACE calculator on all 87,808 band combinations.", "A (6-month), C (in-hospital)",
     'grace_hospital(killip=1, sbp=120, heart_rate=90, age=70, creatinine=1.2,\n               cardiac_arrest=0, st_deviation=1, elevated_enzymes=1)'),
    ("grace2", "GRACE 2.0", "GRACE 2.0 (1- and 3-year)", "2014", "International", "ambulance", "acute",
     "1- and 3-year death, and death or MI, after ACS admission; with substitutions for creatinine and Killip class",
     DOI + "10.1136/bmjopen-2013-004425",
     "The developers' calculator code run unmodified (24,572/24,572 displayed values); Fox 2014 Table 2 (24/24 hazard ratios).", "A, C",
     'grace2(age=70, heart_rate=90, sbp=120, cardiac_arrest=0, st_deviation=1,\n       elevated_enzymes=1, creatinine=1.2, killip=1, horizon=1, outcome="death")'),
    ("smart2", "SMART2", "SMART2 (established atherosclerotic disease)", "2022", "Europe and five other world regions", "utrecht", "conditions",
     "10-year recurrent MI, stroke or vascular death, ages 40-80", PM + "Hageman+SMART2+Eur+Heart+J+2022",
     "Official U-Prevent calculator (50/50 comparable patients); the supplement's SBP scale and a missing constant recovered from it.", "A",
     'smart2(age=65, sex="male", smoker=0, diabetes=1, sbp=135, non_hdl_chol=3.5, egfr=75,\n       hscrp=2.0, years_since_diagnosis=5, cad=1, cerebrovascular=0, pad=0, aaa=0,\n       region="europe_moderate")'),
    ("assign", "ASSIGN", "ASSIGN 1.0 and 2.0 (Scotland)", "2007-2025", "Scotland", "nhs", "primary",
     "10-year CVD risk with social deprivation (SIMD) and family history, ages 30-74", PM + "Welsh+ASSIGN+version+2.0+Heart+2025",
     "Version 2.0: official NHS Scotland calculator, 200/200 displayed scores; version 1.0: constants vs the 2007 appendix.", "A (2.0), C (1.0)",
     'assign(age=55, sex="male", total_chol=6.0, hdl_chol=1.2, sbp=140, diabetes=0,\n       family_history=1, cigarettes_per_day=10, simd_score=20)'),
    ("cuore", "CUORE", "CUORE Project individual score", "2007", "Italy", "global", "primary",
     "10-year first major CVD event, ages 35-69", "https://riviste.unimi.it/index.php/ijphjournal/article/view/23215",
     "Official ISS online calculator (300/300 displayed values); Table 2 hazard ratios (13/14; 1 misprint).", "A, C",
     'cuore(age=55, sex="male", sbp=140, tc=220, hdl=50, smoker=1, diabetes=0, bp_treatment=0)'),
    ("china-par", "China-PAR", "Prediction for ASCVD Risk in China", "2016", "China", "asia_pacific", "primary",
     "10-year first ASCVD event (MI, CHD death, stroke) with waist, region and urbanisation",
     DOI + "10.1161/CIRCULATIONAHA.116.022367",
     "Official calculator (cvdrisk.com.cn): 550/550 displayed values with coefficients recovered within 0.005 of the paper's; the printed 2-decimal coefficients alone miss it by up to 12.7 points.", "A, C",
     'china_par(age=60, sex="male", sbp=130, bp_treated=0, total_chol=5.4, hdl_chol=1.4,\n          waist=80, smoker=0, diabetes=1, north=1, urban=1, family_history=0)'),
    ("suita", "Suita", "Suita score (2014)", "2014", "Japan", "asia_pacific", "primary",
     "10-year CHD points with CKD, total- or LDL-cholesterol sheet", PM + "Nishimura+Suita+J+Atheroscler+Thromb+2014",
     "Point sheets checked against the paper's Cox coefficients; one baseline survival fits both probability tables; two web calculators of the JAS 2017 adaptation agree on every shared item.", "C (+ partial B)",
     'suita(age=55, sex="male", smoker=1, diabetes=0, sbp=140, dbp=85, hdl=50, egfr=70, tc=220)'),
    ("suita-cvd", "Suita CVD", "Suita CVD risk model (2020)", "2020", "Japan", "asia_pacific", "primary",
     "10-year stroke or CHD points, with and without ECG", DOI + "10.5551/jat.48843",
     "Every point equals 10 x the paper's coefficient (40/40); Table 4's probability bands as printed; R Jcvrisk gives the same band except at its SBP 139 / DBP 89 cut-point gap.", "C (+ partial B)",
     'suita_cvd(age=55, sex="male", sbp=135, dbp=85, bp_treated=0, hdl=45, smoker=1,\n          diabetes=0, proteinuria=0, ldl=150)'),
    ("who-ish", "WHO/ISH", "WHO/ISH risk charts (2007)", "2007", "Global, 14 WHO subregions", "who_flag", "primary",
     "10-year fatal and non-fatal CVD risk band, with and without cholesterol", PM + "Mendis+WHO+ISH+risk+prediction+charts+2007",
     "Every cell of the guideline's two printed charts (768/768) and R whoishRisk (3 transcription errors in it corrected).", "A, B",
     'who_ish(age=60, sex="male", smoker=1, sbp=150, diabetes=0, region="eur_a", total_chol=6.0)'),
]
REPORT = {"suita-cvd": "suita_cvd", "score2-op": "score2", "score2-diabetes": "score2", "score2-asia-pacific": "score2", "cha2ds2-vasc": "af",
          "chads2": "af", "has-bled": "af", "uniscore": None}
GITHUB = "https://github.com/m-aljasem/PyCVDRisk/blob/HEAD/"
GROUPS = {"primary": "Primary prevention", "conditions": "Diabetes, kidney disease, established CVD and atrial fibrillation",
          "acute": "Acute chest pain and coronary syndromes"}


def main():
    ns = {k: getattr(cvd_risk, k) for k in cvd_risk.__all__}
    photos = json.loads((Path(__file__).parents[1] / "data" / "photos.json").read_text())
    out = []
    for slug, name, full, year, region, photo, group, outcome, paper, verif, ev, ex in M:
        r = np.asarray(eval(ex, ns), dtype=float)
        assert np.isfinite(r).all(), (slug, r)
        assert photo in photos, photo
        rep = REPORT.get(slug, slug.replace("-", "_"))
        if rep:
            assert (Path(__file__).parents[2] / "reference" / rep / "REPORT.md").exists(), rep
        out.append({"slug": slug, "report": rep and f"{GITHUB}reference/{rep}/REPORT.md", "name": name, "fullName": full, "year": year, "region": region, "photo": photo, "group": group,
                        "outcome": outcome, "paperUrl": paper, "verification": verif, "evidence": ev,
                        "example": f"from cvd_risk import {ex.split('(')[0]}\n\n{ex}",
                        "result": np.array2string(r, precision=4, separator=", ") if r.ndim else f"{float(r):.4g}"})
    dest = Path(__file__).parents[1] / "data" / "models.json"
    dest.write_text(json.dumps({"groups": GROUPS, "models": out, "version": cvd_risk.__version__}, indent=1) + "\n")
    print(len(out), "models ->", dest)
    # sitemap and robots.txt: the site's pages plus one page per model
    site = "https://pycvdrisk.aljasem.eu.org"
    paths = ["/", "/demo", "/about", "/credits"] + [f"/docs/models/{m['slug']}" for m in out]
    public = Path(__file__).parents[1] / "public"
    (public / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                        + "".join(f"  <url><loc>{site}{p if p != '/' else ''}</loc></url>\n" for p in paths) + "</urlset>\n")
    (public / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {site}/sitemap.xml\n")


if __name__ == "__main__":
    main()
