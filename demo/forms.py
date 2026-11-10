"""What the demo asks for, per model: one entry per website model slug.

Each model has one or more variants (e.g. Framingham "Laboratory" / "BMI"). A variant is the
cvd_risk function it calls, the inputs it shows, and how to read the number that comes back.
To add a model: write its Variant(s) below and add the slug to FORMS. Allowed category values
come from the package itself, so they cannot drift from what the functions accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cvd_risk as cr
from cvd_risk.dial2 import REGION_NAMES as DIAL2_REGIONS
from cvd_risk.life_cvd2 import REGION_NAMES as LIFE_REGIONS
from cvd_risk.pce import RACES
from cvd_risk.predict import ACS_TYPE, ETHNICITY as NZ_ETHNICITY, POLICY_ETHNICITY, SMOKING as NZ_SMOKING
from cvd_risk.prevent import OUTCOMES as PREVENT_OUTCOMES
from cvd_risk.procam.procam import RISK_PERCENT as PROCAM_RISK
from cvd_risk.qrisk3 import DIABETES as Q_DIABETES, ETHNICITY as Q_ETHNICITY, SMOKING as Q_SMOKING
from cvd_risk.score.coefficients import REGIONS as SCORE_REGIONS
from cvd_risk.score2 import REGIONS as ESC_REGIONS
from cvd_risk.score2.coefficients import DIPSTICK
from cvd_risk.smart_reach import REGION_NAMES as SMART_REGIONS
from cvd_risk.who2019.coefficients import REGIONS as WHO_REGIONS

# shortcut: country lists come from the packages' private data loaders (no public list yet);
# switch to a public constant if the library adds one.
from cvd_risk.globorisk.globorisk import _data as _globorisk_data
from cvd_risk.who2019.who2019 import _data as _who2019_data

GLOBORISK_COUNTRIES = tuple(sorted(_globorisk_data()[0]))
WHO_COUNTRIES = tuple(sorted(_who2019_data()[0]))


@dataclass(frozen=True)
class Field:
    name: str  # the keyword argument of the cvd_risk function
    label: str
    kind: str  # "num", "bool" or "choice"
    default: Any = None
    unit: str = ""
    lo: float | None = None
    hi: float | None = None
    step: float = 1.0
    options: tuple = ()
    help: str = ""


@dataclass(frozen=True)
class Variant:
    label: str
    fn: Callable
    fields: tuple[Field, ...]
    output: str = "risk"  # "risk" (fraction), "points" or "age" (CVD-free life expectancy)
    max_points: int | None = None
    fixed: dict = field(default_factory=dict)  # arguments passed without showing an input
    points_to_risk: Callable[[float], str | None] | None = None


def num(name, label, default, unit="", lo=None, hi=None, step=1.0, help=""):
    return Field(name, label, "num", default, unit, lo, hi, step, help=help)


def flag(name, label, default=False, help=""):
    return Field(name, label, "bool", default, help=help)


def choice(name, label, options, default=None, help=""):
    return Field(name, label, "choice", default if default is not None else options[0], options=tuple(options), help=help)


# Shared inputs (SI units, as the library expects)
def AGE(default=55, lo=18, hi=100):
    return num("age", "Age", default, "years", lo, hi)


SEX = choice("sex", "Sex", ("male", "female"))
SMOKER = flag("smoker", "Current smoker")
SBP = num("sbp", "Systolic blood pressure", 140, "mmHg", 70, 250)
DBP = num("dbp", "Diastolic blood pressure", 85, "mmHg", 40, 150)
TC = num("total_chol", "Total cholesterol", 5.5, "mmol/L", 1.0, 20.0, 0.1)
HDL = num("hdl_chol", "HDL cholesterol", 1.3, "mmol/L", 0.2, 5.0, 0.1)
DIABETES = flag("diabetes", "Diabetes")
BP_TREATED = flag("bp_treated", "On blood-pressure treatment")
EGFR = num("egfr", "eGFR", 85, "mL/min/1.73m²", 5, 200)
BMI = num("bmi", "Body-mass index", 27.0, "kg/m²", 12.0, 70.0, 0.1)
ESC_REGION = choice("region", "ESC risk region", ESC_REGIONS, "moderate",
                    help="Low: e.g. France, Spain, UK. Moderate: e.g. Germany, Italy. High/very high: Central and Eastern Europe.")
TC_HDL = num("tc_hdl", "Total / HDL cholesterol ratio", 4.0, "", 1.0, 20.0, 0.1)
CHOL_HDL = num("chol_hdl_ratio", "Total / HDL cholesterol ratio", 4.0, "", 1.0, 20.0, 0.1)
NZDEP = num("nzdep", "NZDep quintile", 3, "1 = least deprived", 1, 5)
ETH_NZ = choice("ethnicity", "Ethnicity", NZ_ETHNICITY)
SMOKING_NZ = choice("smoking", "Smoking", NZ_SMOKING)

SCORE2_CORE = (AGE(55), SEX, SMOKER, SBP, TC, HDL, DIABETES)
QRISK_CORE = (AGE(55, 25, 84), SEX, choice("ethnicity", "Ethnicity", Q_ETHNICITY),
              choice("smoking", "Smoking", Q_SMOKING), BMI, CHOL_HDL, SBP)
QRISK_TAIL = (num("townsend", "Townsend deprivation score", 0.0, "0 = UK average", -8.0, 12.0, 0.1),
              choice("diabetes", "Diabetes", Q_DIABETES), flag("af", "Atrial fibrillation"),
              flag("rheumatoid_arthritis", "Rheumatoid arthritis"), flag("ckd", "Chronic kidney disease (stage 3-5)"),
              flag("bp_treated", "On blood-pressure treatment"), flag("family_history", "Angina or heart attack in a 1st-degree relative < 60"))
PREDICT_MEDS = (flag("bp_meds", "Blood-pressure medication"), flag("lipid_meds", "Lipid-lowering medication"),
                flag("antithrombotic", "Antiplatelet or anticoagulant"))


def _procam_risk(points: float) -> str | None:
    p = int(points)
    if p < min(PROCAM_RISK):
        return f"< {PROCAM_RISK[min(PROCAM_RISK)]}%"
    if p > max(PROCAM_RISK):
        return f"> {PROCAM_RISK[max(PROCAM_RISK)]}%"
    return f"{PROCAM_RISK[p]}%"


FORMS: dict[str, list[Variant]] = {
    "score2": [Variant("SCORE2", cr.score2, SCORE2_CORE + (ESC_REGION,))],
    "score2-op": [Variant("SCORE2-OP", cr.score2_op, (AGE(75),) + SCORE2_CORE[1:] + (ESC_REGION,))],
    "score2-diabetes": [Variant("SCORE2-Diabetes", cr.score2_diabetes, SCORE2_CORE[:6] + (
        flag("diabetes", "Diabetes", True),
        num("age_at_diabetes_diagnosis", "Age at diabetes diagnosis", 50, "years", 0, 100),
        num("hba1c", "HbA1c", 58, "mmol/mol", 15, 200), EGFR, ESC_REGION))],
    "score2-ckd": [
        Variant("eGFR", cr.score2_ckd, SCORE2_CORE + (num("egfr", "eGFR", 45, "mL/min/1.73m²", 5, 200), ESC_REGION),
                fixed={"addon": "egfr"}),
        Variant("eGFR + ACR", cr.score2_ckd, SCORE2_CORE + (num("egfr", "eGFR", 45, "mL/min/1.73m²", 5, 200),
                num("acr", "Urine albumin/creatinine ratio", 30.0, "mg/g", 0.1, 5000.0, 0.1), ESC_REGION),
                fixed={"addon": "acr"}),
        Variant("eGFR + dipstick", cr.score2_ckd, SCORE2_CORE + (num("egfr", "eGFR", 45, "mL/min/1.73m²", 5, 200),
                choice("dipstick", "Urine dipstick protein", DIPSTICK), ESC_REGION), fixed={"addon": "dipstick"}),
    ],
    "score2-asia-pacific": [Variant("SCORE2 Asia-Pacific", cr.score2_asia_pacific, SCORE2_CORE + (
        choice("region", "Asia-Pacific risk region", ESC_REGIONS, "moderate",
               help="As assigned per country in the source's Supplementary Table 4."),))],
    "score": [Variant("SCORE", cr.score, (AGE(55, 40, 65), SEX, TC, SBP, SMOKER,
                                          choice("region", "Risk region", SCORE_REGIONS, "high")))],
    "globorisk": [
        Variant(label, cr.globorisk, (AGE(55, 40, 84), SEX, SBP, SMOKER,
                choice("country", "Country (ISO 3166 code)", GLOBORISK_COUNTRIES, "USA"),
                num("year", "Baseline year", 2020, "", 2000, 2020)) + extra + (num("horizon", "Horizon", 10, "years", 1, 10),),
                fixed={"version": version})
        for label, version, extra in (("Laboratory", "lab", (TC, DIABETES)), ("Office (BMI)", "office", (BMI,)),
                                      ("Fatal CVD", "fatal", (TC, DIABETES)))
    ],
    "who2019": [
        Variant("Laboratory", cr.who2019, (AGE(55, 40, 80), SEX, SBP, SMOKER, TC, DIABETES,
                choice("region", "GBD region", WHO_REGIONS, "western_europe")), fixed={"version": "lab"}),
        Variant("Non-laboratory", cr.who2019, (AGE(55, 40, 80), SEX, SBP, SMOKER, BMI,
                choice("region", "GBD region", WHO_REGIONS, "western_europe")), fixed={"version": "nonlab"}),
        Variant("Country-recalibrated", cr.who2019, (AGE(55, 40, 80), SEX, SBP, SMOKER, TC, DIABETES,
                choice("country", "Country (ISO 3166 code)", WHO_COUNTRIES, "DEU")),
                fixed={"version": "lab", "recalibration": "country"}),
    ],
    "prevent": [
        Variant(label, cr.prevent, (AGE(55, 30, 79), SEX, SBP, BP_TREATED, TC, HDL, flag("statin", "On a statin"),
                DIABETES, SMOKER, EGFR, BMI) + extra + (
                choice("years", "Horizon", (10, 30)), choice("outcome", "Outcome", PREVENT_OUTCOMES)),
                fixed={"model": model})
        for label, model, extra in (
            ("Base", "base", ()),
            ("Full", "full", (num("uacr", "Urine albumin/creatinine ratio", 10.0, "mg/g", 0.1, 25000.0, 0.1),
                              num("hba1c", "HbA1c", 5.6, "%", 3.0, 15.0, 0.1),
                              num("sdi", "Social Deprivation Index decile", 5, "", 1, 10))),
        )
    ],
    "pce": [Variant("Pooled Cohort Equations", cr.pce, (AGE(55, 40, 79), SEX, choice("race", "Race", RACES), TC, HDL,
                                                         SBP, BP_TREATED, SMOKER, DIABETES))],
    "framingham": [
        Variant("Laboratory", cr.framingham, (AGE(55, 30, 74), SEX, SBP, BP_TREATED, SMOKER, DIABETES, TC, HDL),
                fixed={"model": "lab"}),
        Variant("BMI", cr.framingham, (AGE(55, 30, 74), SEX, SBP, BP_TREATED, SMOKER, DIABETES, BMI),
                fixed={"model": "bmi"}),
    ],
    "reynolds": [Variant("Reynolds", cr.reynolds, (AGE(55, 45, 80), SEX, SBP, TC, HDL,
                         num("hscrp", "hsCRP", 2.0, "mg/L", 0.1, 50.0, 0.1), SMOKER,
                         flag("parental_history", "Parent with MI before 60"), DIABETES,
                         num("hba1c", "HbA1c (women with diabetes)", 7.0, "%", 3.0, 15.0, 0.1)))],
    "qrisk3": [Variant("QRISK3", cr.qrisk3, QRISK_CORE + (
        num("sbp_sd", "SD of recent SBP readings", 8.0, "mmHg", 0.0, 50.0, 0.1),) + QRISK_TAIL + (
        flag("atypical_antipsychotics", "Atypical antipsychotics"), flag("corticosteroids", "Regular corticosteroids"),
        flag("erectile_dysfunction", "Erectile dysfunction"), flag("migraine", "Migraine"),
        flag("severe_mental_illness", "Severe mental illness"), flag("sle", "Systemic lupus erythematosus")))],
    "qrisk2": [Variant("QRISK2", cr.qrisk2, QRISK_CORE + QRISK_TAIL)],
    "finrisk": [Variant(label, cr.finrisk, (AGE(55, 30, 64), SEX, SMOKER, TC, HDL, SBP, DIABETES,
                        flag("parent_mi", "Parent had a heart attack"), flag("parent_stroke", "Parent had a stroke")),
                        fixed={"outcome": outcome})
                for label, outcome in (("CVD", "cvd"), ("CHD", "chd"), ("Stroke", "stroke"))],
    "adrs": [Variant(label, cr.adrs, (AGE(50), SEX, SMOKER, SBP, DBP, TC_HDL, DIABETES,
                     flag("hypertension", "Hypertension"), BP_TREATED,
                     num("vitamin_d", "Vitamin D", 25.0, "ng/mL", 1.0, 150.0, 0.5), EGFR), fixed={"outcome": outcome})
             for label, outcome in (("ASCVD", "ascvd"), ("Hard CAD", "cad"), ("Stroke", "stroke"))],
    "procam": [Variant("PROCAM", cr.procam, (AGE(50, 20, 80), choice("sex", "Sex", ("male",), help="Derived in men only."),
                       num("ldl_chol", "LDL cholesterol", 3.5, "mmol/L", 0.5, 15.0, 0.1), HDL,
                       num("triglycerides", "Triglycerides", 1.7, "mmol/L", 0.2, 30.0, 0.1), SBP, SMOKER, DIABETES,
                       flag("family_history", "MI in a 1st-degree relative before 60")),
                       output="points", points_to_risk=_procam_risk)],
    "regicor": [Variant(label, cr.regicor, (AGE(55, 35, 74), SEX, TC, HDL, SBP, DBP, DIABETES, SMOKER),
                        fixed={"calibration": cal})
                for label, cal in (("REGICOR (Girona)", "gerona"), ("Original Framingham", "framingham"))],
    "predict": [
        Variant("Primary prevention", cr.predict_cvd, (AGE(55, 30, 79), SEX, ETH_NZ, NZDEP, SMOKING_NZ, DIABETES,
                flag("af", "Atrial fibrillation"), flag("family_history", "Family history of premature CVD"),
                SBP, TC_HDL) + PREDICT_MEDS),
        Variant("With BMI", cr.predict_cvd_bmi, (AGE(55, 30, 79), SEX, ETH_NZ, NZDEP, SMOKING_NZ, DIABETES,
                flag("af", "Atrial fibrillation"), flag("family_history", "Family history of premature CVD"),
                SBP, TC_HDL, BMI) + PREDICT_MEDS),
        Variant("Type 2 diabetes", cr.predict_t2d, (AGE(55, 30, 79), SEX, ETH_NZ, NZDEP, flag("smoker", "Current smoker"),
                flag("af", "Atrial fibrillation"), flag("family_history", "Family history of premature CVD"),
                flag("lipid_meds", "Lipid-lowering medication"), flag("antithrombotic", "Antiplatelet or anticoagulant"),
                flag("bp_meds", "Blood-pressure medication"), flag("oral_hypoglycaemic", "Oral glucose-lowering drugs"),
                flag("insulin", "Insulin"), SBP, TC_HDL, BMI, EGFR,
                num("acr", "Albumin/creatinine ratio", 2.0, "mg/mmol", 0.0, 500.0, 0.1),
                num("hba1c", "HbA1c", 58, "mmol/mol", 15, 200), num("years_since_diagnosis", "Years since diagnosis", 5, "years", 0, 60))),
        Variant("Major bleeding", cr.predict_bleed, (AGE(55, 30, 79), SEX, ETH_NZ, NZDEP, SMOKING_NZ, DIABETES,
                flag("family_history", "Family history of premature CVD"), flag("lipid_meds", "Lipid-lowering medication"),
                flag("bp_meds", "Blood-pressure medication"), flag("cancer", "Cancer"), flag("gi_bleed", "Previous GI bleed"),
                flag("peptic_ulcer", "Peptic ulcer"), flag("alcohol", "Alcohol-related condition"), flag("liver", "Liver disease"),
                flag("peptic_ulcer_meds", "Peptic-ulcer medication"), flag("nsaid", "NSAIDs"), flag("steroids", "Steroids"),
                flag("ssri", "SSRIs"), SBP, TC_HDL)),
        Variant("After ACS", cr.predict_post_acs, (AGE(65, 18, 100), SEX, ETH_NZ, NZDEP, flag("smoker", "Current smoker"),
                DIABETES, flag("af", "Atrial fibrillation"), flag("heart_failure", "Heart failure")) + PREDICT_MEDS + (
                SBP, TC_HDL, BMI, num("creatinine", "Creatinine", 90, "µmol/L", 20, 1500),
                num("hba1c", "HbA1c", 40, "mmol/mol", 15, 200), num("days_since_acs", "Days since ACS", 30, "days", 0, 20000),
                choice("acs_type", "ACS type", ACS_TYPE))),
        Variant("After CVD", cr.predict_post_cvd, (AGE(65, 18, 100), SEX, ETH_NZ, NZDEP, flag("smoker", "Current smoker"),
                DIABETES, flag("af", "Atrial fibrillation"), flag("heart_failure", "Heart failure"),
                flag("other_vd", "Angina, PVD or non-hospitalised cerebrovascular disease only")) + PREDICT_MEDS + (
                SBP, TC_HDL, BMI, num("creatinine", "Creatinine", 90, "µmol/L", 20, 1500),
                num("hba1c", "HbA1c", 40, "mmol/mol", 15, 200),
                num("days_since_event", "Days since event", 365, "days", 0, 20000))),
        Variant("Policy (VARIANZ)", cr.predict_cvd_policy, (AGE(55, 30, 79), SEX,
                choice("ethnicity", "Ethnicity", POLICY_ETHNICITY), NZDEP, DIABETES, flag("af", "Atrial fibrillation"),
                ) + PREDICT_MEDS),
    ],
    "smart-reach": [Variant(label, cr.smart_reach, (AGE(65, 45, 80), SEX, SMOKER, DIABETES, SBP, TC,
                            num("creatinine", "Creatinine", 90, "µmol/L", 20, 1500),
                            flag("cad", "Coronary artery disease", True), flag("cerebrovascular", "Cerebrovascular disease"),
                            flag("pad", "Peripheral artery disease"), flag("af", "Atrial fibrillation"),
                            flag("heart_failure", "Heart failure"), choice("region", "Region", SMART_REGIONS)),
                            output=output, fixed={"outcome": outcome})
                    for label, outcome, output in (("10-year risk", "risk10", "risk"),
                                                   ("CVD-free life expectancy", "life_expectancy", "age"))],
    "life-cvd2": [Variant(label, cr.life_cvd2, (AGE(50, 35, 90), SEX, SMOKER, SBP, TC, HDL,
                          choice("region", "ESC risk region", LIFE_REGIONS, "moderate")), output=output,
                          fixed={"outcome": outcome})
                  for label, outcome, output in (("Lifetime risk to 80", "lifetime_risk", "risk"),
                                                 ("CVD-free life expectancy", "life_expectancy", "age"))],
    "dial2": [Variant(label, cr.dial2, (AGE(60, 30, 85), SEX, SMOKER, SBP, TC, HDL,
                      num("hba1c", "HbA1c", 58, "mmol/mol", 15, 200), EGFR,
                      num("age_at_diagnosis", "Age at diabetes diagnosis", 52, "years", 0, 100),
                      choice("region", "ESC risk region", DIAL2_REGIONS)), output=output, fixed={"outcome": outcome})
              for label, outcome, output in (("10-year risk", "risk10", "risk"), ("Lifetime risk", "lifetime_risk", "risk"),
                                             ("CVD-free life expectancy", "life_expectancy", "age"))],
    "cha2ds2-vasc": [Variant("CHA₂DS₂-VASc", cr.cha2ds2_vasc, (AGE(72), SEX, flag("heart_failure", "Heart failure / LV dysfunction"),
                             flag("hypertension", "Hypertension"), DIABETES,
                             flag("stroke_tia_te", "Prior stroke, TIA or thromboembolism"),
                             flag("vascular_disease", "Vascular disease (prior MI, PAD, aortic plaque)")),
                             output="points", max_points=9)],
    "chads2": [Variant("CHADS₂", cr.chads2, (AGE(72), flag("heart_failure", "Recent heart failure"),
                       flag("hypertension", "Hypertension"), DIABETES, flag("stroke_tia", "Prior stroke or TIA")),
                       output="points", max_points=6)],
    "has-bled": [Variant("HAS-BLED", cr.has_bled, (AGE(70), flag("hypertension", "Uncontrolled hypertension (SBP > 160)"),
                         flag("abnormal_renal", "Abnormal renal function"), flag("abnormal_liver", "Abnormal liver function"),
                         flag("stroke", "Prior stroke"), flag("bleeding", "Bleeding history or predisposition"),
                         flag("labile_inr", "Labile INR"), flag("drugs", "Antiplatelets or NSAIDs"),
                         flag("alcohol", "Alcohol ≥ 8 units/week")), output="points", max_points=9)],
    "heart": [Variant("HEART", cr.heart, (
        choice("history", "History", (0, 1, 2), help="0 slightly, 1 moderately, 2 highly suspicious"),
        choice("ecg", "ECG", (0, 1, 2), help="0 normal, 1 non-specific repolarisation, 2 significant ST deviation"),
        AGE(55), num("risk_factors", "Number of risk factors", 1, "", 0, 7),
        flag("atherosclerosis", "Known atherosclerotic disease"),
        num("troponin_ratio", "Troponin / decision limit", 0.5, "×", 0.0, 100.0, 0.1)), output="points", max_points=10)],
    "timi": [Variant("TIMI UA/NSTEMI", cr.timi, (AGE(60), num("risk_factors", "Number of CAD risk factors", 1, "", 0, 5),
                     flag("coronary_stenosis", "Known coronary stenosis ≥ 50%"), flag("st_deviation", "ST deviation ≥ 0.5 mm"),
                     num("anginal_events_24h", "Anginal events in 24 h", 1, "", 0, 20),
                     flag("aspirin_7d", "Aspirin in the last 7 days"), flag("elevated_markers", "Elevated cardiac markers")),
                     output="points", max_points=7)],
    "edacs": [Variant("EDACS", cr.edacs, (AGE(50), SEX, flag("known_cad", "Known CAD"),
                      num("risk_factors", "Number of risk factors", 1, "", 0, 7),
                      flag("diaphoresis", "Diaphoresis"), flag("radiates_arm_shoulder", "Pain radiates to arm or shoulder"),
                      flag("pleuritic", "Pain worse with inspiration"), flag("palpation", "Pain reproduced by palpation")),
                      output="points")],
    "grace": [
        Variant("In-hospital death", cr.grace_hospital, (choice("killip", "Killip class", (1, 2, 3, 4)), SBP,
                num("heart_rate", "Heart rate", 80, "bpm", 20, 250), AGE(65),
                num("creatinine", "Creatinine", 1.0, "mg/dL", 0.1, 20.0, 0.1), flag("cardiac_arrest", "Cardiac arrest at admission"),
                flag("st_deviation", "ST-segment deviation"), flag("elevated_enzymes", "Elevated cardiac enzymes")),
                output="points"),
        Variant("6-month death", cr.grace_6month, (AGE(65), flag("chf_history", "History of heart failure"),
                flag("mi_history", "History of MI"), num("heart_rate", "Heart rate", 80, "bpm", 20, 250), SBP,
                flag("st_depression", "ST-segment depression"), num("creatinine", "Creatinine", 1.0, "mg/dL", 0.1, 20.0, 0.1),
                flag("elevated_enzymes", "Elevated cardiac enzymes"), flag("in_hospital_pci", "PCI in hospital")),
                output="points"),
    ],
}
