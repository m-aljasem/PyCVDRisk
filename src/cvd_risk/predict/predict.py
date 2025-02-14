"""PREDICT (New Zealand) 5-year risk equations, from the VIEW team's R package PredictRiskScores.

Seven models, each a Cox model: risk = 1 - S0 ** exp(LP - MEAN_LP), with continuous inputs centred
(see coefficients.py). Ethnicity is "european", "maori", "pacific", "indian" or "asian" (Chinese or
other Asian); the policy model also takes "other" (pooled with Asian there). NZDep is the deprivation
quintile 1-5.

Differences from the R package, all deliberate (see reference/predict/REPORT.md):
- Out-of-range ages give NaN; R clamps them (below 30 -> 30, above 79 -> 80).
- Bands are gap-free intervals on real numbers. R tests several bands with integer sequences
  (`bmi %in% 25:29.9`, `hba1c >= 40 & hba1c <= 64`, ...), so non-integer values in those ranges fall
  into no band there.
- Missing (NaN) binary inputs give NaN; R scores them as 0.
- Results are not rounded (R rounds to 4 decimals by default).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.predict import coefficients as C

SOURCE = ("VIEW research programme, University of Auckland: R package PredictRiskScores 0.9 "
          "(github.com/VIEW2020/PredictRiskScores); Pylypchuk R et al. Lancet 2018;391:1897-907; "
          "Selak V et al. Ann Intern Med 2019;170:357-68; Mehta S et al. Int J Epidemiol 2018;47:1571-84; "
          "Pylypchuk R et al. Lancet 2021;397:2264-74; Poppe KK et al. Heart 2017;103:891 and 2020")
ETHNICITY = ("european", "maori", "pacific", "indian", "asian")
POLICY_ETHNICITY = (*ETHNICITY, "other")
SMOKING = ("never", "ex", "current")
ACS_TYPE = ("unstable_angina", "nstemi", "stemi")
SEXES = ((1, "male"), (0, "female"))


def _eth(ethnicity, allowed=ETHNICITY):
    e = as_category(ethnicity, allowed).astype(float)
    e = np.where(e >= 0, e, np.nan)
    return {k: np.where(np.isnan(e), np.nan, e == i) for i, k in enumerate(allowed) if i}


def _smoking(smoking):
    s = as_category(smoking, SMOKING).astype(float)
    s = np.where(s >= 0, s, np.nan)
    return {"ex_smoker": np.where(np.isnan(s), np.nan, s == 1), "smoker": np.where(np.isnan(s), np.nan, s == 2)}


def _nzdep(nzdep):
    q = as_float(nzdep)
    return np.where(np.isin(q, (1, 2, 3, 4, 5)), q, np.nan)


def _pos(x):
    x = as_float(x)
    return np.where(x > 0, x, np.nan)


def _bands(x, edges, names, missing=None):
    """One-hot bands [edges[i-1], edges[i]); names[i] is None for the reference band. NaN x -> the
    `missing` dummy if given (else the row is NaN); negative x -> NaN."""
    x = as_float(x)
    k = np.digitize(np.where(np.isnan(x), 0, x), edges)
    bad = (x < 0) | (np.isnan(x) & (missing is None))
    out = {n: np.where(bad, np.nan, (k == i) & ~np.isnan(x)) for i, n in enumerate(names) if n}
    if missing:
        out[missing] = np.where(bad, np.nan, np.isnan(x))
    return out


def _risk(model, sex, x):
    """x: feature -> array, or callable(sex_name) -> dict for sex-specific centring."""
    sx = as_sex(sex)
    out = np.nan
    for code, name in SEXES:
        f = x(name) if callable(x) else x
        lp = sum(b * f[k] for k, b in model["beta"][name].items())
        out = np.where(sx == code, 1 - model["s0"][name] ** np.exp(lp - model["mean_lp"][name]), out)
    return out


def _centred(model, name, **v):
    c = model["centre"][name]
    return {k: (x - c[k] if k in c else x) for k, x in v.items()}


def _primary(model, age, sex, ethnicity, nzdep, smoking, diabetes, af, family_history, sbp, tc_hdl,
             bp_meds, lipid_meds, antithrombotic, extra=None):
    a = as_float(age)
    a = np.where((a >= 30) & (a < 80), a, np.nan)
    dm, bpl = as_bool(diabetes), as_bool(bp_meds)
    fixed = {**_eth(ethnicity), **_smoking(smoking), "diabetes": dm, "af": as_bool(af),
             "family_history": as_bool(family_history), "lipid_meds": as_bool(lipid_meds),
             "antithrombotic": as_bool(antithrombotic), "bp_meds": bpl, **(extra or {})}

    def x(name):
        c = _centred(model, name, age=a, nzdep=_nzdep(nzdep), sbp=_pos(sbp), tc_hdl=_pos(tc_hdl))
        return {**fixed, **c, "age_x_diabetes": c["age"] * dm, "age_x_sbp": c["age"] * c["sbp"],
                "bp_meds_x_sbp": bpl * c["sbp"]}

    return _risk(model, sex, x)


def predict_cvd(age, sex, ethnicity, nzdep, smoking, diabetes, af, family_history, sbp, tc_hdl,
                bp_meds, lipid_meds, antithrombotic) -> np.ndarray:
    """PREDICT-1 primary prevention (Pylypchuk 2018, Table 3): 5-year CVD risk, ages 30-79.
    smoking: "never" / "ex" / "current"; tc_hdl: total/HDL cholesterol ratio; flags 0/1."""
    return _primary(C.CVD, age, sex, ethnicity, nzdep, smoking, diabetes, af, family_history, sbp, tc_hdl,
                    bp_meds, lipid_meds, antithrombotic)


def predict_cvd_bmi(age, sex, ethnicity, nzdep, smoking, diabetes, af, family_history, sbp, tc_hdl, bmi,
                    bp_meds, lipid_meds, antithrombotic) -> np.ndarray:
    """PREDICT 2018.2 primary prevention with BMI bands (<18.5, 18.5-25 ref, 25-30, 30-35, 35-40, >=40;
    NaN = BMI not recorded, its own category)."""
    bands = _bands(bmi, [18.5, 25, 30, 35, 40], ["bmi_lt_18_5", None, "bmi_25_30", "bmi_30_35", "bmi_35_40",
                                                 "bmi_ge_40"], missing="bmi_missing")
    return _primary(C.CVD_BMI, age, sex, ethnicity, nzdep, smoking, diabetes, af, family_history, sbp, tc_hdl,
                    bp_meds, lipid_meds, antithrombotic, extra=bands)


def predict_cvd_policy(age, sex, ethnicity, nzdep, diabetes, af, bp_meds, lipid_meds, antithrombotic) -> np.ndarray:
    """VARIANZ 2018 policy-level equation (Mehta 2018), no clinical measurements; for populations,
    not individuals. ethnicity also accepts "other" (pooled with "asian")."""
    a = as_float(age)
    a = np.where((a >= 30) & (a < 80), a, np.nan)
    e = as_category(ethnicity, POLICY_ETHNICITY).astype(float)
    e = np.where(e >= 0, e, np.nan)
    eth = {k: np.where(np.isnan(e), np.nan, e == i) for i, k in enumerate(POLICY_ETHNICITY[:4]) if i}
    eth["other"] = np.where(np.isnan(e), np.nan, e >= 4)
    dm, f_af, bpl, at = as_bool(diabetes), as_bool(af), as_bool(bp_meds), as_bool(antithrombotic)

    def x(name):
        c = _centred(C.CVD_POLICY, name, age=a, nzdep=_nzdep(nzdep))
        return {**eth, **c, "diabetes": dm, "af": f_af, "lipid_meds": as_bool(lipid_meds), "antithrombotic": at,
                "bp_meds": bpl, "age_x_bp_meds": c["age"] * bpl, "age_x_diabetes": c["age"] * dm,
                "age_x_af": c["age"] * f_af, "bp_meds_x_diabetes": bpl * dm, "antithrombotic_x_diabetes": at * dm,
                "bp_meds_x_af": bpl * f_af}

    return _risk(C.CVD_POLICY, sex, x)


def predict_t2d(age, sex, ethnicity, nzdep, smoker, af, family_history, lipid_meds, antithrombotic, bp_meds,
                oral_hypoglycaemic, insulin, sbp, tc_hdl, bmi, egfr, acr, hba1c, years_since_diagnosis) -> np.ndarray:
    """PREDICT type 2 diabetes, no prior CVD (2018.1; Pylypchuk 2021). egfr mL/min/1.73m2, acr mg/mmol,
    hba1c mmol/mol."""
    a = as_float(age)
    a = np.where((a >= 30) & (a < 80), a, np.nan)
    r = as_float(acr)
    r = np.where(r >= 0, r, np.nan)
    yrs = as_float(years_since_diagnosis)
    m = C.T2D
    fixed = {**_eth(ethnicity), "smoker": as_bool(smoker), "af": as_bool(af),
             "family_history": as_bool(family_history), "lipid_meds": as_bool(lipid_meds),
             "antithrombotic": as_bool(antithrombotic), "bp_meds": as_bool(bp_meds),
             "oral": as_bool(oral_hypoglycaemic), "insulin": as_bool(insulin)}

    def x(name):
        c = _centred(m, name, age=a, nzdep=_nzdep(nzdep), sbp=_pos(sbp), tc_hdl=_pos(tc_hdl), bmi=_pos(bmi),
                     egfr=_pos(egfr), hba1c=_pos(hba1c), years=np.where(yrs >= 0, yrs, np.nan))
        return {**fixed, **c, "log_acr": np.log((r + m["acr_offset"]) / 1000) + m["log_acr_shift"][name]}

    return _risk(m, sex, x)


def predict_bleed(age, sex, ethnicity, nzdep, smoking, diabetes, family_history, lipid_meds, bp_meds, cancer,
                  gi_bleed, peptic_ulcer, alcohol, liver, peptic_ulcer_meds, nsaid, steroids, ssri, sbp,
                  tc_hdl) -> np.ndarray:
    """PREDICT major bleeding, no prior CVD (Selak 2019, Table 4): 5-year risk without aspirin."""
    a = as_float(age)
    a = np.where((a >= 30) & (a < 80), a, np.nan)
    x = {**_eth(ethnicity), **_smoking(smoking), "age": a, "nzdep": _nzdep(nzdep), "diabetes": as_bool(diabetes),
         "family_history": as_bool(family_history), "lipid_meds": as_bool(lipid_meds), "bp_meds": as_bool(bp_meds),
         "cancer": as_bool(cancer), "gi_bleed": as_bool(gi_bleed), "peptic_ulcer": as_bool(peptic_ulcer),
         "alcohol": as_bool(alcohol), "liver": as_bool(liver), "peptic_ulcer_meds": as_bool(peptic_ulcer_meds),
         "nsaid": as_bool(nsaid), "steroids": as_bool(steroids), "ssri": as_bool(ssri), "sbp": _pos(sbp),
         "tc_hdl": _pos(tc_hdl)}
    return _risk(C.BLEED, sex, x)


def _post(model, age, sex, ethnicity, nzdep, smoker, diabetes, af, heart_failure, bp_meds, lipid_meds,
          antithrombotic, sbp, tc_hdl, bmi, creatinine, hba1c, days, extra):
    a = as_float(age)
    sx = as_sex(sex).astype(float)
    x = {"male": np.where(sx >= 0, sx, np.nan),
         **_bands(np.where(a >= 18, a, np.nan), [50, 60, 70], [None, "age_50_59", "age_60_69", "age_ge_70"]),
         **_eth(ethnicity), "smoker": as_bool(smoker), "nzdep": _nzdep(nzdep), "diabetes": as_bool(diabetes),
         "af": as_bool(af), "hf": as_bool(heart_failure), "bp_meds": as_bool(bp_meds),
         "lipid_meds": as_bool(lipid_meds), "antithrombotic": as_bool(antithrombotic),
         **_bands(_pos(sbp), [100, 120, 140, 160], ["sbp_lt_100", None, "sbp_120_140", "sbp_140_160", "sbp_ge_160"]),
         "tc_hdl": _pos(tc_hdl),
         **_bands(bmi, [20, 25, 30, 35, 40], ["bmi_lt_20", "bmi_20_25", None, "bmi_30_35", "bmi_35_40", "bmi_ge_40"],
                  missing="bmi_missing"),
         **_bands(creatinine, [100, 150], [None, "creat_100_150", "creat_ge_150"], missing="creat_missing"),
         **_bands(hba1c, [40, 65], [None, "hba1c_40_65", "hba1c_ge_65"], missing="hba1c_missing"),
         # unknown time since the event counts as >= 5 years (as in R)
         **_bands(np.where(np.isnan(as_float(days)), np.inf, as_float(days)), [182, 366, 1826],
                  ["event_lt_6m", "event_6_12m", None, "event_ge_5y"]),
         **extra}
    lp = sum(b * x[k] for k, b in model["beta"]["all"].items())
    return 1 - model["s0"]["all"] ** np.exp(lp - model["mean_lp"]["all"])


def predict_post_cvd(age, sex, ethnicity, nzdep, smoker, diabetes, af, heart_failure, other_vd, bp_meds, lipid_meds,
                     antithrombotic, sbp, tc_hdl, bmi, creatinine, hba1c, days_since_event) -> np.ndarray:
    """PREDICT secondary prevention, prior CVD (2019 update of Poppe 2017). creatinine umol/L, hba1c
    mmol/mol; bmi/creatinine/hba1c/days NaN = not recorded. other_vd (angina, PVD or non-hospitalised
    cerebrovascular disease only): time since event is then scored as >= 5 years."""
    ov = as_bool(other_vd)
    d = np.where(ov == 1, np.inf, as_float(days_since_event))
    d = np.where(np.isnan(ov), np.nan, d)
    r = _post(C.POST_CVD, age, sex, ethnicity, nzdep, smoker, diabetes, af, heart_failure, bp_meds, lipid_meds,
              antithrombotic, sbp, tc_hdl, bmi, creatinine, hba1c, d, {})
    return np.where(np.isnan(ov), np.nan, r)


def predict_post_acs(age, sex, ethnicity, nzdep, smoker, diabetes, af, heart_failure, bp_meds, lipid_meds,
                     antithrombotic, sbp, tc_hdl, bmi, creatinine, hba1c, days_since_acs, acs_type) -> np.ndarray:
    """PREDICT after acute coronary syndrome (Poppe 2019, Heart). acs_type: "unstable_angina" (reference),
    "nstemi" or "stemi"; days_since_acs NaN = unknown (scored as >= 5 years, as in R)."""
    t = as_category(acs_type, ACS_TYPE).astype(float)
    t = np.where(t >= 0, t, np.nan)
    extra = {"nstemi": np.where(np.isnan(t), np.nan, t == 1), "stemi": np.where(np.isnan(t), np.nan, t == 2)}
    return _post(C.POST_ACS, age, sex, ethnicity, nzdep, smoker, diabetes, af, heart_failure, bp_meds, lipid_meds,
                 antithrombotic, sbp, tc_hdl, bmi, creatinine, hba1c, days_since_acs, extra)
