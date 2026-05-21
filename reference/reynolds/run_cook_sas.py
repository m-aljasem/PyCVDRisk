"""Run the Reynolds blocks of Nancy Cook's risk_score_code.sas (source/cook_sas/) on the papers' worked
examples and a seeded set, and compare with this package. Run from repo root:
PYTHONPATH=src python3 reference/reynolds/run_cook_sas.py

The SAS file is not executed by SAS: its Reynolds statements (plain `name=expression;` arithmetic) are cut out
of the file and evaluated statement by statement with only + - * / ** exp() and numbers/variable names allowed.
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk import reynolds
from cvd_risk.prevent.prevent import MG_PER_MMOL

HERE = Path(__file__).parent
SAS = (HERE / "source/cook_sas/risk_score_code.sas").read_text()
N, SEED = 2000, 2007


def blocks():
    """{'male': [statements], 'female': [statements]} from the two Reynolds 'if sex=..' blocks."""
    rrs = SAS[SAS.index("** Reynolds Risk Score;"):]
    out = {}
    for sex, body in re.findall(r"if sex=([12]) then do;(.*?)\nend;", rrs, re.S):
        stmts = [re.sub(r"\s+", "", s) for s in body.split(";") if s.strip()]
        assert all(re.fullmatch(r"[a-z_0-9]+=[a-z_0-9.+\-*/()]+", s) for s in stmts), stmts
        out["male" if sex == "1" else "female"] = stmts
    assert set(out) == {"male", "female"}
    return out


BLOCKS = blocks()


def sas_risk(sex, env):
    env = dict(env)
    for s in BLOCKS[sex]:
        name, expr = s.split("=", 1)
        env[name] = eval(expr, {"__builtins__": {}, "exp": math.exp}, env)  # whitelisted arithmetic only
    return env["pctrrs"]


def sas_env(age, sbp, tc, hdl, crp, smoker, fh, a1c_if_dm):
    # the women's block names smoking and parental history currsmk / famhxmi, unlike the header (currsmoke / parhxmi)
    return dict(age=age, lnage=math.log(age), lnsbp=math.log(sbp), lnchol=math.log(tc), lnhdl=math.log(hdl),
                lncrp=math.log(crp), currsmoke=smoker, currsmk=smoker, parhxmi=fh, famhxmi=fh, diaba1c=a1c_if_dm)


def ours(sex, age, sbp, tc, hdl, crp, smoker, fh, dm, a1c):
    return 100 * reynolds(age, sex, sbp, tc / MG_PER_MMOL, hdl / MG_PER_MMOL, crp, smoker, fh, dm, a1c)


TABLE6 = [(0.1, 4.9, 7.5), (0.5, 6.5, 9.9), (1.0, 7.4, 11.2), (3.0, 8.9, 13.4), (5.0, 9.7, 14.6),
          (8.0, 10.5, 15.8), (10.0, 10.9, 16.4), (20.0, 12.3, 18.4)]


def main():
    ex = [("female", 50, 155, 240, 35, c, 1, fh, p) for c, p0, p1 in TABLE6 for fh, p in ((0, p0), (1, p1))]
    ex.append(("male", 65, 130, 205, 45, 4.0, 0, 1, 20.4))
    rows = []
    for sex, age, sbp, tc, hdl, crp, sm, fh, printed in ex:
        s = sas_risk(sex, sas_env(age, sbp, tc, hdl, crp, sm, fh, 0))
        o = float(ours(sex, age, sbp, tc, hdl, crp, sm, fh, 0, np.nan))
        rows.append(dict(sex=sex, hscrp=crp, parental_history=fh, printed=printed, sas=round(s, 4), ours=round(o, 4)))
    w = pd.DataFrame(rows)

    rng = np.random.default_rng(SEED)
    p = pd.DataFrame(dict(sex=rng.choice(["male", "female"], N), age=rng.integers(45, 81, N),
                          sbp=rng.integers(90, 191, N), tc=rng.integers(120, 321, N), hdl=rng.integers(25, 101, N),
                          crp=np.round(np.exp(rng.uniform(np.log(0.1), np.log(20), N)), 2),
                          smoker=rng.integers(0, 2, N), fh=rng.integers(0, 2, N)))
    p["dm"] = np.where(p.sex == "female", (rng.random(N) < 0.15).astype(int), 0)
    p["a1c"] = np.where(p.dm == 1, np.round(rng.uniform(5.5, 10, N), 1), np.nan)
    p["sas"] = [sas_risk(r.sex, sas_env(r.age, r.sbp, r.tc, r.hdl, r.crp, r.smoker, r.fh, r.a1c if r.dm else 0))
                for r in p.itertuples()]
    p["ours"] = ours(p.sex, p.age, p.sbp, p.tc, p.hdl, p.crp, p.smoker, p.fh, p.dm, p.a1c)
    # women, Cook's statements with S0 replaced by the printed 0.98634 (see REPORT: S0 discrepancy)
    s0_sas = float(re.search(r"bsurv_rrs=(0\.\d+);\s*\nbxave_rrs=22", SAS).group(1))
    BLOCKS["female_printed_s0"] = [st.replace(f"bsurv_rrs={s0_sas}", "bsurv_rrs=0.98634") for st in BLOCKS["female"]]
    w["sas_printed_s0"] = [round(sas_risk("female_printed_s0", sas_env(50, 155, 240, 35, c, 1, fh, 0)), 4)
                           if sx == "female" else np.nan for sx, c, fh in zip(w.sex, w.hscrp, w.parental_history)]
    w.to_csv(HERE / "fixtures/cook_sas_worked_examples.csv", index=False)
    q = p[p.sex == "female"]
    p.loc[q.index, "sas_printed_s0"] = [
        sas_risk("female_printed_s0", sas_env(r.age, r.sbp, r.tc, r.hdl, r.crp, r.smoker, r.fh, r.a1c if r.dm else 0))
        for r in q.itertuples()]
    p.round(6).to_csv(HERE / "fixtures/cook_sas_seeded.csv", index=False)

    m, f_ = p[p.sex == "male"], p[p.sex == "female"]
    fw, mw = w[w.sex == "female"], w[w.sex == "male"]
    st = dict(
        n_m=len(m), m_max=(m.ours - m.sas).abs().max(), m_eq=int((m.ours.round(1) == m.sas.round(1)).sum()),
        n_f=len(f_), f_rel_lo=(f_.ours / f_.sas - 1).min(), f_rel_hi=(f_.ours / f_.sas - 1).max(),
        f_eq=int((f_.ours.round(1) == f_.sas.round(1)).sum()), f_max=(f_.ours - f_.sas).abs().max(),
        f2_max=(f_.ours - f_.sas_printed_s0).abs().max(),
        f2_rel=(f_.ours / f_.sas_printed_s0 - 1).abs().max(),
        f2_eq=int((f_.ours.round(1) == f_.sas_printed_s0.round(1)).sum()),
        t6_sas=int((fw.sas.round(1) == fw.printed).sum()), t6_sas2=int((fw.sas_printed_s0.round(1) == fw.printed).sum()),
        t6_ours=int((fw.ours.round(1) == fw.printed).sum()), men_ex=float(mw.sas.iloc[0]), s0_sas=s0_sas)
    assert st["m_max"] < 1e-9 and st["t6_sas2"] == 16 and st["f2_rel"] < 0.01
    L = ["## 3. Developers' code: Nancy Cook's SAS program", "",
         "`source/cook_sas/risk_score_code.sas` (\"SAS code for computing AHA/ACC, ATP III, and Reynolds risk",
         "scores\", Nancy Cook, BWH, 25 Aug 2014; Cook is the statistician and senior author of both papers),",
         "linked from https://ncook.bwh.harvard.edu/sas-macros.html (\"provided as is\"). ",
         "sha256 bc0673afc94017adb41e4eea22319103c049bb79771956e9a16f48740cd69bf4. `run_cook_sas.py` cuts the two",
         "Reynolds blocks out of the file and evaluates their statements unchanged (arithmetic only; no SAS needed)",
         f"on the 17 worked examples and {N:,} seeded patients (seed {SEED}; age 45-80, SBP 90-190, TC 120-320,",
         "HDL 25-100 mg/dL, hsCRP 0.1-20 mg/L log-uniform, 15% of women diabetic with HbA1c 5.5-10%).", "",
         f"**Men.** The SAS block has the printed coefficients, S0 and mean. {st['n_m']}/{st['n_m']} seeded men agree",
         f"(max difference {st['m_max']:.1e} percentage points); the Clinical Example gives {st['men_ex']:.4f}% in both. Level A.", "",
         "**Women.** The SAS block has unrounded coefficients (0.07988 age, 3.13693 ln SBP, 0.17989 ln hsCRP,",
         "1.38163 ln TC, -1.17186 ln HDL, 0.13360 HbA1c if diabetic, 0.81795 smoking, 0.43770 parental history;",
         "mean 22.324591018), each of which rounds to the printed value, but a baseline survival of",
         f"**{s0_sas}** where the paper prints **0.98634**. With the file as published, risks are",
         f"{100*st['f_rel_lo']:.1f}-{100*st['f_rel_hi']:.1f}% lower than this package (max {st['f_max']:.2f} points;",
         f"{st['f_eq']}/{st['n_f']} equal to one decimal) and {st['t6_sas']}/16 Table 6 values are reproduced.",
         f"With S0 replaced by the printed 0.98634, the SAS statements reproduce {st['t6_sas2']}/16 Table 6 values exactly",
         f"(this package, with the printed rounded coefficients: {st['t6_ours']}/16), and the seeded women agree with",
         f"this package within {100*st['f2_rel']:.2f}% relative (max {st['f2_max']:.3f} points; {st['f2_eq']}/{st['n_f']} equal to one",
         "decimal): the residual is the rounding of the printed coefficients.", "",
         f"Classification: the S0 of {s0_sas} in the SAS file disagrees with the paper's printed S0 and with the",
         "paper's own Table 6, which needs S0 between 0.98625 and 0.98640 with these coefficients; the paper is",
         "self-consistent, so this package keeps 0.98634. The file's value may belong to another model or",
         "time point (unknown). The women's block also names smoking and parental",
         "history `currsmk` and `famhxmi`, while the header documents `currsmoke` and `parhxmi` (would give",
         "missing risks in SAS unless the data set uses those names); not a numeric issue.", ""]
    print("\n".join(L))
    return L

if __name__ == "__main__":
    main()
