"""Build reference/regicor/REPORT.md. Run from repo root: python reference/regicor/report.py

Sources: Wilson et al., Circulation 1998 (source/wilson1998.pdf) and Marrugat et al., Rev Esp Cardiol
2003 (source/marrugat2003_rev_esp_cardiol.pdf).
1. Wilson Table 6 is rebuilt from the package's coefficients and found verbatim in the PDF text.
2. Rev Esp Cardiol Table 1 (coefficients to 4 decimals, Girona prevalences) row by row; S0 and the mean.
3. Every band gives its coefficient at both ends (inputs in mg/dL, converted).
4. Both worked examples (Wilson: man 55, Framingham; Rev Esp Cardiol: man 56, Girona).
5. The official Framingham-REGICOR charts (chart_check.py).
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pypdf

from cvd_risk import regicor
from cvd_risk.prevent.prevent import MG_PER_MMOL as K
from cvd_risk.regicor import mean_lp
from cvd_risk.regicor.coefficients import BETA, GIRONA, S0

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import chart_check  # noqa: E402

logging.getLogger("pypdf").setLevel(logging.ERROR)
SEXES = ("male", "female")
EDGES = {"tc": [(100, 159.99), (160, 199.99), (200, 239.99), (240, 279.99), (280, 400)],
         "hdl": [(20, 34.99), (35, 44.99), (45, 49.99), (50, 59.99), (60, 100)],
         "sbp": [(100, 119.99), (120, 129.99), (130, 139.99), (140, 159.99), (160, 220)],
         "dbp": [(60, 79.99), (80, 84.99), (85, 89.99), (90, 99.99), (100, 120)]}
BASE = {"age": 50, "total_chol": 180 / K, "hdl_chol": 55 / K, "sbp": 125, "dbp": 82, "diabetes": 0, "smoker": 0}
W_LABELS = {"tc": ("TC, mg/dL ,160", "160–199", "200–239", "240–279", "$280"),
            "hdl": ("HDL-C, mg/dL ,35", "35–44", "45–49", "50–59", "$60"),
            "bp": ("Blood pressure Optimal", "Normal", "High normal", "Stage I hypertension", "Stage II–IV hypertension")}
R_LABELS = {"tc": ("< 160", "160-199", "200-239", "240-279", "≥ 280"),
            "hdl": ("< 35", "35-44", "45-49", "50-59", "> 59"),
            "bp": ("Óptima (PAS < 120)/(PAD < 80)", "Normal (PAS 120-129)/(PAD 80-84)",
                   "Normal alta (PAS 130-139)/(PAD 85-89)", "Grado I (PAS 140-159)/(PAD 90-99)",
                   "Grados II-III (PAS ≥ 160)/(PAD ≥ 100)")}


def text(name, dash=False):
    t = re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / name).pages))
    return t.replace("–", "-") if dash else t


def wilson_table6():
    """Table 6 as the PDF text prints it ("2" is its minus sign), rebuilt from BETA."""
    w = lambda v: "Referent" if v == 0 else ("2" if v < 0 else "") + f"{abs(v):.5f}"
    m, f = BETA["male"], BETA["female"]
    s = (f"TABLE 6. b-Coefficients Underlying CHD Prediction Sheets Using TC Categories Variable Men Women "
         f"Age, y {w(m['age'])} {w(f['age'])} Age squared, y {w(f['age2'])}")
    for k, labels in W_LABELS.items():
        s += "".join(f" {lab} {w(m[k][i])} {w(f[k][i])}" for i, lab in enumerate(labels))
    s += f" Diabetes {w(m['diabetes'])} {w(f['diabetes'])} Smoker {w(m['smoker'])} {w(f['smoker'])}"
    return s + f" Baseline survival function at 10 years, S(t) {S0['male', 'framingham']} {S0['female', 'framingham']}"


def rev_esp_rows(t):
    """(row, found) for Rev Esp Cardiol Table 1: coefficient to 4 decimals, Framingham %, Girona % per sex."""
    c = lambda v: "0" if v == 0 else str(Decimal(repr(v)).quantize(Decimal("0.0001"), ROUND_HALF_UP)).replace(".", ",")
    p = lambda v: re.escape(f"{v:.1f}".replace(".", ",") + "%")
    fr = r"\d+,\d%"
    rows = [("age", re.escape(f"Edad (media en años) {c(BETA['male']['age'])} 48,3 {GIRONA['male']['age']:.1f} "
                              f"{c(BETA['female']['age'])} 49,6 {GIRONA['female']['age']:.1f}".replace(".", ","))),
            ("age squared", re.escape(f"Edad al cuadrado {c(BETA['female']['age2'])} 2.604,5 3.054,9"))]
    for k, labels in R_LABELS.items():
        for i, lab in enumerate(labels):
            rows.append((f"{k} {lab}", re.escape(lab) + "".join(
                f" {re.escape(c(BETA[s][k][i]))} {fr} {p(GIRONA[s][k][i])}" for s in SEXES)))
    for k, lab in (("diabetes", "Diabetes"), ("smoker", "Tabaco")):
        rows.append((k, lab + "".join(f" {re.escape(c(BETA[s][k]))} {fr} {p(GIRONA[s][k])}" for s in SEXES)))
    return [(name, bool(re.search(rx, t))) for name, rx in rows]


def lp_diff(sex, **kw):
    """log-hazard difference from the base person, recovered from two risks."""
    r0, r = (float(regicor(sex=sex, **{**BASE, **x})) for x in ({}, kw))
    return math.log(math.log(1 - r) / math.log(1 - r0))


def bands():
    rows, ok, n = [], 0, 0
    ref = {"tc": 1, "hdl": 3, "bp": 1}  # BASE categories
    for sex in SEXES:
        for name, key, other in (("tc", "total_chol", None), ("hdl", "hdl_chol", None),
                                 ("sbp", "sbp", {"dbp": 70}), ("dbp", "dbp", {"sbp": 110})):
            k = "bp" if name in ("sbp", "dbp") else name
            conv = (lambda v: v / K) if name in ("tc", "hdl") else (lambda v: v)
            for i, (lo, hi) in enumerate(EDGES[name]):
                want = BETA[sex][k][i] - BETA[sex][k][ref[k]]
                got = [lp_diff(sex, **(other or {}), **{key: conv(v)}) for v in (lo, hi)]
                good = all(abs(g - want) < 1e-9 for g in got)
                n, ok = n + 1, ok + good
                rows.append(f"| {sex} | {name} | {lo}-{hi} | {got[0]:+.5f} / {got[1]:+.5f} | {want:+.5f} |"
                            f" {'ok' if good else 'MISMATCH'} |")
    return rows, ok, n


def examples():
    wilson = float(regicor(55, "male", 250 / K, 39 / K, 146, 88, 0, 1, calibration="framingham"))
    b = BETA["male"]
    sum_bx = b["age"] * 56 + b["tc"][3] + b["hdl"][1] + b["bp"][3] + b["smoker"]
    rev = float(regicor(56, "male", 256 / K, 37 / K, 146, 92, 0, 1))
    return dict(wilson=wilson, sum_bx=sum_bx, rev=rev,
                rev_printed_g=1 - S0["male", "gerona"] ** math.exp(4.496 - 3.489),
                rev_slip=1 - math.exp(-0.050 * math.exp(0.986)))


def summary():
    tw, tr = text("wilson1998.pdf"), text("marrugat2003_rev_esp_cardiol.pdf", dash=True)
    rows = rev_esp_rows(tr)
    return dict(table6=wilson_table6() in tw, rev_rows=rows, rev_ok=sum(f for _, f in rows),
                s0=all(s in tr for s in ("–ln(0,951)".replace("–", "-"), "97,8%")),
                g=(mean_lp("male", "gerona"), mean_lp("female", "gerona")),
                g_wilson=all(s in tw for s in ("3.0975 for men, 9.92545 for women", "0.90015 for men, 0.96246 for women")),
                ex=examples(), bands=bands(), charts=chart_check.main())


def main():
    s = summary()
    ex, (brows, bok, bn), cc = s["ex"], s["bands"], s["charts"]
    miss = [n for n, f in s["rev_rows"] if not f]
    pk, cv = {sx: cc["package", sx][1] for sx in SEXES}, {sx: cc["cvrcalc", sx][1] for sx in SEXES}
    f_cells = [c for c in chart_check.cells() if c[0] == "female"]
    off = pk["female"][1]
    as_5564 = sum(int(chart_check.np.ceil(chart_check.package("female", (55 if b == "65-74" else int(b[:2])) + off,
                                                               bp, tc, 2, sm)) == v) for _, sm, b, bp, tc, v in f_cells)
    L = ["# REGICOR: verification report", "", f"Generated by `reference/regicor/report.py`.", "",
         "Model: the Framingham coronary function of Wilson et al. (Circulation 1998;97:1837, Table 6; categorical,",
         "total cholesterol) calibrated to Girona by Marrugat et al. (Rev Esp Cardiol 2003;56:253), the function of",
         "the official Framingham-REGICOR charts. It replaced the package's earlier implementation of Marrugat et al.,",
         "J Epidemiol Community Health 2003;57:634 (coefficients printed to 2 decimals), which reproduced about half",
         "of the official chart cells; its checks are in this file's git history.", "",
         "Evidence level A + C: the official charts (regicor.cat) and both papers' tables and worked examples.", "",
         "## 1. Wilson 1998, Table 6", "",
         f"Rebuilt from the package's coefficients and S(10), the table is {'found verbatim' if s['table6'] else 'NOT found'} in the PDF",
         "text (28 coefficients to 5 decimals). Wilson's Appendix prints G = 3.0975 (men) / 9.92545 (women) and",
         f"S(10) = 0.90015 / 0.96246 ({'found' if s['g_wilson'] else 'NOT found'}); calibration=\"framingham\" uses them.", "",
         "## 2. Rev Esp Cardiol 2003, Table 1 and calibration", "",
         f"{s['rev_ok']}/{len(s['rev_rows'])} rows found (Wilson's coefficients rounded to 4 decimals, Girona prevalences).",
         *[f"- Not found: {n}" for n in miss], "",
         "The one difference: men's optimal blood pressure is printed +0.0023; Wilson prints -0.00226 (Table 6 and",
         "again in the Appendix's G, \"-0.00226 x 0.20048\"). The package follows Wilson. The effect is 0.0045 on the",
         "linear predictor (0.45% relative risk) for men with optimal blood pressure.", "",
         f"S0 = 0.951 (men) and 0.978 (women; 100 - 2.2%) {'found' if s['s0'] else 'NOT found'} in the text. Sum(b*Xmean) from the Girona",
         f"prevalences: men {s['g'][0]:.4f} (printed 3.489, which the +0.0023 reading gives: 3.4889), women",
         f"{s['g'][1]:.4f} (not printed). Rev Esp Cardiol's mean squared age for Girona women, 3054.9, is consistent",
         "with the mean age (54.2^2 = 2937.6); the JECH paper printed 2931.7, below the square of the mean.", "",
         "## 3. Bands", "",
         "Each band at both ends, against a base person aged 50 (TC 180, HDL 55 mg/dL, BP 125/82). SBP bands",
         "with DBP 70, DBP bands with SBP 110.", "",
         "| Sex | Input | Band | Log-hazard change at low / high | Table 6 difference | |", "|---|---|---|---|---:|---|",
         *brows, "", f"{bok}/{bn} bands give the Table 6 coefficient at both ends.", "",
         "## 4. Worked examples", "",
         f"- Wilson (man 55, TC 250, HDL 39, BP 146/88, smoker; Framingham): printed 0.3336; package {ex['wilson']:.4f}.",
         f"- Rev Esp Cardiol Table 3 (man 56, TC 256, HDL 37, BP 146/92, smoker; Girona): sum(b*X) printed 4.4961,",
         f"  package {ex['sum_bx']:.4f}; risk printed 12.8%, package {100 * ex['rev']:.2f}% (with the printed 3.489:",
         f"  {100 * ex['rev_printed_g']:.2f}%). The text's intermediate step \"4.496 - 3.489 = 0.986\" is a slip (1.007);",
         f"  continuing with 0.986 would give {100 * ex['rev_slip']:.1f}%, so the printed 12.8% used the correct difference.", "",
         "## 5. Official Framingham-REGICOR charts", "",
         "Source: regicor.cat, \"Taules de Framingham-REGICOR calibrades per al seu us a la poblacio catalana\"",
         "(non-diabetic; `source/regicor_official_charts_catalonia.jpg`), 400 cells transcribed by eye into",
         "`fixtures/official_charts_transcribed.txt`. Rev Esp Cardiol states the inputs: HDL 47.5 mg/dL (category",
         "45-49) and risk rounded up to the next integer. The age within each band is not stated; one offset from",
         "the band start is searched for all bands.", "",
         "| Function | Sex | Exact cells | Age | Largest miss beyond the cell (points) | Per band (35-44, 45-54, 55-64, 65-74) |",
         "|---|---|---:|---|---:|---|",
         *[f"| {name} | {sx} | {r[0]}/200 | band start + {r[1]:g} | {r[2]:.3f} | {', '.join(str(int(v)) for v in r[3].values())} |"
           for name, d in (("package", pk), ("cvrcalc 3.0", cv)) for sx, r in d.items()], "",
         f"- Men: {pk['male'][0]}/200 cells; every miss is within {pk['male'][2]:.3f} points of the printed cell, i.e. a",
         "  rounding edge. Reproduced.",
         f"- Women: the 65-74 panel is identical, cell for cell, to the 55-64 panel (men's panels all differ). The paper",
         "  does not say why; women's risk under this function peaks at 63 (age-squared term), so the chart seems to",
         "  repeat the 55-64 panel.",
         f"  Scoring 65-74 at the 55-64 ages gives {as_5564}/200; every cell is within {pk['female'][2]:.2f} points.",
         "  Not reproduced cell for cell: the women's age is probably not a single offset per band.", "",
         "cvrcalc 3.0 (third-party R) uses +0.00226 for men's optimal blood pressure with a G (3.4881) that equals",
         "the package's -0.00226 sum, so it is internally inconsistent by 0.0045; on the charts it fits the same.", ""]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"table6 {s['table6']}; rev rows {s['rev_ok']}/{len(s['rev_rows'])} {miss}; bands {bok}/{bn}; "
          f"examples {ex}; charts {pk}; women as 55-64 {as_5564}")
    assert s["table6"] and s["g_wilson"] and s["s0"] and miss == ["bp Óptima (PAS < 120)/(PAD < 80)"] and bok == bn
    assert round(ex["wilson"], 4) == 0.3336 and round(ex["sum_bx"], 4) == 4.4961 and round(100 * ex["rev"], 1) in (12.8, 12.9)
    assert pk["male"][0] >= 198 and pk["male"][2] < 0.01 and pk["female"][2] < 1


if __name__ == "__main__":
    main()
