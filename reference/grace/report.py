"""Build reference/grace/REPORT.md. Run from repo root: python reference/grace/report.py

Each band of both nomograms is checked twice: its printed labels and points must occur in the PDF
text (so the transcription cannot drift), and the function must give those points at both ends of
the band. Then Granger's two worked examples.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
from pathlib import Path

import pypdf

from cvd_risk import grace_6month, grace_hospital

HERE = Path(__file__).parent
logging.getLogger("pypdf").setLevel(logging.ERROR)


def text(name):
    return re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / name).pages))


H_BASE = {"killip": 1, "sbp": 250, "heart_rate": 40, "age": 20, "creatinine": 0.2, "cardiac_arrest": 0,
          "st_deviation": 0, "elevated_enzymes": 0}  # 0+0+0+0+1+0+0+0
M_BASE = {"age": 20, "chf_history": 0, "mi_history": 0, "heart_rate": 40, "sbp": 250, "st_depression": 0,
          "creatinine": 0.2, "elevated_enzymes": 0, "in_hospital_pci": 1}  # 1 point
# (input, printed text in PDF order, points, (low, high) inputs inside each band, points of the base value)
HOSPITAL = [
    ("killip", "I 0 II 20 III 39 IV 59", [0, 20, 39, 59], [(1, 1), (2, 2), (3, 3), (4, 4)], 0),
    ("sbp", "≤80 58 80-99 53 100-119 43 120-139 34 140-159 24 160-199 10 ≥200 0", [58, 53, 43, 34, 24, 10, 0],
     [(0, 80), (80.1, 99.9), (100, 119.9), (120, 139.9), (140, 159.9), (160, 199.9), (200, 300)], 0),
    ("heart_rate", "≤50 0 50-69 3 70-89 9 90-109 15 110-149 24 150-199 38 ≥200 46", [0, 3, 9, 15, 24, 38, 46],
     [(0, 50), (50.1, 69.9), (70, 89.9), (90, 109.9), (110, 149.9), (150, 199.9), (200, 300)], 0),
    ("age", "≤30 0 30-39 8 40-49 25 50-59 41 60-69 58 70-79 75 80-89 91 ≥90 100", [0, 8, 25, 41, 58, 75, 91, 100],
     [(18, 30), (30.1, 39.9), (40, 49.9), (50, 59.9), (60, 69.9), (70, 79.9), (80, 89.9), (90, 110)], 0),
    ("creatinine", "0-0.39 1 0.40-0.79 4 0.80-1.19 7 1.20-1.59 10 1.60-1.99 13 2.00-3.99 21 >4.0 28",
     [1, 4, 7, 10, 13, 21, 28], [(0, 0.39), (0.4, 0.79), (0.8, 1.19), (1.2, 1.59), (1.6, 1.99), (2, 3.99), (4.01, 9)], 1),
    ("cardiac_arrest", "Cardiac Arrest at Admission 39", [0, 39], [(0, 0), (1, 1)], 0),
    ("st_deviation", "ST-Segment Deviation 28", [0, 28], [(0, 0), (1, 1)], 0),
    ("elevated_enzymes", "Elevated Cardiac Enzyme Levels 14", [0, 14], [(0, 0), (1, 1)], 0),
]
SIX_MONTH = [
    ("age", "≤29 30–39 40–49 50–59 60–69 70–79 80–89 ≥90 0 0 18 36 55 73 91 100", [0, 0, 18, 36, 55, 73, 91, 100],
     [(18, 29.9), (30, 39.9), (40, 49.9), (50, 59.9), (60, 69.9), (70, 79.9), (80, 89.9), (90, 110)], 0),
    ("chf_history", "Heart Failure 24", [0, 24], [(0, 0), (1, 1)], 0),
    ("mi_history", "Myocardial Infarction 12", [0, 12], [(0, 0), (1, 1)], 0),
    ("heart_rate", "≤49.9 50–69.9 70–89.9 90–109.9 110–149.9 150–199.9 ≥200 0 3 9 14 23 35 43",
     [0, 3, 9, 14, 23, 35, 43], [(0, 49.9), (50, 69.9), (70, 89.9), (90, 109.9), (110, 149.9), (150, 199.9), (200, 300)], 0),
    ("sbp", "≤79.9 80–99.9 100–119.9 120–139.9 140–159.9 160–199.9 ≥200 24 22 18 14 10 4 0",
     [24, 22, 18, 14, 10, 4, 0], [(0, 79.9), (80, 99.9), (100, 119.9), (120, 139.9), (140, 159.9), (160, 199.9), (200, 300)], 0),
    ("st_depression", "11ST-Segment Depression", [0, 11], [(0, 0), (1, 1)], 0),
    ("creatinine", "0–0.39 0.4–0.79 0.8–1.19 1.2–1.59 1.6–1.99 2–3.99 ≥4 1 3 5 7 9 15 20", [1, 3, 5, 7, 9, 15, 20],
     [(0, 0.39), (0.4, 0.79), (0.8, 1.19), (1.2, 1.59), (1.6, 1.99), (2, 3.99), (4, 9)], 1),
    ("elevated_enzymes", "15Elevated Cardiac Enzymes", [0, 15], [(0, 0), (1, 1)], 0),
    ("in_hospital_pci", "Coronary Invervention 14", [14, 0], [(0, 0), (1, 1)], 0),
]


def check(fn, base, table, pdf):
    ref = float(fn(**base))
    rows, ok, n, found = [], 0, 0, 0
    for name, printed, pts, ranges, base_pts in table:
        hit = printed in pdf
        found += hit
        for p, (lo, hi) in zip(pts, ranges):
            got = [float(fn(**{**base, name: v})) - ref + base_pts for v in (lo, hi)]
            n += 1
            ok += got == [p, p]
            rows.append(f"| {name} | {lo}-{hi} | {got[0]:g} / {got[1]:g} | {p} |")
        rows[-len(pts)] += f" printed: \"{printed}\" {'found' if hit else 'NOT FOUND'} |"
    return rows, ok, n, found


# The developers' methods document (UMass Center for Outcomes Research, 2010): its SAS "if a <= x < b then
# out = p0 +/- (x - a)*(slope);" lines, read back from the PDF with whitespace removed.
SEG = r"(\d*\.?\d+)<=(\w+)<(\d*\.?\d+)then(\w+)=(\d+)(?:([+-])\(\w+-[\d.]+\)\*\(([\d./]+)\))?;"
RSEG = r"(\d*\.?\d+)<=(\w+(?:\.\w+)?(?:/88\.4)?)&\2<(\d*\.?\d+)~(\d+)(?:([+-])\(\2-[\d.]+\)\*\(([\d./]+)\))?"
# Figure 4 bands [a, b) per input, and the open ends; SAS outputs of section 2 (Granger) and 8 (Fox)
EDGES = {"sbp": [80, 100, 120, 140, 160, 200], "heart_rate": [50, 70, 90, 110, 150, 200],
         "age": [30, 40, 50, 60, 70, 80, 90], "creatinine": [0, 0.4, 0.8, 1.2, 1.6, 2.0, 4.0]}
SAS = {"sbp": "sysbp2", "heart_rate": "pulse2", "age": "age2", "creatinine": "crt2"}
R_VARS = {"sbp": "systolic.bp", "heart_rate": "heart.rate", "age": "Age", "creatinine": "creat/88.4"}


def _frac(s):
    a, _, b = s.partition("/")
    return float(a) / float(b) if b else float(a)


def _segs(found):
    return [(float(a), float(b), float(p), (1 if sg == "+" else -1) * _frac(sl) if sl else 0.0)
            for a, b, p, sg, sl in found]


def umass_methods():
    """Section 4: the developers' own code for Granger 2003 Figure 4 (and, section 8, the Fox nomogram)."""
    pdf = pypdf.PdfReader(HERE / "source" / "GRACE_RiskModel_Coefficients.pdf")
    pg = lambda *ix: re.sub(r"\s+", "", "".join(pdf.pages[i].extract_text() for i in ix))  # noqa: E731
    fx = json.loads((HERE / "fixtures" / "umass_methods_2010.json").read_text())
    nomo, lg = fx["granger_section2_nomogram"], fx["granger_section1_logistic"]
    g, fox = pg(2, 3), pg(11, 12)
    out = {}
    # (1) the transcribed knots equal the PDF's SAS lines (start points and slopes)
    n_seg = n_ok = 0
    for k, var in SAS.items():
        got = _segs([(a, b, p, sg, sl) for a, v, b, o, p, sg, sl in re.findall(SEG, g) if o == var])
        kn = nomo[k]
        want = [(a, b, p, (q - p) / (b - a)) for (a, p), (b, q) in zip(kn, kn[1:])]
        if k != "creatinine":  # its first segment starts at 0, not at a constant band
            want = [(0.0, kn[0][0], kn[0][1], 0.0)] + want
        n_seg += len(want)
        n_ok += sum(len(got) == len(want) and all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(G, W))
                    for G, W in zip(got, want))
    tail = "Death_pt=killips+sysbp2+pulse2+age2+crt2+28*stchange+14*posinit+39*carrst;"
    kil = "ifkillip=1thenkillips=0;elseifkillip=2thenkillips=20;elseifkillip=3thenkillips=39;elseifkillip=4thenkillips=59;"
    out["segments"] = (n_ok, n_seg, tail in g and kil in g)
    # (2) Figure 4's band points against the developers' continuous score at each band's midpoint
    def sas(k, x):
        kn = nomo[k]
        if x <= kn[0][0]:
            return kn[0][1]
        for (a, p), (b, q) in zip(kn, kn[1:]):
            if a <= x < b:
                return p + (x - a) * (q - p) / (b - a)
        return kn[-1][1]
    n_mid, n_eq, worst = 0, 0, 0.0
    base = H_BASE
    for k, e in EDGES.items():
        pts = ([e[0] - 10] if k != "creatinine" else []) + [(a + b) / 2 for a, b in zip(e, e[1:])] + [e[-1] + 10]
        for x in pts:
            # this item's Figure 4 points: H_BASE scores 0 on every item but creatinine (1)
            fig = float(grace_hospital(**{**base, k: x})) - float(grace_hospital(**base)) + (k == "creatinine")
            v = sas(k, x)
            n_mid += 1
            n_eq += math.floor(v + 1e-9) == fig
            worst = max(worst, abs(v - fig))
    out["mid"] = (n_eq, n_mid, worst)
    # (3) the logistic model: odds ratios of the Granger abstract, and the document's worked example
    ors = [(math.exp(10 * lg["age"]), "1.7 per 10 years"), (math.exp(lg["killip"]), "OR, 2.0 per class"),
           (math.exp(-20 * lg["sbp"]), "OR, 1.4 per 20-mm Hg"), (math.exp(lg["st_deviation"]), "OR, 2.4)"),
           (math.exp(lg["cardiac_arrest"]), "OR, 4.3)"), (math.exp(lg["creatinine"]), "OR, 1.2 per 1-mg/dL"),
           (math.exp(lg["elevated_enzymes"]), "OR, 1.6)"), (math.exp(30 * lg["heart_rate"]), "OR, 1.3 per 30-beat/min")]
    gt = text("granger2003.pdf")
    out["or"] = (sum(q in gt and f"{v:.1f}" in q for v, q in ors), len(ors))
    ex = lg["example"]
    xb = lg["intercept"] + sum(lg[k] * ex[k] for k in ex if k in lg)
    out["example"] = (xb, 1 / (1 + math.exp(-xb)), ex["xb"], ex["p"])
    # (4) RiskScorescvd's GRACE() against section 8 (Fox, death from admission to 6 months)
    src = re.sub(r"\s+", "", (HERE.parent / "vendor" / "RiskScorescvd" / "R" / "05_GRACE_func.R").read_text())
    n_r = n_rok = 0
    for k, var in SAS.items():
        s8 = _segs([(a, b, p, sg, sl) for a, v, b, o, p, sg, sl in re.findall(SEG, fox) if o == var])
        rs = _segs([(a, b, p, sg, sl) for a, v, b, p, sg, sl in re.findall(RSEG, src) if v == R_VARS[k]])
        n_r += len(s8)
        n_rok += sum(all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(A, B)) for A, B in zip(s8, rs)) * (len(s8) == len(rs))
    fox_items = "deatha6_pt=killips+sysbp2+pulse2+age2+crt2+17*stchange+13*posinit+30*carrst;" in fox
    r_items = all(x in src for x in ["killip.class==2~15", "killip.class==3~29", "killip.class==4~44",
                                      "cardiac.arrest==1~30", "ecg.st.depression==1~17"]) and "~13" in src
    out["fox"] = (n_rok, n_r, fox_items and r_items)
    return out


def main():
    g, e = text("granger2003.pdf"), text("eagle2004.pdf")
    u = umass_methods()
    hrows, hok, hn, hf = check(grace_hospital, H_BASE, HOSPITAL, g)
    mrows, mok, mn, mf = check(grace_6month, M_BASE, SIX_MONTH, e)
    ex1 = float(grace_hospital(2, 100, 100, 65, 1, 0, 1, 1))
    ex2 = float(grace_hospital(1, 80, 60, 55, 0.4, 0, 0, 0))
    alt = (float(grace_hospital(2, 99.99, 100, 65, 1, 0, 1, 1)), float(grace_hospital(1, 80, 60, 55, 0.39, 0, 0, 0)))
    quotes = ["His score would be: 20 + 53 + 15 + 58 + 7 + 0 + 28 + 14 = 196", "0 + 58 + 3 + 41 + 1 = 103"]
    qf = sum(q in g for q in quotes)
    head = ["| Input | Band (low-high input) | Points at low / high | Printed |  |", "|---|---|---|---:|---|"]
    L = ["# GRACE (in-hospital and 6-month post-discharge): verification report", "",
         f"Generated by `reference/grace/report.py`.", "",
         "Evidence: the 6-month score A (developers' calculator, section 3) + C; the in-hospital score C plus",
         "the developers' own code for its weights (section 4). RiskScorescvd's `GRACE()` is not a reference for",
         "either paper: it is the developers' admission-to-6-month death nomogram (Fox 2006; section 4).", "",
         "## 1. In-hospital death (Granger 2003, Figure 4)", "", *head, *hrows, "",
         f"{hok}/{hn} bands give the printed points at both ends; {hf}/{len(HOSPITAL)} printed rows found in the PDF.",
         "", "### Worked examples", "",
         "| Example (Figure 4) | Printed | This implementation |", "|---|---:|---:|",
         f"| Killip II, SBP 100, HR 100, age 65, creatinine 1, ST deviation, enzymes | 196 | {ex1:g} |",
         f"| Killip I, SBP 80, HR 60, age 55, creatinine 0.4 | 103 | {ex2:g} |", "",
         f"The examples are quoted from the PDF ({qf}/2 found). They do not follow the printed table:",
         "example 1 gives SBP 100 53 points (table: 100-119 = 43) and its terms add to 195, not 196;",
         "example 2 gives creatinine 0.4 1 point (table: 0.40-0.79 = 4). Both examples put a value on a",
         f"band's printed lower edge into the band below. With that reading they give {alt[0]:g} and {alt[1]:g}.",
         "We follow the table, whose 0.40 is explicit.", "",
         "**Confirmed by the authors:** asked whether the table or the examples is the intended rule,",
         "the authors confirmed that the table is correct (personal communication).", "",
         "## 2. 6-month post-discharge death (Eagle 2004, Figure 2)", "", *head, *mrows, "",
         f"{mok}/{mn} bands give the printed points at both ends; {mf}/{len(SIX_MONTH)} printed rows found in the PDF.",
         "Figure 2 also prints a stray \"1\" under SBP >=200; it belongs to no row and is ignored.",
         "The paper gives no worked example.", "",
         "## 3. The developers' calculator (evidence level A for the 6-month score)", "",
         "The GRACE \"ACS Risk Model\" tool of the University of Massachusetts Center for Outcomes Research",
         "(Fox et al., BMJ 2006, Figure 2) survives as Wayback Machine captures of",
         "www.outcomes-umassmed.org/grace/acs_risk/: the Flash version (2013 capture, `source/grace_acs_risk_2013.swf`)",
         "and a Pocket PC build dated August 2004 (`acs_risk.cab`, not stored; identical tables). `calculator.py`",
         "disassembles its ActionScript (`as2.py`) into `fixtures/calculator_tables.json`: the points of every",
         "dropdown entry and checkbox, the items each total adds, the points-to-probability tables and the labels.", "",
         "- **6-month post-discharge death (\"At Discharge\")**: the tool's total is the sum of the nine Eagle",
         "  2004 items, and `grace_6month` equals it for all 87,808 combinations of the bands and yes/no items, at",
         "  both ends of every labelled band (`test_6month_matches_developers_calculator`). The labels also fix",
         "  the band edges (\"<80\", \"80-99\", ..., \">=4\"), as implemented.",
         "- **In-hospital death (\"At Admission\")**: the tool does not use Granger 2003 Figure 4. All eight items",
         "  carry other points (Killip 0/21/43/64 against 0/20/39/59; cardiac arrest 43 against 39; age 40-49",
         "  18 against 25). Fox et al. 2006 describe the tool as computing risk from their simplified model,",
         "  whose points the abridged BMJ paper does not print. The package keeps Figure 4 (the published model,",
         "  confirmed by its authors), which stays at level C; which nomogram other software calls \"the GRACE",
         "  score\" was not checked.",
         "  `test_finding_developers_calculator_uses_a_revised_in_hospital_nomogram`.",
         "- The tool's other outcomes (in-hospital death or MI, death and death or MI from admission to 6",
         "  months, death or MI after discharge) are not implemented.", "",
         "## 4. The developers' methods document (2010)", "",
         "\"Methods and formulas used to calculate the GRACE Risk Scores\" (Center for Outcomes Research, UMass",
         "Medical School, updated 21 April 2010; author O. Dabbous, co-author of Granger 2003), Wayback capture of",
         "www.outcomes-umassmed.org/GRACE/publicfiles/GRACE_RiskModel_Coefficients.pdf",
         "(`source/GRACE_RiskModel_Coefficients.pdf`). Its section 2 is SAS code \"for obtaining risk score as in",
         "article\" (Granger 2003, p. 2351): Figure 4's weights, but continuous, interpolated linearly between",
         "knots (`fixtures/umass_methods_2010.json`). The same folder holds the developers' copy of the paper",
         "(`source/Granger_GRACE_Risk_Model_Archives.pdf`, the same article as `granger2003.pdf`). Downloaded",
         "from web.archive.org (captures 20190924222907 and 20220817021049); sha256 6448320308fac0a5...",
         "(methods document) and 3e32fb858ad78759... (paper).", "",
         f"- Transcription: {u['segments'][0]}/{u['segments'][1]} SAS segments (start points and slopes) read back from the PDF",
         "  equal the fixture's knots; the Killip lines (0/20/39/59) and the total (28 ST deviation, 14 enzymes,",
         f"  39 cardiac arrest) are found verbatim: {u['segments'][2]}.",
         "- Killip class and the three yes/no items: identical to Figure 4 (7/7 non-zero points).",
         f"- Bands: at the midpoint of each Figure 4 band (open bands: 10 beyond the edge) the developers' score,",
         f"  rounded down, is Figure 4's points for {u['mid'][0]}/{u['mid'][1]} bands; the largest difference before rounding",
         f"  is {u['mid'][2]:g} (age 30-39, 50-59, 60-69, 80-89: 8.5, 41.5, 58.5, 91.5 against 8, 41, 58, 91).",
         "  Figure 4 is therefore the developers' continuous score read at band midpoints. Within a band the two",
         "  differ (e.g. SBP 80-99: 58 to 48 against 53); the package keeps the published banded table, which the",
         "  authors confirmed.",
         f"- Section 1, the logistic model behind it: its coefficients give the 8 odds ratios of Granger's abstract",
         f"  ({u['or'][0]}/{u['or'][1]} found), and its worked example XB = {u['example'][0]:.5f}, P = {u['example'][1]:.5f}",
         f"  (printed {u['example'][2]}, {u['example'][3]}). Not implemented.",
         "- Neither the in-hospital death nomogram of the archived calculator (Killip 0/21/43/64, cardiac arrest",
         "  43, ST deviation 30, enzymes 15; section 3) nor a source for it appears in this document, which lists",
         "  every nomogram the developers used (Granger, Eagle, and the Palm Pilot death/MI and admission-to-6-month",
         "  scores). Its table is reprinted in reviews (e.g. Exp Clin Cardiol 2009, PMC2722456; J Intensive Care",
         "  2018, PMC5856388, \"range 2-372\") without a derivation; its source publication was not identified.",
         "  The document's section 4 (in-hospital death or MI, Killip 0/33/67/100, arrest 98, ST 67, enzymes",
         "  54) and section 8/9 (admission to 6 months) match calculator totals 2, 3 and 4 in their Killip and",
         "  yes/no points.",
         f"- Section 8 (Fox, death from admission to 6 months, \"as used in Palm Pilot software\"): RiskScorescvd's",
         f"  `GRACE()` (v0.3.1, `reference/vendor/RiskScorescvd`) equals it segment for segment ({u['fox'][0]}/{u['fox'][1]}",
         f"  SBP, heart-rate, age and creatinine segments; Killip 0/15/29/44, arrest 30, ST 17, enzymes 13: {u['fox'][2]})."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"hospital {hok}/{hn} found {hf}/{len(HOSPITAL)}; 6-month {mok}/{mn} found {mf}/{len(SIX_MONTH)}; "
          f"examples {ex1:g} {ex2:g} quotes {qf}/2")
    assert hok == hn and mok == mn and hf == len(HOSPITAL) and mf == len(SIX_MONTH) and qf == 2
    assert (ex1, ex2) == (185, 106) and alt == (195, 103)
    print(f"umass: segments {u['segments']}, midpoints {u['mid'][:3]}, OR {u['or']}, example {u['example']}, fox {u['fox']}")
    assert u["segments"][0] == u["segments"][1] and u["segments"][2] and u["mid"][0] == u["mid"][1]
    assert u["or"][0] == u["or"][1] and abs(u["example"][0] - u["example"][2]) < 1e-4
    assert abs(u["example"][1] - u["example"][3]) < 1e-5 and u["fox"][0] == u["fox"][1] and u["fox"][2]


if __name__ == "__main__":
    main()
