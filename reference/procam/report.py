"""Build reference/procam/REPORT.md. Run from repo root: python reference/procam/report.py

Source: Assmann et al., Eur J Clin Invest 2007 (source/assmann2007.pdf).
1. Table 2 (coronary points): each band, transcribed by eye, occurs in the PDF text and the package
   gives its points at both ends (inputs in mg/dL, converted, so the unit round trip is tested).
2. Tables 3-4 (points -> 10-year risk category by age) are parsed from the PDF and compared row by row.
3. Tables 6-7 (stroke points and risk) as in 1, and Table 7 value by value.
4. Table 6 points against the Table 5 Cox coefficients (consistency of the paper, not a pass/fail check).
5. The Assmann Foundation's online tests (fixtures/calculator_*.csv from query_calculator.py) vs the package.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import logging
import re
from pathlib import Path

import pypdf

from cvd_risk import procam, procam_stroke
from cvd_risk.prevent.prevent import MG_PER_MMOL
from cvd_risk.procam import STROKE_RISK
from cvd_risk.procam.procam import CORONARY_TABLES, TG_MG_PER_MMOL

HERE = Path(__file__).parent
PDF = HERE / "source" / "assmann2007.pdf"
logging.getLogger("pypdf").setLevel(logging.ERROR)
MAX_POINTS = 71
# all-zero person: LDL 90, HDL 60, TG 90 mg/dL, SBP 100, no smoking/diabetes/family history
BASE = dict(age=50, sex="male", ldl_chol=90, hdl_chol=60, triglycerides=90, sbp=100, smoker=0, diabetes=0,
            family_history=0)
# Table 2 as printed: (input, printed band, low, high, points); high is the band's last value in mg/dL
_LDL = [("ldl_chol", "≤ 100", 50, 100.99, 0)] + [
    ("ldl_chol", f"{lo}–{lo + 4}", lo, lo + 4.99, i + 1) for i, lo in enumerate(range(101, 192, 5))] + [
    ("ldl_chol", "≥ 196", 196, 300, 20)]
_HDL = [("hdl_chol", "≤ 35", 20, 35.99, 11)] + [
    ("hdl_chol", f"{lo}–{lo + 1}", lo, lo + 1.99, 10 - i) for i, lo in enumerate(range(36, 55, 2))] + [
    ("hdl_chol", "> 55", 56, 100, 0)]
_SBP = [("sbp", "< 110", 90, 109.99, 0)] + [
    ("sbp", f"{lo}–{lo + 9}", lo, lo + 9.99, i + 1) for i, lo in enumerate(range(110, 171, 10))] + [
    ("sbp", "≥ 180", 180, 250, 8)]
_TG = [("triglycerides", "< 100", 50, 99.99, 0), ("triglycerides", "100–149", 100, 149.99, 2),
       ("triglycerides", "150–199", 150, 199.99, 3), ("triglycerides", "≥ 200", 200, 500, 4)]
TABLE2 = _LDL + _HDL + _SBP + _TG
FLAGS = [("smoker", "Yes 12", "male", 12), ("family_history", "Yes 5", "male", 5),
         ("diabetes", "Yes (men) 9", "male", 9), ("diabetes", "Yes (women) 11", "female", 11)]
TABLE6_SBP = [("< 140 mmHg 0", 120, 139.99, 0)] + [
    (f"{lo}–{lo + 4} mmHg {i + 1}", lo, lo + 4.99, i + 1) for i, lo in enumerate(range(140, 176, 5))] + [
    ("> 180 mmHg 9", 180, 220, 9)]
TABLE5 = {"female": -0.626, "age": 0.109, "smoker": 0.852, "sbp": 0.022, "diabetes": 0.615}


def pages():
    return [p.extract_text() or "" for p in pypdf.PdfReader(PDF).pages]


def flat(t):
    return re.sub(r"\s+", " ", t)


def _mg(var, v):
    return v / MG_PER_MMOL if var in ("ldl_chol", "hdl_chol") else v / TG_MG_PER_MMOL if var == "triglycerides" else v


def table2_rows(t):
    """(input, band, points, found in PDF, package points at low, at high)."""
    out, base = [], {k: _mg(k, v) for k, v in BASE.items()}
    for var, band, lo, hi, pts in TABLE2:
        got = [float(procam(**{**base, var: _mg(var, v)})) for v in (lo, hi)]
        out.append((var, band, pts, f"{band} {pts}" in t, *got))
    for var, band, sex, pts in FLAGS:
        got = float(procam(**{**base, "sex": sex, var: 1}))
        # pypdf < 7 splits the kerned "Yes" into "Y es"; accept either extraction
        out.append((var, band, pts, band in t.replace("Y es", "Yes"), got, got))
    return out


def risk_rows(page):
    """({sex: {first age: (upper bounds, printed "<")}}, [(sex, row) not tiling]) parsed from Tables 3-4 (page 5).
    A category starting before the previous one ends (a misprint) takes the later category's start."""
    women, men = page.split("Table 3 Ten-year")
    out, slips = {}, []
    for sex, part in (("female", women), ("male", men)):
        rows = {}
        for ln in part.splitlines():
            m = re.match(r"^(\d\d)(?:–\d\d)? (≤|<) (\d+)(.*)$", ln.strip())
            if not m:
                continue
            bounds, prev = [int(m[3])], int(m[3])
            for tok in re.findall(r"(\d+)–(\d+)|=+ (\d+)|(\d+)", m[4]):
                a, b, ge, bare = tok
                start = int(a or ge or bare)
                if start != prev + 1:  # rows tile the points when "<" is read as "<="
                    slips.append((sex, ln.strip()))
                    bounds[-1] = start - 1
                if a:
                    bounds.append(int(b))
                    prev = int(b)
            if bounds[-1] == MAX_POINTS:
                bounds.pop()  # a range ending at the maximum is the open last category
            rows[int(m[1])] = (tuple(bounds), m[2] == "<")
        out[sex] = rows
    return out, slips


def table7_rows(t):
    printed = dict((int(p), float(r.replace("·", "."))) for p, r in re.findall(r"(\d\d) (\d+·\d)", t)
                   if 42 <= int(p) <= 80)
    for lo, hi, r in re.findall(r"(\d\d)–(\d\d) (\d·\d)", t):
        for p in range(int(lo), int(hi) + 1):
            printed[p] = float(r.replace("·", "."))
    return printed


def _csv(name):
    return list(csv.DictReader(ln for ln in (HERE / "fixtures" / name).open() if not ln.startswith("#")))


def tool_section():
    """Section 5: the Assmann Foundation's PROCAM-Tests (fixtures from query_calculator.py) vs the package."""
    rows = []
    for r in _csv("calculator_coronary.csv"):
        f = {k: float(v) for k, v in r.items() if v != ""}
        sx = "female" if r["gender"] == "2" else "male"
        args = (f["age"], sx, f["ldl"] / MG_PER_MMOL, f["hdl"] / MG_PER_MMOL, f["triglyceride"] / TG_MG_PER_MMOL,
                f["systrr"], f["smoking"], f["diabetes"], f["anamnesis"])
        tool = sum(f["risk"] >= b for b in (5, 10, 20, 30))  # the tool's % into CATEGORIES (0-4% means < 5%)
        is_open = f["ldl"] > 196 or f["hdl"] < 35 or f["systrr"] > 180  # inputs in an open-ended Table 2 band
        rows.append((int(procam(*args, output="category")), tool, is_open, f["risk"]))
    cnt = {}
    for name, keep in (("all", lambda o: True), ("no input in an open-ended band", lambda o: not o),
                       ("LDL > 196, HDL < 35 or SBP > 180", lambda o: o)):
        sel = [(k, t, rk) for k, t, o, rk in rows if keep(o)]
        cnt[name] = (len(sel), sum(k == t for k, t, _ in sel), sum(k < t for k, t, _ in sel),
                     sum(k > t for k, t, _ in sel), sum(abs(k - t) > 1 for k, t, _ in sel))
    near = max(min(abs(rk / b - 1) for b in (5, 10, 20, 30)) for k, t, o, rk in rows if k != t and not o)
    stroke, tbl, sex_gap = _csv("calculator_stroke.csv"), {}, {}
    for r in stroke:
        sx = "female" if r["gender"] == "2" else "male"
        p = int(procam_stroke(int(r["age"]), sx, int(r["smoking"]), int(r["diabetes"]), int(r["systrr"])))
        sex_gap.setdefault(p, set()).add(float(r["risk"]))
        tbl.setdefault(p + 2 * (sx == "female"), set()).add(float(r["risk"]))  # the tool's sex weight is 4, not 6
    s_eq = sum(STROKE_RISK.get(int(procam_stroke(int(r["age"]), "female" if r["gender"] == "2" else "male",
                                                 int(r["smoking"]), int(r["diabetes"]), int(r["systrr"])))) ==
               float(r["risk"]) for r in stroke)
    one = sum(len(v) == 1 for v in tbl.values())
    T = {p: min(v) for p, v in tbl.items()}
    probes = _csv("calculator_probes.csv")
    sbp_sc = [(int(r["value"]), float(r["risk"])) for r in probes if r["tool"] == "strokecheck" and r["input"] == "systrr"]
    ldl_hc = [(int(r["value"]), r["risk"]) for r in probes if r["tool"] == "healthcheck" and r["input"] == "ldl"]
    show = (55, 60, 65, 70, 75, 77, 78, 80)
    L = ["## 5. The Assmann Foundation's online tests", "",
         "Source: www.assmann-stiftung.de/procam-tests/ embeds `/PROCAM21/index_neu.html` (saved in `source/procam21/`,",
         "sha256 in the table below). The page computes nothing in the browser: each slider sends a",
         "GET to `/PROCAM/api/healthcheck.php` (PROCAM-Gesundheitstest, men and women 20-75, myocardial infarction in",
         "10 years) or `/PROCAM/api/strokecheck.php` (Schlaganfalltest, 35-65), and shows the JSON `risk` (coronary 2",
         "decimals, stroke 1 decimal, capped at \">= 11.5%\"). A third tab (Schnelltest: BMI instead of lipids) is not",
         "a published PROCAM 2007 score and was not used. The linked services.herzalter-bestimmen.de \"Herzalter-Test\"",
         "cites the 2007 paper and posts the same inputs to an API that stores each record (\"Datenspende\"), so it was",
         "not queried. The 2006 Wayback captures (`/assmann-stiftung/3_herz/tests/frameset-herz-t2.html`, `-t3.html`)",
         "hold only the framesets; their content frames (`praev1test2.html`, `praev1test3.html`) were never archived",
         "(404), so the 2002-era test code cannot be recovered.", "",
         f"`query_calculator.py` sent {len(rows)} seeded coronary and {len(stroke)} seeded stroke patients (seeds 2007,",
         "2008; integer inputs over the page's slider ranges), one GET per patient 1 s apart. `fixtures/calculator_probes.csv`",
         "varies one input at a time from a base patient.", "",
         "| File | sha256 |", "|---|---|",
         *[f"| `{p.relative_to(HERE)}` | `{hashlib.sha256(p.read_bytes()).hexdigest()}` |"
           for p in sorted((HERE / "source" / "procam21").iterdir()) + sorted((HERE / "fixtures").glob("calculator_*.csv"))],
         "", "### 5a. Coronary: the tool runs the continuous Weibull function", "",
         "The tool's risk moves within every Table 2 band and beyond its open ends (probes, LDL: " +
         ", ".join(f"{v} -> {r}%" for v, r in ldl_hc) + "), so it is the full Weibull function, not the",
         "point score. It is the 2007 model (men and women, 20-75; the herzalter page cites Eur J Clin Invest",
         "2007;37:925). It cannot check the points value by value; it can check that Tables 3-4 place each person in",
         "the category the function gives (the tool's % binned at 5, 10, 20, 30):", "",
         "| Patients | n | Same category | Package lower | Package higher | 2+ categories apart |", "|---|---:|---:|---:|---:|---:|",
         *[f"| {k} | {n} | {s} | {lo} | {hi} | {far} |" for k, (n, s, lo, hi, far) in cnt.items()], "",
         f"With every input inside a closed band the point score and the function agree on {cnt['no input in an open-ended band'][1]}/"
         f"{cnt['no input in an open-ended band'][0]} categories; each",
         f"disagreement is one category, in both directions, with the tool's risk within {100 * near:.0f}% (relative) of",
         "a category boundary: what rounding inputs to bands and points to categories does. With LDL > 196, HDL < 35 or",
         "SBP > 180 the point score saturates and gives the lower category far more often (all 2-apart cases are here).",
         "Classified as a property of the published point score, not an error of either. Not identity evidence: the",
         "level stays C for the point score.", "",
         "### 5b. Stroke: the tool departs from Tables 6-7", "",
         "SBP probes: " + ", ".join(f"{v} -> {r}" for v, r in sbp_sc) + ". The tool is flat below 140 and steps at",
         "Table 6's 5-mmHg cut-points, so it is a point score, not the Table 5 Cox model. But:", "",
         f"- Package = tool for {s_eq}/{len(stroke)} patients. Grouped by the Table 6 total, {sum(len(v) == 2 for v in sex_gap.values())} totals",
         "  have two tool values: women get the value men get 2 points higher. Adding 2 points to women makes the tool",
         f"  a function of the total alone ({one}/{len(tbl)} totals single-valued), i.e. the tool's male-sex weight is 4",
         "  points where Table 6 prints 6 (Table 5's β 0.626 / 0.109 per year = 5.7).",
         "- Its points-to-risk table (indexed with women +2) is within 0.1 of Table 7 up to 54 points (42: 0.1 vs 0.2),",
         "  then above it by a growing margin:", "",
         "| Points | " + " | ".join(map(str, show)) + " |", "|---|" + "---:|" * len(show),
         "| Tool, % | " + " | ".join(f"{T[p]:g}" for p in show) + " |",
         "| Table 7, % | " + " | ".join(f"{STROKE_RISK[p]:g}" for p in show) + " |", "",
         "  The tool shows \">= 11.5%\" from 78 points; Table 7 prints values up to 80 and \"> 10.0\" above.", "",
         "Classified as the tool departing from the paper (perhaps a later recalibration; nothing on the page says so).",
         "The package keeps the paper.", ""]
    assert cnt["all"][:2] == (500, 391) and cnt["no input in an open-ended band"][1:] == (167, 14, 12, 0)
    assert one == len(tbl) and s_eq == 26 and T[70] == 4.1 and T[42] == 0.1
    assert all(abs(T[p] - STROKE_RISK[p]) <= 0.1 + 1e-9 for p in range(42, 55)) and all(T[p] > STROKE_RISK[p] for p in range(55, 81))
    return L


def main():
    pg = pages()
    t2 = table2_rows(flat(pg[3]))
    t2_ok = sum(found and lo == p == hi for _, _, p, found, lo, hi in t2)
    parsed, slips = risk_rows(pg[4])
    rr = {sex: sum(parsed[sex][a][0] == CORONARY_TABLES[sex].get(a, ()) for a in parsed[sex]) for sex in parsed}
    lt = sum(lt for sex in parsed for _, lt in parsed[sex].values())
    t6 = flat(pg[5])
    s_rows = []
    for band, lo, hi, pts in TABLE6_SBP:
        got = [float(procam_stroke(35, "female", 0, 0, v)) - 35 for v in (lo, hi)]
        s_rows.append((band, pts, band.replace("Y es", "Yes") in t6, *got))
    s_ok = sum(found and a == p == b for _, p, found, a, b in s_rows)
    flags6 = [("Male 6", procam_stroke(35, "male", 0, 0, 120) - 35, 6), ("Y es 9", procam_stroke(35, "f", 1, 0, 120) - 35, 9),
              ("Yes7", procam_stroke(35, "f", 0, 1, 120) - 35, 7), ("Age* 1 per year", procam_stroke(36, "f", 0, 0, 120) - 35, 1)]
    f6_ok = sum(lbl.replace(" ", "") in t6.replace(" ", "") and float(v) == p for lbl, v, p in flags6)  # pypdf versions space "Yes" differently
    t7 = table7_rows(t6)
    t7_ok = sum(t7.get(p) == r for p, r in STROKE_RISK.items())
    ratio = {k: v / TABLE5["age"] for k, v in TABLE5.items()}
    L = ["# PROCAM: verification report", "", f"Generated by `reference/procam/report.py`.", "",
         "Model: PROCAM 2007 (Assmann, Schulte, Cullen, Seedorf; Eur J Clin Invest 2007;37:925), the Weibull coronary",
         "score for men and women aged 20-75 and the cerebral ischaemic event score (35-65). It replaced the package's",
         "2002 point score (Circulation 2002;105:310, men 35-65; its checks are in this file's git history).", "",
         "Evidence level C: the paper's printed tables (`source/assmann2007.pdf`). The paper refers the full Weibull",
         "function to an annex that is not part of the article, so the published point score is implemented. The",
         "Assmann Foundation's own online tests (section 5) run the full Weibull function for the coronary score, so",
         "they check the point score's categories only, and a stroke score that departs from Tables 6-7.", "",
         "## 1. Table 2: coronary points", "",
         "| Input | Printed band | Points | In PDF | Package at low / high end |", "|---|---|---:|---|---|",
         *[f"| {v} | {b} | {p} | {'yes' if f else 'NO'} | {lo:g} / {hi:g} |" for v, b, p, f, lo, hi in t2],
         "", f"{t2_ok}/{len(t2)} bands found in the PDF with the package giving their points at both ends.", "",
         "## 2. Tables 3-4: points to 10-year risk category", "",
         f"Parsed from the PDF: men {len(parsed['male'])} rows, women {len(parsed['female'])} rows. Equal to the package:",
         f"men {rr['male']}/{len(parsed['male'])}, women {rr['female']}/{len(parsed['female'])}.", "",
         f"Each row prints the first category as \"< n\" ({lt} rows) and the next as \"n+1-...\", so the value n itself",
         "belongs to no printed category. The package reads \"<\" as \"≤\"; under that reading every row covers 0-71",
         "points without a gap or overlap (asserted while parsing). The first row of each table does print \"≤ 71\",",
         "and Table 2 prints both \"≤\" and \"<\", so the \"<\" may be a typesetting slip; at the boundary value (one",
         "point total per row) the category is ambiguous.", "",
         f"Rows that do not tile ({len(slips)}): " + "; ".join(f"{sx} \"{r}\"" for sx, r in slips) + ". The 20-29%",
         "range overlaps the \">= 30%\" one; the package gives the overlap to \">= 30%\" (its printed start), as the",
         "neighbouring rows imply (age 45: 65-69, >= 70; age 47: 61-66, >= 67).", "",
         "## 3. Tables 6-7: stroke score", "",
         "| Printed SBP band | Points | In PDF | Package at low / high end |", "|---|---:|---|---|",
         *[f"| {b} | {p} | {'yes' if f else 'NO'} | {a:g} / {c:g} |" for b, p, f, a, c in s_rows],
         "", f"SBP {s_ok}/{len(s_rows)}; sex, smoking, diabetes and age points {f6_ok}/{len(flags6)}. Table 6 prints the",
         "top band as \"> 180\" after \"175-179\"; the package gives 180 itself 9 points.",
         f"Table 7: {t7_ok}/{len(STROKE_RISK)} point totals (42-80) equal the printed risk.", "",
         "## 4. Table 6 points vs the Table 5 Cox coefficients", "",
         "With one point per year of age (β 0.109), the other points should be about β/0.109 per unit:", "",
         f"- male sex: {-ratio['female']:.1f} (printed 6); smoking: {ratio['smoker']:.1f} (printed 9); diabetes:",
         f"  {ratio['diabetes']:.1f} (printed 7); SBP per 5 mmHg: {5 * ratio['sbp']:.1f} (printed 1).", "",
         "Smoking and diabetes get more points than their coefficients imply; the paper does not say how the",
         "points were derived. Reported, not used as a check.", "", *tool_section()]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"table2 {t2_ok}/{len(t2)}; tables3-4 men {rr['male']}/{len(parsed['male'])} women "
          f"{rr['female']}/{len(parsed['female'])}; stroke sbp {s_ok}/{len(s_rows)} flags {f6_ok}/4 table7 {t7_ok}/39")
    assert t2_ok == len(t2) and rr["male"] == len(parsed["male"]) == 52 and rr["female"] == len(parsed["female"]) == 43
    assert slips == [("female", "46 < 44 45–53 54–62 63–69 = 68")]
    assert s_ok == len(s_rows) and f6_ok == 4 and t7_ok == 39


if __name__ == "__main__":
    main()
