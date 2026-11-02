"""Build PREVENT coefficients and the authors' worked examples from the official supplement.

Source: Khan SS et al. Circulation 2024;149:430-449, doi:10.1161/CIRCULATIONAHA.123.067626,
Supplemental Tables S12.A-J (revised supplement, 2024-01-26), files in source/.

Writes
  src/cvd_risk/prevent/coefficients.csv       one row per (model, years, term)
  reference/prevent/fixtures/worked_examples.csv   the authors' example inputs and risks
and checks every coefficient equals the original (2023) supplement's S12 tables.

Run from repo root: python reference/prevent/build_coefficients.py
Needs the sheets dumped to TSV first:
  python reference/prevent/read_xlsx.py reference/prevent/source/khan2024_supplemental_tables_rev20240126.xlsx <dir>/rev
  python reference/prevent/read_xlsx.py reference/prevent/source/khan2024_supplemental_tables.xlsx <dir>/orig
then: python reference/prevent/build_coefficients.py <dir>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
SHEETS = range(16, 26)  # S12.A-J
MODELS = ("base", "uacr", "hba1c", "sdi", "full")
OUTCOMES = ("total_cvd", "ascvd", "heart_failure", "chd", "stroke")
COLS = [f"{o}_{s}" for o in OUTCOMES for s in ("female", "male")]  # source order: Women, Men


def term(label: str) -> str:
    """Map a source row label (either supplement version) to the preventr term name."""
    l = " ".join(label.lower().replace("≥", ">=").split())
    for prefix in ("age per 10yr * ", "age per 10 yr * "):
        if l.startswith(prefix):
            return "age_" + term(l[len(prefix):])
    rules = [
        ("squared", "age_squared"), ("treated sbp", "bp_tx_sbp_gte_110"),
        ("treated non-hdl", "statin_non_hdl_c"), ("non-hdl", "non_hdl_c"), ("hdl", "hdl_c"),
        ("sbp <110", "sbp_lt_110"), ("sbp >=110", "sbp_gte_110"), ("diabetes", "dm"),
        ("smoking", "smoking"), ("bmi <30", "bmi_lt_30"), ("bmi 30+", "bmi_gte_30"),
        ("egfr <60", "egfr_lt_60"), ("egfr 60+", "egfr_gte_60"), ("anti-hypertensive", "bp_tx"),
        ("statin", "statin"), ("4-6", "sdi_4_to_6"), ("7-10", "sdi_7_to_10"),
        ("missing sdi", "missing_sdi"), ("missing uacr", "missing_uacr"),
        ("missing acr", "missing_uacr"), ("ln-uacr", "ln_uacr"), ("ln-acr", "ln_uacr"),
        ("hba1c in dm", "hba1c_dm"), ("hba1c no dm", "hba1c_no_dm"),
        ("missing hba1c", "missing_hba1c"), ("constant", "constant"), ("age", "age"),
    ]
    for key, name in rules:
        if key in l:
            return name
    raise ValueError(f"unknown label {label!r}")


def rows(path: Path) -> list[list[str]]:
    return [line.rstrip("\n").split("\t") for line in path.open(encoding="utf-8")]


def num(s: str) -> str:
    """Shortest repr of the stored double (the value as typed by the authors); '' if blank."""
    return repr(float(s)) if s.strip() else ""


def table(r: list[list[str]], start: int, first_col: int) -> dict[str, list[str]]:
    out = {}
    for row in r[start:]:
        if not row[0].strip() or row[0].startswith(("*", "R-square")):
            break
        vals = [num(v) for v in (row + [""] * 12)[first_col:first_col + 10]]
        name = term(row[0])
        assert name not in out, (name, row[0])
        out[name] = vals
    return out


def model_years(title: str) -> tuple[str, int]:
    years = 10 if "10 year" in title else 30
    t = title.lower()
    model = ("full" if "all novel" in t else "uacr" if "acr" in t else "hba1c" if "hba1c" in t
             else "sdi" if "sdi" in t else "base")
    return model, years


# Erratum, 2026-10-14 (doi:10.1161/CIR.0000000000001491): HbA1c coefficients corrected for the
# 10- and 30-year HbA1c and full models. Applied as a swap of the "HbA1c in DM" and "HbA1c no DM"
# rows. The erratum (source/erratum_CIR.0000000000001491.txt) says the two row headers were
# inverted in exactly the tabs S12C/S12E (HbA1c, full; 10 y) and S12H/S12J (30 y). preventr
# 0.12.2 differs from S12 in exactly these values and nothing else.
# The worked examples in S12 predate the erratum and are kept as published.
ERRATUM_MODELS = {"hba1c", "full"}
ERRATUM_SWAP = {"hba1c_dm": "hba1c_no_dm", "hba1c_no_dm": "hba1c_dm"}

INPUTS = [("age", "age"), ("total cholesterol", "total_chol_mgdl"), ("high-density", "hdl_chol_mgdl"),
          ("systolic", "sbp"), ("diabetes", "diabetes"), ("smoking", "smoker"), ("body-mass", "bmi"),
          ("egfr", "egfr"), ("anti-hypertension", "bp_treated"), ("statin", "statin"),
          ("uacr", "uacr"), ("hba1c", "hba1c"), ("sdi", "sdi")]


def main(dump: Path):
    coef_rows, examples = [], []
    for i in SHEETS:
        rev, orig = rows(dump / "rev" / f"{i:02d}.tsv"), rows(dump / "orig" / f"{i:02d}.tsv")
        model, years = model_years(rev[0][0])
        assert model_years(orig[0][0]) == (model, years), (i, orig[0][0])

        start = next(k for k, r in enumerate(rev) if r[0].startswith("Risk Factors")) + 2
        new = table(rev, start, 2)
        old = table(orig, 3, 1)
        assert new == old, f"S12 sheet {i}: revised and original coefficients differ"
        for name, vals in new.items():
            if model in ERRATUM_MODELS and name in ERRATUM_SWAP:
                name = ERRATUM_SWAP[name]
            coef_rows.append([model, years, name, *vals])

        ex = {"model": model, "years": years}
        k = next(k for k, r in enumerate(rev) if r[0].startswith("Input variables")) + 1
        while rev[k][0].strip():
            field = next(f for key, f in INPUTS if rev[k][0].lower().startswith(key) or key in rev[k][0].lower())
            ex[field] = rev[k][2]
            k += 1
        risk = next(r for r in rev if r[0].strip() == "Risk")
        ex.update(zip(COLS, (num(v) for v in risk[2:12])))
        examples.append(ex)

    out = ROOT / "src" / "cvd_risk" / "prevent" / "coefficients.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "years", "term", *COLS])
        w.writerows(coef_rows)

    fields = ["model", "years", *[f for _, f in INPUTS], *COLS]
    with (ROOT / "reference" / "prevent" / "fixtures" / "worked_examples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields, lineterminator="\n")
        w.writeheader()
        w.writerows(examples)
    print(f"{len(coef_rows)} coefficient rows (HbA1c erratum applied), {len(examples)} worked examples; "
          "revised == original supplement for all 10 tables")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
