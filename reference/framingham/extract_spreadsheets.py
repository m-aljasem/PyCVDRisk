"""Export the worked examples cached in the Framingham Heart Study's official calculator
spreadsheets (source/gencardio_{lipids,bmi}.xls, framinghamheartstudy.org, downloaded
2026-11-01) to fixtures/spreadsheet_examples.csv. Needs xlrd (not a package dependency):
  pip install --target /tmp/xlrd xlrd==2.0.1 && PYTHONPATH=/tmp/xlrd python reference/framingham/extract_spreadsheets.py

Each spreadsheet's Sheet2 holds one example patient, scored four ways (male/female x
untreated/treated SBP), with the computed sum of beta*x and the 10-year risk.
"""

from pathlib import Path

import xlrd

HERE = Path(__file__).parent
COLS = {3: ("male", 0), 4: ("male", 1), 5: ("female", 0), 6: ("female", 1)}


def main():
    out = ["model,sex,bp_treated,age,sbp,total_chol_mgdl,hdl_chol_mgdl,bmi,smoker,diabetes,sum_beta_x,risk"]
    for model, name in (("lab", "gencardio_lipids.xls"), ("bmi", "gencardio_bmi.xls")):
        sh = xlrd.open_workbook(HERE / "source" / name).sheet_by_name("Sheet2")
        cells = {str(sh.cell_value(r, 0)).strip(): sh.cell_value(r, 1) for r in range(sh.nrows)}
        rows = {str(sh.cell_value(r, 2)).strip(): r for r in range(sh.nrows)}
        yn = lambda k: int(cells[k] == "y")  # noqa: E731
        for c, (sex, trt) in COLS.items():
            out.append(",".join(map(str, [
                model, sex, trt, cells["AGE"], cells["SBP"], cells.get("TCL", ""), cells.get("HDL", ""),
                cells.get("BMI", ""), yn("SMOKE"), yn("DIAB"),
                repr(sh.cell_value(rows["Σβx"], c)), repr(sh.cell_value(rows["Risk Score"], c))])))
    (HERE / "fixtures" / "spreadsheet_examples.csv").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
