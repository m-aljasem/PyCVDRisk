"""Reference answers from the authors' SMART-REACH calculator (source/kaasenbrood2018_calculator.xls),
recalculated by LibreOffice Calc. Run from repo root:
  SOFFICE=/path/to/soffice python reference/smart_reach/run_reference.py
Needs openpyxl (not a package dependency).

LibreOffice converts the published workbook to xlsx with every formula intact. For each patient
the engine sheet's input cells (normally filled by the front sheet's drop-down controls) are set
to the calculator's codes, LibreOffice recalculates every formula on load, and the results are
read from 'Calculator engine'!G90 (10-year risk) and G78 (life expectancy free of events).
"""

import datetime as dt
import os
import subprocess
import tempfile
from pathlib import Path

import openpyxl
import pandas as pd

HERE = Path(__file__).parent
FX = HERE / "fixtures"
SRC = HERE / "source" / "kaasenbrood2018_calculator.xls"
RECALC = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema">
<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="OOXMLRecalcMode" oor:op="fuse"><value>0</value></prop></item>
<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="ODFRecalcMode" oor:op="fuse"><value>0</value></prop></item>
</oor:items>
"""
REGION_CODE = {"north_america": 1, "western_europe": 2, "netherlands": 3}  # the calculator's list order


def soffice(args, tmp):
    prof = Path(tmp) / "profile"
    (prof / "user").mkdir(parents=True, exist_ok=True)
    (prof / "user" / "registrymodifications.xcu").write_text(RECALC)
    cmd = [os.environ["SOFFICE"], "--headless", "--invisible", "--norestore", f"-env:UserInstallation=file://{prof}", *args]
    if subprocess.run(cmd, capture_output=True).returncode == 81:  # first start: profile created, restart requested
        subprocess.run(cmd, check=True, capture_output=True)


def engine_inputs(r):
    yes = lambda v: 2 if v else 1  # noqa: E731  (drop-down index: 1 = No, 2 = Yes)
    return {"C1": 1 if r.sex == "male" else 2, "C3": int(r.age), "C4": yes(r.smoker), "C6": yes(r.diabetes),
            "C8": yes(r.cad), "C10": yes(r.cerebrovascular), "C12": yes(r.pad), "C15": yes(r.af),
            "C17": yes(r.heart_failure), "C19": float(r.sbp), "C20": float(r.total_chol),
            "C21": float(r.creatinine), "C23": REGION_CODE[r.region]}


def main():
    p = pd.read_csv(FX / "patients.csv")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        soffice(["--convert-to", "xlsx", "--outdir", str(tmp), str(SRC)], tmp)
        base = tmp / (SRC.stem + ".xlsx")
        (tmp / "in").mkdir()
        for r in p.itertuples():
            wb = openpyxl.load_workbook(base)
            ws = wb["Calculator engine"]
            for cell, v in engine_inputs(r).items():
                ws[cell] = v
            wb.save(tmp / "in" / f"p{r.id:05d}.xlsx")
        soffice(["--convert-to", "xlsx", "--outdir", str(tmp / "out"), *sorted(map(str, (tmp / "in").glob("*.xlsx")))], tmp)
        rows = []
        for r in p.itertuples():
            ws = openpyxl.load_workbook(tmp / "out" / f"p{r.id:05d}.xlsx", data_only=True)["Calculator engine"]
            rows.append({"id": r.id, "risk10": ws["G90"].value, "life_expectancy": ws["G78"].value})
    out = pd.DataFrame(rows)
    out.to_csv(FX / "reference.csv", index=False)
    ver = subprocess.run([os.environ["SOFFICE"], "--version"], capture_output=True, text=True).stdout.strip()
    (FX / "VERSION").write_text(f"reference: {SRC.name} (authors' calculator, Data S2), unmodified formulas\n"
                                f"engine: {ver}\ngenerated: {dt.date.today()}\n")


if __name__ == "__main__":
    main()
