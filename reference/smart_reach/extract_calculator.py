"""Write the SMART-REACH calculator's model formulas and baseline table to
fixtures/calculator_formulas.txt, for check_coefficients.py. Run from repo root:
  pip install --target /tmp/xl xlrd==2.0.1 msoffcrypto-tool==5.4.2
  PYTHONPATH=/tmp/xl python reference/smart_reach/extract_calculator.py

The workbook is saved with Excel's default read protection (the built-in "VelvetSweatshop" key,
which Excel and LibreOffice open without asking); msoffcrypto removes it in memory, and xlrd's own
formula decompiler prints each FORMULA record.
"""

import io
import struct
from pathlib import Path

import msoffcrypto
import xlrd
from xlrd import formula as F
from xlrd.biffh import XL_EOF, XL_FORMULA

HERE = Path(__file__).parent
SRC = HERE / "source" / "kaasenbrood2018_calculator.xls"
KEEP = {"D28", "E28", "O28", "G28", "H28", "I28", "J28", "G76", "G77", "G78", "G82", "G87", "G88", "G89", "G90"}


def main():
    f = msoffcrypto.OfficeFile(SRC.open("rb"))
    f.load_key(password="VelvetSweatshop")
    buf = io.BytesIO()
    f.decrypt(buf)
    bk = xlrd.open_workbook(file_contents=buf.getvalue(), on_demand=True)
    sx = bk.sheet_names().index("Calculator engine")
    sh = bk.sheet_by_index(sx)
    out = [f"# {SRC.name}, sheet 'Calculator engine' (rows 28-73 are ages 45-90)"]
    bk._position = bk._sh_abs_posn[sx]
    while True:
        rc, _, data = bk.get_record_parts()
        if rc == XL_EOF:
            break
        if rc == XL_FORMULA:
            r, c = struct.unpack("<HH", data[0:4])
            ref = f"{xlrd.colname(c)}{r + 1}"
            if ref in KEEP:
                cce = struct.unpack("<H", data[20:22])[0]
                out.append(f"{ref} = " + F.decompile_formula(bk, data[22:], cce, F.FMLA_TYPE_CELL, browx=r, bcolx=c, blah=0))
    out.append("# age, B (1-year survival free of stroke or MI), C (1-year non-CVD survival)")
    out += [f"{sh.cell_value(r, 0):g},{sh.cell_value(r, 1)!r},{sh.cell_value(r, 2)!r}" for r in range(27, 72)]
    (HERE / "fixtures" / "calculator_formulas.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
