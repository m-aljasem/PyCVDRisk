"""Check every SCORE2-CKD coefficient appears, as printed, in its own equation in the supplement.

Covers the SCORE2-OP dipstick values, which no reference implementation checks.
Run from repo root: python reference/score2_ckd/check_coefficients.py
"""

import re
import zipfile
from pathlib import Path

from cvd_risk.score2 import coefficients as C

DOCX = Path(__file__).parent / "source" / "matsushita2023_zwac176_supplementary_data.docx"


def lines():
    x = zipfile.ZipFile(DOCX).read("word/document.xml").decode()
    x = re.sub(r"</w:p>", "\n", x)
    t = re.sub(r"<[^>]+>", "", x).replace("–", "-").replace("0. 0109", "0.0109")
    return [line.strip() for line in t.split("\n")]


def equation(ls, start):
    return next(line for line in ls if line.startswith(start))


def check(line, values):
    nums = [float(n) for n in re.findall(r"\d+\.\d{4}", line)]
    missing = [v for v in values if abs(v) not in nums]
    assert not missing, (line[:60], missing)


def main():
    ls = lines()
    check(equation(ls, "exeGFR ="), C.CKD_EXPECTED_EGFR.values())
    check(equation(ls, "exACR ="), C.CKD_EXPECTED_LOG8_ACR.values())
    for model, start in (("score2", "New risk with eGFR add-on SCORE2 ="),
                         ("score2_op", "New risk with eGFR add-on SCORE2-OP =")):
        check(equation(ls, start), [v for v in C.CKD_EGFR_ADDON[model].values() if v])
    check(equation(ls, "New risk with CKDACR add-on SCORE2 ="), [C.CKD_ACR_ADDON["score2"]])
    check(equation(ls, "New risk with CKDACR add-on SCORE2-OP ="), [C.CKD_ACR_ADDON["score2_op"]])
    check(equation(ls, "New risk with CKDdip add-on SCORE2 ="), [v for v in C.CKD_DIPSTICK_ADDON["score2"] if v])
    check(equation(ls, "New risk with CKDdip add-on SCORE2-OP ="), [v for v in C.CKD_DIPSTICK_ADDON["score2_op"] if v])
    print("all SCORE2-CKD coefficients found in their source equations")


if __name__ == "__main__":
    main()
