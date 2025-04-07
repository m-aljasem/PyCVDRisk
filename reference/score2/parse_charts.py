"""Parse the official SCORE2 / SCORE2-OP risk charts into fixtures/esc2021_charts.csv.

Source: 2021 ESC Guidelines on CVD prevention (Visseren 2021, Eur Heart J 42:3227), Figure 3,
pages 20-23 of source/visseren2021_esc_prevention_guideline.pdf (one page per risk region:
low, moderate, high, very high). Layout (checked by eye on each page): 40 rows = age bands
85-89 .. 40-44 (top to bottom; 70-89 is SCORE2-OP, 40-69 is SCORE2), each split into SBP
160-179, 140-159, 120-139, 100-119; 16 columns = women non-smoking, women smoking, men
non-smoking, men smoking, each x non-HDL cholesterol 3.0-3.9, 4.0-4.9, 5.0-5.9, 6.0-6.9 mmol/L.

The chart digits are kerned, so plain text extraction merges them ("8899" for 8 8 9 9); this
script groups glyphs by position instead, which needs pdfplumber (not a package dependency:
`pip install pdfplumber`). Second print of the SCORE2-OP part: the authors' supplement
(source/score2_op_supplement_accepted_manuscript.pdf, Supplementary Figure 8, pp. 32-35);
this script asserts both prints agree cell for cell.
Run from repo root: python reference/score2/parse_charts.py
"""

import collections
import csv
from pathlib import Path

import pdfplumber
import pypdf

HERE = Path(__file__).parent
GUIDELINE = HERE / "source" / "visseren2021_esc_prevention_guideline.pdf"
SUPPLEMENT = HERE / "source" / "score2_op_supplement_accepted_manuscript.pdf"
REGIONS = ("low", "moderate", "high", "very_high")  # guideline pages 20-23
AGES = ["85-89", "80-84", "75-79", "70-74", "65-69", "60-64", "55-59", "50-54", "45-49", "40-44"]
SBPS = ["160-179", "140-159", "120-139", "100-119"]
NONHDL = ["3.0-3.9", "4.0-4.9", "5.0-5.9", "6.0-6.9"]


def page_rows(page):
    """The 40 chart rows of one guideline page, each 16 integers (SBP label dropped)."""
    digits = [c for c in page.chars if c["text"].isdigit()]
    size = collections.Counter(round(c["size"], 1) for c in digits).most_common(1)[0][0]
    chars = sorted((c for c in digits if abs(c["size"] - size) < 0.2), key=lambda c: c["top"])
    lines = []
    for c in chars:
        if lines and abs(c["top"] - lines[-1][0]["top"]) < 3:
            lines[-1].append(c)
        else:
            lines.append([c])
    rows = []
    for line in lines:
        line.sort(key=lambda c: c["x0"])
        nums, cur = [], line[0]["text"]
        for prev, c in zip(line, line[1:]):
            if c["x0"] - prev["x1"] < size / 7.2:  # glyphs of one number touch
                cur += c["text"]
            else:
                nums.append(int(cur))
                cur = c["text"]
        nums.append(int(cur))
        if len(nums) == 18 and f"{nums[0]}-{nums[1]}" in SBPS:
            rows.append(nums[2:])
    assert len(rows) == 40, len(rows)
    return rows


def supplement_rows(page):
    """The 16 point-estimate rows of one SCORE2-OP supplement page (CI rows skipped)."""
    rows = [[int(x) for x in ln.split()] for ln in page.extract_text().split("\n")
            if len(ln.split()) == 16 and all(x.isdigit() for x in ln.split())]
    return rows[:16]


def main():
    with pdfplumber.open(GUIDELINE) as pdf:
        grids = {reg: page_rows(pdf.pages[19 + i]) for i, reg in enumerate(REGIONS)}
    supp = pypdf.PdfReader(SUPPLEMENT).pages
    for i, reg in enumerate(REGIONS):
        assert supplement_rows(supp[31 + i]) == grids[reg][:16], reg  # 1,024 SCORE2-OP cells
    out = HERE / "fixtures" / "esc2021_charts.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "variant", "sex", "smoker", "age_band", "sbp_band", "nonhdl_band", "risk_pct"])
        for reg, grid in grids.items():
            for r, row in enumerate(grid):
                age = AGES[r // 4]
                for c, v in enumerate(row):
                    w.writerow([reg, "score2_op" if int(age[:2]) >= 70 else "score2",
                                "female" if c < 8 else "male", (c // 4) % 2, age, SBPS[r % 4],
                                NONHDL[c % 4], v])
    print(f"wrote {out} (4 regions x 640 cells; SCORE2-OP part equal to the supplement's print)")


if __name__ == "__main__":
    main()
