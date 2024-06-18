"""Parse the official 2019 WHO risk charts (supplementary appendix 2) into a table.

Writes reference/who2019/fixtures/charts.csv: one row per chart cell with the
printed 10-year risk (integer %). Layout per region page (checked by eye, p.3, p.25):
rows age 70-74 .. 40-44 (top to bottom), within each SBP >=180 .. <120; columns
men non-smoker, men smoker, women non-smoker, women smoker, each x 5 bands of
total cholesterol (lab) or BMI (non-lab). Lab pages: without diabetes | with diabetes.
Run from repo root: python reference/who2019/parse_charts.py
"""

import csv
import logging
import re
from pathlib import Path

import pypdf

HERE = Path(__file__).parent
PDF = HERE / "source" / "kaptoge2019_supplementary_appendix2_charts.pdf"
AGES = ["70-74", "65-69", "60-64", "55-59", "50-54", "45-49", "40-44"]
SBPS = [">=180", "160-179", "140-159", "120-139", "<120"]
CHOL = ["<4", "4-4.9", "5-5.9", "6-6.9", ">=7"]
BMI = ["<20", "20-24", "25-29", "30-35", ">=35"]
GROUPS = [("male", 0), ("male", 1), ("female", 0), ("female", 1)]  # (sex, smoker)
INT = re.compile(r"(?<![\d.<≥-])\d{1,2}(?![\d.%-])")


def region_key(title: str) -> str:
    t = title.lower().replace("-", "_").replace(" and ", "_").replace(" ", "_")
    return t.replace("sub_saharan", "subsaharan")


def main():
    logging.disable(logging.CRITICAL)
    pages = pypdf.PdfReader(PDF).pages
    out = []
    for page_no in range(3, 46):
        if page_no == 24:
            continue
        lines = (pages[page_no - 1].extract_text() or "").split("\n")
        lab = page_no <= 23
        title = lines[0].strip()
        region = region_key(title.split(" " + title.split()[0])[0] if lab else title)
        if lab:
            half = len(title) // 2
            region = region_key(title[:half].strip())
        width = 40 if lab else 20
        rows = []
        for line in lines[1:]:
            line = re.sub(r"\S*-\S*", " ", line)  # drop age and SBP band labels
            nums = [int(x) for x in INT.findall(line)]
            if len(nums) == width:
                rows.append(nums)
        assert len(rows) == 35, (page_no, len(rows))
        for r, nums in enumerate(rows):
            age, sbp = AGES[r // 5], SBPS[r % 5]
            for panel in range(width // 20):
                for g, (sex, smoker) in enumerate(GROUPS):
                    for b in range(5):
                        out.append({
                            "version": "lab" if lab else "nonlab", "region": region,
                            "diabetes": panel if lab else "", "sex": sex, "smoker": smoker,
                            "age_band": age, "sbp_band": sbp,
                            "x_band": (CHOL if lab else BMI)[b],
                            "risk_pct": nums[panel * 20 + g * 5 + b],
                        })
    with (HERE / "fixtures" / "charts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(len(out), "chart cells,", len({(o["version"], o["region"]) for o in out}), "charts")


if __name__ == "__main__":
    main()
