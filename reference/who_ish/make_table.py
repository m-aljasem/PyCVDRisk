"""Write src/cvd_risk/who_ish/charts.csv: whoishRisk's table, with every cell the guideline
prints (Annexes 3 and 4) taken from the guideline. Run from repo root:
python reference/who_ish/make_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import charts

OUT = HERE.parents[1] / "src" / "cvd_risk" / "who_ish" / "charts.csv"


def main():
    table = charts.whoishrisk()
    printed, worst = charts.official()
    assert worst < 40, worst
    table.update(printed)
    regions = sorted({k[0] for k in table})
    keys = sorted({k[1:] for k in table})
    assert len(regions) == 14 and len(keys) == 768 and len(table) == 14 * 768
    rows = ["diabetes,male,smoker,age,sbp,chol," + ",".join(regions)]
    rows += [",".join(map(str, k)) + "," + ",".join(str(table[(r,) + k]) for r in regions) for k in keys]
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}: {len(keys)} rows x {len(regions)} subregions")


if __name__ == "__main__":
    main()
