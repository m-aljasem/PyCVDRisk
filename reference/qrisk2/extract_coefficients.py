"""Generate src/cvd_risk/qrisk2/coefficients.py from ClinRisk's QRISK2-2014 C source, keeping
every literal exactly as printed (same parser as QRISK3). Run from repo root:
python reference/qrisk2/extract_coefficients.py (tests/qrisk2 checks the committed file matches).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
OUT = ROOT / "src" / "cvd_risk" / "qrisk2" / "coefficients.py"
_spec = importlib.util.spec_from_file_location("q3x", ROOT / "reference" / "qrisk3" / "extract_coefficients.py")
q3x = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q3x)


def models() -> dict:
    return {"female": q3x.parse((HERE / "source" / "c" / "Q80_model_4_0.c").read_text()),
            "male": q3x.parse((HERE / "source" / "c" / "Q80_model_4_1.c").read_text())}


def render(ms: dict) -> str:
    return q3x.render(ms, name="QRISK2-2014", pkg="qrisk2", src="c/Q80_model_4_{0,1}.c", year=2013)


if __name__ == "__main__":
    OUT.write_text(render(models()))
    print(f"wrote {OUT}")
