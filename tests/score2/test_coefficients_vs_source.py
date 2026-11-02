import importlib.util
from pathlib import Path


def test_every_score2_family_coefficient_matches_the_supplement_text():
    p = Path(__file__).parents[2] / "reference" / "score2" / "check_coefficients.py"
    spec = importlib.util.spec_from_file_location("check_score2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.main() == 0
