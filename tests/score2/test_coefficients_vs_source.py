import importlib.util
from pathlib import Path

import pytest


@pytest.mark.skipif(not (Path(__file__).parents[2] / "reference" / "score2" / "source" / "score2_ehab309_updated_supplement.docx").exists(), reason="publisher document not in the repository (see reference/README.md)")
def test_every_score2_family_coefficient_matches_the_supplement_text():
    p = Path(__file__).parents[2] / "reference" / "score2" / "check_coefficients.py"
    spec = importlib.util.spec_from_file_location("check_score2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.main() == 0
