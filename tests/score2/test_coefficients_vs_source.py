import importlib.util
from pathlib import Path

import pytest


@pytest.mark.skipif(not (Path(__file__).parents[2] / "reference/score2/source/score2_asia_pacific_accepted_manuscript.pdf").exists() or not (Path(__file__).parents[2] / "reference/score2/source/score2_diabetes_ehad260_supplement.pdf").exists() or not (Path(__file__).parents[2] / "reference/score2/source/score2_ehab309_updated_supplement.docx").exists() or not (Path(__file__).parents[2] / "reference/score2/source/score2_op_ehab312_supplement.docx").exists() or not (Path(__file__).parents[2] / "reference/score2/source/score2_op_supplement_accepted_manuscript.pdf").exists() or not (Path(__file__).parents[2] / "reference/score2/source/visseren2021_esc_prevention_guideline.pdf").exists(), reason="publisher document not in the repository (see reference/*/source/SOURCES.md)")
def test_every_score2_family_coefficient_matches_the_supplement_text():
    p = Path(__file__).parents[2] / "reference" / "score2" / "check_coefficients.py"
    spec = importlib.util.spec_from_file_location("check_score2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.main() == 0
