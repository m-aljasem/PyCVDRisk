"""Every model variant runs with its default inputs and gives a finite number."""

import math

import numpy as np
import pytest

from forms import FORMS


@pytest.mark.parametrize("slug,variant", [(s, v) for s, vs in FORMS.items() for v in vs],
                         ids=lambda x: x if isinstance(x, str) else x.label)
def test_defaults_give_a_result(slug, variant):
    args = {f.name: f.default for f in variant.fields} | variant.fixed
    value = float(np.asarray(variant.fn(**args), dtype=float).ravel()[0])
    assert math.isfinite(value)
    for f in variant.fields:
        if f.kind == "choice":
            assert f.default in f.options


def test_every_website_model_has_a_form():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).parents[1] / "website" / "data" / "models.json").read_text())
    assert {m["slug"] for m in data["models"]} == set(FORMS)


@pytest.mark.parametrize("variant", [v for vs in FORMS.values() for v in vs if v.output in ("percent", "band")],
                         ids=lambda v: v.label)
def test_percent_and_band_outputs_are_in_range(variant):
    value = float(np.asarray(variant.fn(**{f.name: f.default for f in variant.fields} | variant.fixed)).ravel()[0])
    assert 0 <= value <= 100 if variant.output == "percent" else value in range(len(variant.labels))
