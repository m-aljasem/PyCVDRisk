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
