import numpy as np

from cvd_risk import grace_6month, grace_hospital


def test_hospital_worked_examples_follow_the_table():
    # Granger 2003 Fig 4 prints 196 and 103; its terms use SBP 100 -> 53 and creatinine 0.4 -> 1,
    # against the printed table (43 and 4). See reference/grace/REPORT.md.
    assert grace_hospital(2, 100, 100, 65, 1, 0, 1, 1) == 185
    assert grace_hospital(1, 80, 60, 55, 0.4, 0, 0, 0) == 106


def test_hospital_edges_and_maximum():
    np.testing.assert_array_equal(grace_hospital(1, [80, 80.1, 200], 50, 30, 4.0, 0, 0, 0),
                                  [58 + 28, 53 + 28, 0 + 28])
    assert grace_hospital(4, 250, 250, 95, 5, 1, 1, 1) == 314


def test_six_month_items_and_edges():
    base = grace_6month(20, 0, 0, 40, 250, 0, 0.2, 0, 1)
    assert base == 1
    assert grace_6month(20, 1, 1, 40, 250, 1, 0.2, 1, 0) == base + 24 + 12 + 11 + 15 + 14
    np.testing.assert_array_equal(grace_6month([39.9, 40, 90], 0, 0, [49.9, 50, 200], [79.9, 80, 200], 0, 4, 0, 1),
                                  [0 + 0 + 24 + 20, 18 + 3 + 22 + 20, 100 + 43 + 0 + 20])


def test_invalid_rows_are_nan():
    assert np.isnan(grace_hospital([5, 1, 1], [120, -1, 120], 80, 60, [1, 1, np.nan], 0, 0, 0)).all()
    assert np.isnan(grace_6month([-1, 60], 0, 0, 80, 120, 0, 1, 0, [1, 2])).all()


def _calculator():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parents[2] / "reference/grace/fixtures/calculator_tables.json").read_text())


def _ends(label):
    """Both ends of a dropdown band as the calculator labels it: '<50', '50-69', '>=200'."""
    if label.startswith("<"):
        return 0.0, float(label[1:]) - 0.01
    if label.startswith(">="):
        return float(label[2:]), float(label[2:]) * 2
    lo, hi = label.split("-")
    return float(lo), float(hi)


def test_6month_matches_developers_calculator():
    # The "At Discharge" calculator of the developers' GRACE tool (reference/grace/calculator.py):
    # every combination of the 8 x 7 x 7 x 7 bands and 5 yes/no items, at both ends of each band.
    cal = _calculator()
    lab, dd, cb = cal["labels"], cal["dropdowns"], cal["checkboxes"]
    assert cal["totals"]["5"] == ["agePoint5", "SBPPoint5", "HRPoint5", "creatPoint5", "chfPoint5", "historymiPoint5",
                                  "inhospitalpciPoint5", "STPoint5", "enzymesPoint5"]
    bands = [("age", "age2_cb", "agePoint5"), ("heart_rate", "hr2_cb", "HRPoint5"), ("sbp", "sbp2_cb", "SBPPoint5"),
             ("creatinine", "creatinine2_cb", "creatPoint5")]
    flags = [("chf_cb", "chfPoint5"), ("historymi_cb", "historymiPoint5"), ("stsegment2_cb", "STPoint5"),
             ("enzymes2_cb", "enzymesPoint5"), ("inhospitalpci_cb", "inhospitalpciPoint5")]
    sizes = [len(lab[n]) - 1 for n, _, _ in bands] + [2] * len(flags)
    idx = [g.ravel() for g in np.indices(sizes)]
    for end in (0, 1):
        x = [np.array([_ends(t)[end] for t in lab[n][1:]])[i] for (n, _, _), i in zip(bands, idx[:4])]
        tool = sum(np.array(dd[box][var][1:])[i] for (_, box, var), i in zip(bands, idx[:4]))
        tool = tool + sum(np.array(cb[box][var])[1 - i] for (box, var), i in zip(flags, idx[4:]))  # [ticked, not]
        chf, mi, st, enz, pci = idx[4:]
        got = grace_6month(x[0], chf, mi, x[1], x[2], st, x[3], enz, pci)
        assert len(got) == 87_808
        np.testing.assert_array_equal(got, tool)


def test_finding_developers_calculator_uses_a_revised_in_hospital_nomogram():
    # The tool's "At Admission" in-hospital death score is not Granger 2003 Figure 4 in any of its eight items
    # (Fox et al. 2006 describe the tool's simplified model); the 2004 Pocket PC build has the same tables.
    cal = _calculator()
    dd, cb = cal["dropdowns"], cal["checkboxes"]
    assert dd["chf_cb"]["CHFPoint1"][1:] == [0, 21, 43, 64]                     # Killip; Granger 0, 20, 39, 59
    assert dd["age_cb"]["agePoint1"][1:] == [0, 0, 18, 36, 55, 73, 91, 100]      # Granger 0, 8, 25, 41, 58, ...
    assert dd["hr_cb"]["HRPoint1"][1:] == [0, 0, 7, 13, 23, 36, 46]              # Granger 0, 3, 9, 15, 24, 38, 46
    assert dd["sbp_cb"]["SBPPoint1"][1:] == [63, 58, 47, 37, 26, 11, 0]          # Granger 58, 53, 43, 34, 24, 10, 0
    assert dd["creatinine_cb"]["creatPoint1"][1:] == [2, 5, 8, 11, 14, 23, 31]   # Granger 1, 4, 7, 10, 13, 21, 28
    assert [cb[b][v][0] for b, v in (("cardiac_cb", "arrestPoint1"), ("stsegment_cb", "STPoint1"),
                                     ("enzymes_cb", "enzymesPoint1"))] == [43, 30, 15]  # Granger 39, 28, 14
    assert grace_hospital(2, 90, 80, 45, 1.0, 1, 1, 1) == 20 + 53 + 9 + 25 + 7 + 39 + 28 + 14


def test_hospital_bands_are_the_developers_score_at_band_midpoints():
    # The developers' 2010 methods document gives Figure 4 as continuous SAS code (fixtures/umass_methods_2010.json);
    # each printed band is that score at the band's midpoint, rounded down (reference/grace/REPORT.md section 4).
    import json
    import math
    from pathlib import Path

    nomo = json.loads((Path(__file__).parents[2] / "reference" / "grace" / "fixtures" / "umass_methods_2010.json")
                      .read_text())["granger_section2_nomogram"]
    base = dict(killip=1, sbp=250, heart_rate=40, age=20, creatinine=0.2, cardiac_arrest=0, st_deviation=0,
                elevated_enzymes=0)
    edges = {"sbp": [80, 100, 120, 140, 160, 200], "heart_rate": [50, 70, 90, 110, 150, 200],
             "age": [30, 40, 50, 60, 70, 80, 90], "creatinine": [0, 0.4, 0.8, 1.2, 1.6, 2.0, 4.0]}
    for k, e in edges.items():
        xs, ys = zip(*nomo[k])
        pts = ([e[0] - 10] if k != "creatinine" else []) + [(a + b) / 2 for a, b in zip(e, e[1:])] + [e[-1] + 10]
        for x in pts:
            item = float(grace_hospital(**{**base, k: x})) - float(grace_hospital(**base)) + (k == "creatinine")
            assert item == math.floor(np.interp(x, xs, ys) + 1e-9), (k, x)
    assert [float(grace_hospital(k, 250, 40, 20, 0.2, 0, 0, 0)) - 1 for k in (1, 2, 3, 4)] == nomo["killip"]
    assert float(grace_hospital(1, 250, 40, 20, 0.2, 1, 1, 1)) - 1 == \
        nomo["cardiac_arrest"] + nomo["st_deviation"] + nomo["elevated_enzymes"]
