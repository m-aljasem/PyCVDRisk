"""SCORE2-CKD against the developers' online calculator (CKD Prognosis Consortium, ckdpcrisk.org/ckdpatchscore2).

    python reference/score2_ckd/ckdpc_calculator.py fetch   # queries the calculator, writes fixtures/ckdpc_results.jsonl

The calculator page sends each input set to scorepatch.php and shows the JSON it returns: SCORE2 /
SCORE2-OP (r0_k) and the same with the CKD add-on (r1_k) for risk regions k = 1-4 (low, moderate,
high, very high), already rounded by the server to 0.1%. ACR "not available" (code 111) gives the
eGFR add-on; an ACR in mg/g gives the eGFR + ACR add-on. The calculator has no dipstick option.
Each patient is queried both ways. Patients: patient A of the supplement, a base patient, one-change
variants (every predictor, both sides of the eGFR knots at 60 and 90, SCORE2-OP ages), and 40
random patients, all inside the calculator's own dropdown ranges (age 40-85, SBP 90-180, total
cholesterol 3.9-7.2 and HDL 0.78-2.07 mmol/L, eGFR 15-120, ACR 5-1500 mg/g).
The comparison is tests/score2/test_score2_ckd.py::test_matches_ckdpc_online_calculator.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

FX = Path(__file__).parent / "fixtures"
URL = "https://ckdpcrisk.org/ckdpatchscore2/scorepatch.php"
REGIONS = ("low", "moderate", "high", "very_high")
NO_ACR = 111  # the page's "Not available"
A = dict(age=42, sex="male", sbp=128, total_chol=3.8, hdl_chol=1.4, smoker=1, diabetes=0, egfr=45, acr=150)
BASE = dict(age=60, sex="male", sbp=140, total_chol=5.5, hdl_chol=1.3, smoker=0, diabetes=0, egfr=50, acr=100)
CHANGES = [dict(age=45), dict(age=69), dict(age=70), dict(age=78), dict(age=85), dict(sex="female"), dict(sbp=100),
           dict(sbp=175), dict(total_chol=4.0), dict(total_chol=7.0), dict(hdl_chol=0.8), dict(hdl_chol=2.0),
           dict(smoker=1), dict(diabetes=1), dict(egfr=15), dict(egfr=30), dict(egfr=59), dict(egfr=61), dict(egfr=75),
           dict(egfr=89), dict(egfr=91), dict(egfr=105), dict(egfr=120), dict(acr=5), dict(acr=30), dict(acr=300),
           dict(acr=1500), dict(age=78, egfr=105), dict(age=78, egfr=75, acr=30)]


def patients(seed=2023):
    g = np.random.default_rng(seed)
    ps = [A, BASE] + [{**BASE, **c} for c in CHANGES]
    for i in range(40):
        ps.append(dict(age=int(g.integers(40, 86)), sex=("male", "female")[i % 2], sbp=int(g.integers(90, 181)),
                       total_chol=round(float(g.uniform(3.9, 7.2)), 1), hdl_chol=round(float(g.uniform(0.8, 2.0)), 1),
                       smoker=int(g.random() < 0.3), diabetes=int(g.random() < 0.3), egfr=int(g.integers(15, 121)),
                       acr=int(round(float(np.exp(g.uniform(np.log(5), np.log(1500))))))))
    return [dict(id=i, **p) for i, p in enumerate(ps)]


def query(p, acr):
    return dict(age=p["age"], sex=1 if p["sex"] == "male" else 2, sbp=p["sbp"], chol=p["total_chol"], hdlc=p["hdl_chol"],
                smoking=2 * p["smoker"], diabetes=p["diabetes"], egfr=p["egfr"], acr=acr)


def fetch():
    """One request per second; every query is stored with its response so the run is auditable offline."""
    out = []
    for p in patients():
        for addon, acr in (("egfr", NO_ACR), ("acr", p["acr"])):
            q = query(p, acr)
            req = urllib.request.Request(f"{URL}?{urllib.parse.urlencode(q)}", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out.append(dict(id=p["id"], addon=addon, query=q, response=json.loads(r.read())))
            time.sleep(1)
    head = dict(source=URL, n_queries=len(out))
    (FX / "ckdpc_results.jsonl").write_text("\n".join(json.dumps(x) for x in [head, *out]) + "\n")


def shown(v):
    return Decimal(repr(v)).quantize(Decimal("0.1"), ROUND_HALF_UP)


def compare():
    """(id, addon, 'base'|'ckd', region, calculator %, package %, package % with the SCORE2-OP eGFR >= 90
    segment dropped, as the calculator does) for every returned value; SCORE2 values once per patient."""
    from cvd_risk.score2 import coefficients as C
    from cvd_risk.score2.score2 import score2, score2_op
    from cvd_risk.score2.score2_ckd import score2_ckd
    ps = {p["id"]: p for p in patients()}
    published = C.CKD_EGFR_ADDON
    no_ge90 = {**published, "score2_op": {**published["score2_op"], "ge90": 0.0, "age_ge90": 0.0}}
    rows = []
    for ln in (FX / "ckdpc_results.jsonl").read_text().splitlines()[1:]:
        r = json.loads(ln)
        p = ps[r["id"]]
        assert query(p, r["query"]["acr"]) == r["query"], r["id"]
        x = (p["age"], p["sex"], p["smoker"], p["sbp"], p["total_chol"], p["hdl_chol"], p["diabetes"])
        for k, reg in enumerate(REGIONS, 1):
            if r["addon"] == "egfr":
                base = 100 * float((score2_op if p["age"] >= 70 else score2)(*x, reg))
                rows.append((r["id"], "-", "base", reg, r["response"][f"r0_{k}"], base, base))
            ckd = []
            for table in (published, no_ge90):
                C.CKD_EGFR_ADDON = table
                try:
                    ckd.append(100 * float(score2_ckd(*x, p["egfr"], reg, acr=p["acr"], addon=r["addon"])))
                finally:
                    C.CKD_EGFR_ADDON = published
            rows.append((r["id"], r["addon"], "ckd", reg, r["response"][f"r1_{k}"], *ckd))
    return rows


def agrees(calc, pkg):
    return shown(pkg) == Decimal(repr(float(calc)))


if __name__ == "__main__" and sys.argv[1:] == ["fetch"]:
    fetch()
elif __name__ == "__main__":
    ps = {p["id"]: p for p in patients()}
    for kind, ge90 in (("base", False), ("base", True), ("ckd", False), ("ckd", True)):
        sel = [r for r in compare() if r[2] == kind and (ps[r[0]]["age"] >= 70 and ps[r[0]]["egfr"] >= 90) == ge90]
        print(kind, "OP eGFR>=90" if ge90 else "other", len(sel),
              *(f"{sum(agrees(r[4], r[c]) for r in sel)} agree, max |diff| {max(abs(r[4] - r[c]) for r in sel):.3f};" for c in (5, 6)))
