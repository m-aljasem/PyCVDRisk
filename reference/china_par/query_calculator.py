"""Query the official China-PAR calculator (National Center for Cardiovascular Diseases / Fuwai Hospital,
https://www.cvdrisk.com.cn/ASCVD/Eval) for seeded synthetic patients and append them to fixtures/calculator_reference.csv.

    python reference/china_par/query_calculator.py [n] [seed]

The risk is computed on the server (no formula in the page). Each patient is one GET (fresh anti-forgery token)
and one POST to /ASCVD/Eval/Result, then a 1 s pause. The page shows the 10-year risk and the risk "under ideal
risk factors" with one decimal ("10.8%"), and below age 60 also the lifetime risk and its ideal value (a separate
China-PAR model), written here as shown. Form ranges: age 20-85, waist 50-130 cm,
TC 80-400 and HDL 20-130 mg/dl, SBP 70-200, DBP 40-140 mmHg. Row 0 of seed 0 is the paper's example (man, 60).
"""

import csv
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

URL = "https://www.cvdrisk.com.cn/ASCVD/Eval"
OUT = Path(__file__).parent / "fixtures" / "calculator_reference.csv"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
COLS = ["seed", "id", "sex", "age", "north", "urban", "waist", "tc", "hdl", "sbp", "dbp", "bp_treated", "diabetes",
        "smoker", "family_history", "risk_shown", "ideal_shown", "lifetime_shown", "lifetime_ideal_shown"]


def patients(n, seed):
    rnd = random.Random(seed)
    out = []
    if seed == 0:  # Supplemental Table 1 example (man; Table 2 also prints the woman)
        out.append(dict(sex=1, age=60, north=1, urban=1, waist=80, tc=210, hdl=55, sbp=130, dbp=80, bp_treated=0,
                        diabetes=1, smoker=0, family_history=0))
    while len(out) < n:
        tc = rnd.randint(120, 320)
        out.append(dict(sex=rnd.choice([1, 2]), age=rnd.randint(30, 80), north=rnd.randint(0, 1), urban=rnd.randint(0, 1),
                        waist=rnd.randint(60, 115), tc=tc, hdl=rnd.randint(25, min(100, tc - 20)), sbp=rnd.randint(95, 195),
                        dbp=rnd.randint(55, 110), bp_treated=rnd.randint(0, 1), diabetes=int(rnd.random() < 0.3),
                        smoker=int(rnd.random() < 0.4), family_history=int(rnd.random() < 0.3)))
    return out


def query(p):
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    opener.addheaders = [("User-Agent", UA)]
    form = opener.open(URL, timeout=60).read().decode("utf-8")
    token = re.search(r'name="__RequestVerificationToken" type="hidden" value="([^"]+)"', form).group(1)
    data = dict(sex=p["sex"], age=p["age"], region=p["north"], area=p["urban"], waist=p["waist"], tc_unit=1, tc=p["tc"],
                hdlc_unit=1, hdlc=p["hdl"], sbp=p["sbp"], dbp=p["dbp"], drug=p["bp_treated"], dm=p["diabetes"],
                csmoke=p["smoker"], fh_ascvd=p["family_history"], __RequestVerificationToken=token)
    page = opener.open(URL + "/Result", urllib.parse.urlencode(data).encode(), timeout=60).read().decode("utf-8")
    risk = re.findall(r"风险为\s*(?:<[^>]+>\s*)*([\d.]+)%", page)
    ideal = re.findall(r"应小于\s*(?:<[^>]+>\s*)*([\d.]+)%", page)
    if len(risk) != len(ideal) or len(risk) != 1 + (p["age"] < 60):  # lifetime risk only below 60
        raise RuntimeError(f"unexpected result page for {p}")
    shown = [risk[0], ideal[0], *(risk[1:] + ideal[1:] or ["", ""])]
    echoed = re.search(r"年龄 \(岁\)\s*(?:<[^>]+>\s*)*(\d+)", re.sub(r"\s+", " ", page))
    assert echoed is None or int(echoed.group(1)) == p["age"], "calculator echoed a different age"
    return shown


def main(n, seed):
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(int(r["seed"]), int(r["id"])) for r in csv.DictReader(OUT.open())}
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        if new:
            w.writeheader()
        for i, p in enumerate(patients(n, seed)):
            if (seed, i) in done:
                continue
            for attempt in range(4):  # the server drops a connection now and then
                try:
                    shown = query(p)
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(10 * (attempt + 1))
            w.writerow(dict(seed=seed, id=i, **p, **dict(zip(COLS[-4:], shown))))
            f.flush()
            print(i, p["sex"], p["age"], *shown, flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
