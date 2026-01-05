"""Query the CUORE Project's online individual-score calculator (ISS; progettocuore.medisoft.it/CRCOnline, embedded
at www.cuore.iss.it/valutazione/calc-rischio) for seeded synthetic patients and write fixtures/calculator_reference.csv.

    python -I reference/cuore/query_calculator.py [n]

Each patient is one GET (fresh ASP.NET form state) and one POST, then a 0.7 s pause. The page shows the 10-year
risk with one decimal ("8,9% nei prossimi 10 anni"), written here as shown. A third of the patients get two
different SBP readings (the form asks for two), and some sit at the edges of the score's stated ranges
(age 35-69, SBP 90-200, TC 130-320, HDL 20-100 mg/dl).
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

URL = "https://progettocuore.medisoft.it/CRCOnline/"
OUT = Path(__file__).parent / "fixtures" / "calculator_reference.csv"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


def patients(n, rnd):
    for i in range(n):
        edge = i % 10 == 0
        sbp1 = rnd.choice([90, 200]) if edge else rnd.randint(90, 200)
        yield {"Sex": rnd.choice([1, 2]), "Age": rnd.choice([35, 69]) if edge else rnd.randint(35, 69),
               "Smoke": rnd.randint(0, 1), "Sbp1": sbp1,
               "Sbp2": min(200, max(90, sbp1 + rnd.randint(-15, 15))) if i % 3 == 1 else sbp1,
               "Cholesterol": rnd.choice([130, 320]) if edge else rnd.randint(130, 320),
               "HDL": rnd.choice([20, 100]) if edge else rnd.randint(20, 100),
               "Diabetes": rnd.randint(0, 1), "HT": rnd.randint(0, 1)}


def main(n):
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    op.addheaders = [("User-Agent", UA)]
    rows = []
    for p in patients(n, random.Random(2007)):
        for attempt in range(4):  # the server occasionally times out the TLS handshake
            try:
                page = op.open(URL, timeout=30).read().decode()
                state = dict(re.findall(r'<input type="hidden" name="([^"]+)" id="[^"]+" value="([^"]*)"', page))
                body = urllib.parse.urlencode({**state, **{k: str(v) for k, v in p.items()}, "DoCRC": "Calcola Rischio"})
                r = op.open(URL, body.encode(), timeout=30).read().decode()
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))
        m = re.search(r"([<&;\w\s]*?[\d]+(?:,\d)?)%\s*(?:</[^>]+>\s*)*nei prossimi 10 anni", r)
        rows.append({**p, "shown": m.group(1).strip().replace(",", ".") if m else ""})
        print(rows[-1], flush=True)
        time.sleep(0.7)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
