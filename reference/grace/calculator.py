"""GRACE against the developers' own calculator, the GRACE "ACS Risk Model" tool.

    python reference/grace/calculator.py   # writes fixtures/calculator_tables.json

The University of Massachusetts Center for Outcomes Research published the tool as a Flash application
at www.outcomes-umassmed.org/grace (Fox et al., BMJ 2006, Figure 2, shows it). source/grace_acs_risk_2013.swf
is the Wayback Machine capture of
https://www.outcomes-umassmed.org/grace/acs_risk/acs_risk.swf (2013). Its ActionScript is disassembled by
as2.py, and this script reads back:

- the point table of every input (one value per dropdown entry, index 0 being the unselected header;
  checkboxes as points when ticked / not ticked),
- which items each total adds up,
- the points-to-probability table of each total,
- the dropdown labels, which give the band edges.

Totals 1-4 are the "At Admission" calculator (in-hospital death, in-hospital death or MI, death to 6
months, death or MI to 6 months); 5-6 the "At Discharge" one (death and death or MI to 6 months after
discharge). The Pocket PC build of August 2004 (acs_risk.cab from the same folder) gives identical
tables; it is not stored here (1 MB), see REPORT.md.
"""

import json
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import as2  # noqa: E402

HERE = Path(__file__).parent
SWF = HERE / "source" / "grace_acs_risk_2013.swf"
OUT = HERE / "fixtures" / "calculator_tables.json"
UNITS = {"Years": "age", "bpm": "heart_rate", "mmHg": "sbp", "mg/dL": "creatinine", "Killip Class": "killip"}


def num(v):
    f = float(v)
    return int(f) if f == int(f) else f


def extract(path=SWF):
    raw = path.read_bytes()
    assert raw[:3] == b"CWS"
    body = b"FWS" + raw[3:8] + zlib.decompress(raw[8:])
    lines = [x.strip() for x in as2.disassemble(body)]
    s = " | ".join(lines)

    # dropdowns: switch on <name>_cb.value, then one block of assignments per case, in case order
    dropdowns = {}
    for m in re.finditer(r"Push '(\w+_cb)' \| GetVariable \| Push 'value' \| GetMember \| StoreRegister r0 \| ", s):
        rest = s[m.end():]
        cases = re.match(r"(?:Push (?:r0 )?[-\d.]+ \| StrictEquals \| If -?\d+ \| )+Jump -?\d+ \| ", rest)
        n = len(re.findall("StrictEquals", cases.group(0)))
        tail = rest[cases.end():].split(" | ")
        groups, cur, k = [], {}, 0
        while len(groups) < n:
            a = re.match(r"Push '(\w+)' (\S+)$", tail[k])
            if a and tail[k + 1] == "SetVariable":
                cur[a.group(1)] = num(a.group(2))
                k += 2
                continue
            if tail[k].startswith("Jump"):
                groups.append(cur)
                cur = {}
            k += 1
        dropdowns[m.group(1)] = {v: [g[v] for g in groups] for v in groups[0]}

    # checkboxes: if <name>_cb.selected then A else B
    checkboxes = {}
    for m in re.finditer(r"Push '(\w+_cb)' \| GetVariable \| Push 'selected' \| GetMember \| Not \| If \d+ \| "
                         r"((?:Push '\w+' \S+ \| SetVariable \| )+)Jump \d+ \| ((?:Push '\w+' \S+ \| SetVariable \| )+)", s):
        on = dict(re.findall(r"Push '(\w+)' (\S+)", m.group(2)))
        off = dict(re.findall(r"Push '(\w+)' (\S+)", m.group(3)))
        checkboxes[m.group(1)] = {v: [num(on[v]), num(off[v])] for v in on}

    totals = {}
    for k in range(1, 7):
        m = re.search(rf"Push 'total{k}' '(\w+)' \| GetVariable \| ((?:Push '\w+' \| GetVariable \| Add2 \| )+)SetVariable", s)
        totals[k] = [m.group(1)] + re.findall(r"Push '(\w+)'", m.group(2))

    probability = {}
    for k in range(1, 7):
        steps = re.findall(rf"Push 'total{k}' \| GetVariable \| Push (\S+) \| Less2 \| Not \| If \d+ \| "
                           rf"Push 'percent{k}' (\S+) \| SetVariable", s)
        last = re.findall(rf"Push 'percent{k}' (\S+) \| SetVariable", s)[-1]
        probability[k] = [[num(a), num(b)] for a, b in steps] + [[None, num(last)]]

    # dropdown labels: the unit header, then the entries, in the component parameters
    labels = {}
    for unit, name in UNITS.items():
        i = body.find(unit.encode() + b"\x00")
        while i >= 0 and name not in labels:
            items = body[i:i + 400].split(b"\x00")[1:]
            entries = []
            for x in items:
                t = x.decode("latin-1")
                if not (re.match(r"(<|>=)?\d", t) or (name == "killip" and re.match(r"I|V", t))):
                    break
                entries.append(t)
            if len(entries) >= 4:
                labels[name] = [unit] + entries
            i = body.find(unit.encode() + b"\x00", i + 1)
    return dict(source="https://web.archive.org/web/2013/http://www.outcomes-umassmed.org/grace/acs_risk/acs_risk.swf",
                labels=labels, dropdowns=dropdowns, checkboxes=checkboxes, totals=totals, probability=probability)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(extract(), indent=1) + "\n")
