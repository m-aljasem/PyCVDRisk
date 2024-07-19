"""Build src/cvd_risk/uniscore/routing.csv: which score each country gets.
Run from repo root: python reference/uniscore/build_routing.py

Rules, first match wins (each is the score the local guideline body recommends):
  1. USA -> PREVENT (AHA 2023).
  2. GBR -> QRISK3 (NICE NG238).
  3. The 55 countries in SCORE2 Figure 5 (ESC) -> SCORE2 / SCORE2-OP, with that risk region.
  4. The 48 countries in SCORE2 Asia-Pacific supplementary Table 4 -> SCORE2 Asia-Pacific.
  5. Any other country with WHO 2019 parameters -> WHO 2019 (lab), country recalibration.
"""

from __future__ import annotations

import csv
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[2]
AP_DOCX = ROOT / "reference" / "score2" / "source" / "score2_asia_pacific_ehae609_supplement.docx"
OUT = ROOT / "src" / "cvd_risk" / "uniscore" / "routing.csv"

# SCORE2 working group, Eur Heart J 2021;42:2439, Figure 5 (country names as printed).
SCORE2_FIG5 = {
    "low": ["Netherlands", "France", "Norway", "Belgium", "Denmark", "Spain", "Switzerland", "Israel",
            "Luxembourg", "United Kingdom"],
    "moderate": ["Slovenia", "Greece", "Malta", "Italy", "Finland", "Austria", "Portugal", "Germany",
                 "Cyprus", "Sweden", "Ireland", "Iceland"],
    "high": ["Croatia", "Bosnia and Herzegovina", "Poland", "Albania", "Estonia", "Hungary", "Slovakia",
             "Czech Republic", "Kazakhstan", "Turkey"],
    "very_high": ["Serbia", "Montenegro", "Georgia", "Bulgaria", "Armenia", "Lithuania", "Latvia", "Ukraine",
                  "Russian Federation", "Romania", "Belarus", "Moldova", "Macedonia (TFYR)", "Kyrgyzstan",
                  "Egypt", "Azerbaijan", "Tunisia", "Lebanon", "Morocco", "Uzbekistan", "Syrian Arab Republic",
                  "Libya", "Algeria"],
}

ISO3 = {
    "Netherlands": "NLD", "France": "FRA", "Norway": "NOR", "Belgium": "BEL", "Denmark": "DNK", "Spain": "ESP",
    "Switzerland": "CHE", "Israel": "ISR", "Luxembourg": "LUX", "United Kingdom": "GBR", "Slovenia": "SVN",
    "Greece": "GRC", "Malta": "MLT", "Italy": "ITA", "Finland": "FIN", "Austria": "AUT", "Portugal": "PRT",
    "Germany": "DEU", "Cyprus": "CYP", "Sweden": "SWE", "Ireland": "IRL", "Iceland": "ISL", "Croatia": "HRV",
    "Bosnia and Herzegovina": "BIH", "Poland": "POL", "Albania": "ALB", "Estonia": "EST", "Hungary": "HUN",
    "Slovakia": "SVK", "Czech Republic": "CZE", "Kazakhstan": "KAZ", "Turkey": "TUR", "Serbia": "SRB",
    "Montenegro": "MNE", "Georgia": "GEO", "Bulgaria": "BGR", "Armenia": "ARM", "Lithuania": "LTU",
    "Latvia": "LVA", "Ukraine": "UKR", "Russian Federation": "RUS", "Romania": "ROU", "Belarus": "BLR",
    "Moldova": "MDA", "Macedonia (TFYR)": "MKD", "Kyrgyzstan": "KGZ", "Egypt": "EGY", "Azerbaijan": "AZE",
    "Tunisia": "TUN", "Lebanon": "LBN", "Morocco": "MAR", "Uzbekistan": "UZB", "Syrian Arab Republic": "SYR",
    "Libya": "LBY", "Algeria": "DZA",
    # Asia-Pacific supplementary Table 4
    "Hong Kong, special administrative region of China": "HKG", "South Korea": "KOR", "Japan": "JPN",
    "Australia": "AUS", "New Zealand": "NZL", "Singapore": "SGP", "Thailand": "THA", "Sri Lanka": "LKA",
    "Kuwait": "KWT", "Maldives": "MDV", "Tonga": "TON", "Nepal": "NPL", "Jordan": "JOR", "Bhutan": "BTN",
    "Brunei": "BRN", "China": "CHN", "Malaysia": "MYS", "India": "IND", "Bangladesh": "BGD", "Iran": "IRN",
    "Bahrain": "BHR", "United Arab Emirates": "ARE", "Vietnam": "VNM", "Cambodia": "KHM", "North Korea": "PRK",
    "Timor-Leste": "TLS", "Qatar": "QAT", "Myanmar": "MMR", "Saudi Arabia": "SAU", "Papua New Guinea": "PNG",
    "Laos": "LAO", "Indonesia": "IDN", "American Samoa": "ASM", "Philippines": "PHL", "Pakistan": "PAK",
    "Syria": "SYR", "Iraq": "IRQ", "Fiji": "FJI", "Turkmenistan": "TKM", "Oman": "OMN", "Yemen": "YEM",
    "Mongolia": "MNG", "Afghanistan": "AFG", "Solomon Islands": "SLB", "Kiribati": "KIR",
    "Micronesia": "FSM", "Vanuatu": "VUT", "Tajikistan": "TJK",
}


def asia_pacific() -> dict[str, list[str]]:
    """Country names by risk region, parsed from supplementary Table 4."""
    x = zipfile.ZipFile(AP_DOCX).read("word/document.xml").decode()
    t = re.sub(r"</w:tc>", "|", x)
    t = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", " ", t))
    t = t[t.rindex("Supplementary Table 4: Age"):t.rindex("Supplementary Table 5: Region")]
    out, region = {}, None
    cells = [c.strip() for c in t.split("|")]
    for i, c in enumerate(cells):
        m = re.fullmatch(r"(?:.*\s)?(Low|Moderate|High|Very high) risk region", c)
        if m:
            region = m.group(1).lower().replace(" ", "_")
            out[region] = []
        elif region and i + 2 < len(cells) and re.fullmatch(r"\d+(\.\d+)?", cells[i + 2]) and c:
            out[region].append(c)
    return out


def rows():
    who = {r["iso3"] for r in csv.DictReader((ROOT / "src/cvd_risk/who2019/countries.csv").open())}
    out = {"USA": ("prevent", "", "AHA 2023 (PREVENT)"), "GBR": ("qrisk3", "", "NICE NG238 (QRISK3)")}
    for region, names in SCORE2_FIG5.items():
        for n in names:
            out.setdefault(ISO3[n], ("score2", region, f"SCORE2 Fig 5: {n}"))
    for region, names in asia_pacific().items():
        for n in names:
            out.setdefault(ISO3[n], ("score2_asia_pacific", region, f"SCORE2 Asia-Pacific suppl. Table 4: {n}"))
    for iso in sorted(who):
        out.setdefault(iso, ("who2019", "", "WHO 2019 country recalibration"))
    return out


if __name__ == "__main__":
    ap = asia_pacific()
    assert sum(map(len, ap.values())) == 48 and sum(map(len, SCORE2_FIG5.values())) == 55, ap
    r = rows()
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso3", "score", "region", "source"])
        for iso in sorted(r):
            w.writerow([iso, *r[iso]])
    from collections import Counter
    print(Counter(v[0] for v in r.values()), file=sys.stderr)
