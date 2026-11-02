"""Read the two sample WHO/ISH charts printed in the 2007 guideline, and the whoishRisk table.

The guideline (source/who2007_guideline.pdf) prints the charts for two subregions:
Annex 3 (PDF page 85), SEAR-D with cholesterol; Annex 4 (PDF page 87), EMR-B without.
Each cell is read from a ghostscript render: the colour at the cell centre is matched to
the five key colours. The worst colour distance is reported, so a misread cannot pass silently.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PDF = HERE / "source" / "who2007_guideline.pdf"
WHOISH = HERE / "source" / "whoishRisk" / "WHO_ISH_Scores.csv"
BANDS = ("<10%", "10% to <20%", "20% to <30%", "30% to <40%", ">=40%")
PALETTE = np.array([(120, 176, 72), (248, 240, 88), (232, 136, 72), (216, 32, 56), (160, 16, 72)])
AGES, SBPS = (70, 60, 50, 40), (180, 160, 140, 120)  # top to bottom, as printed


def render(page, dpi=200):
    out = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=ppmraw", f"-r{dpi}",
                          f"-dFirstPage={page}", f"-dLastPage={page}", "-sOutputFile=-", str(PDF)],
                         capture_output=True, check=True).stdout
    head, pos = [], 0
    while len(head) < 4:  # P6 width height maxval, skipping '#' comment lines
        end = out.index(b"\n", pos)
        line, pos = out[pos:end], end + 1
        if not line.startswith(b"#"):
            head += line.split()
    w, h = int(head[1]), int(head[2])
    return np.frombuffer(out[pos:pos + w * h * 3], np.uint8).reshape(h, w, 3).astype(int)


def _runs(profile, min_gap):
    """Runs where profile > 0, merging runs split by fewer than min_gap pixels (grid lines)."""
    idx = np.nonzero(profile > 0)[0]
    out = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - out[-1][1] <= min_gap:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b + 1) for a, b in out if b + 1 - a > 20]


def _cells(page, box, ncol, nrow):
    """Band index of each cell; panels found as coloured blocks inside `box` (page fractions)."""
    img = render(page)
    h, w, _ = img.shape
    sub = img[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
    d = np.linalg.norm(sub[:, :, None, :] - PALETTE[None, None], axis=3)
    coloured = d.min(axis=2) < 40
    xs, ys = _runs(coloured.sum(axis=0), 8), _runs(coloured.sum(axis=1), 8)
    grid = np.zeros((len(ys) * nrow, len(xs) * ncol), int)
    worst = 0.0
    for i, (ya, yb) in enumerate(ys):
        for j, (xa, xb) in enumerate(xs):
            for r in range(nrow):
                for c in range(ncol):
                    cy = int(ya + (r + 0.5) * (yb - ya) / nrow)
                    cx = int(xa + (c + 0.5) * (xb - xa) / ncol)
                    dist = d[cy - 3:cy + 4, cx - 3:cx + 4].mean(axis=(0, 1))
                    grid[i * nrow + r, j * ncol + c] = dist.argmin()
                    worst = max(worst, dist.min())
    return grid, worst


def official():
    """{(region, diabetes, male, smoker, age, sbp, chol): band} for every printed cell (chol 0 = no
    cholesterol chart), and the worst colour distance (0-441; key colours are >= 90 apart)."""
    out = {}
    # Annex 3: 8 panel rows (diabetes: 70..40, then without: 70..40) x 4 panel columns
    # (men non-smoker, men smoker, women non-smoker, women smoker); 4 SBP rows x 5 cholesterol columns.
    g, w3 = _cells(85, (0.1, 0.3, 0.7, 0.9), 5, 4)
    assert g.shape == (32, 20), g.shape
    for i in range(8):
        for j, (male, sm) in enumerate(((1, 0), (1, 1), (0, 0), (0, 1))):
            for r in range(4):
                for c in range(5):
                    out[("sear_d", int(i < 4), male, sm, AGES[i % 4], SBPS[r], 4 + c)] = g[i * 4 + r, j * 5 + c]
    # Annex 4: 4 panel rows (age 70..40) x 8 single columns (without diabetes: men smoker, men
    # non-smoker, women smoker, women non-smoker; then the same with diabetes); 4 SBP rows.
    g, w4 = _cells(87, (0.1, 0.3, 0.86, 0.73), 1, 4)
    assert g.shape == (16, 8), g.shape
    for i in range(4):
        for j in range(8):
            male, sm = ((1, 1), (1, 0), (0, 1), (0, 0))[j % 4]
            for r in range(4):
                out[("emr_b", int(j >= 4), male, sm, AGES[i], SBPS[r], 0)] = g[i * 4 + r, j]
    return out, max(w3, w4)


def whoishrisk():
    """{(region, diabetes, male, smoker, age, sbp, chol): band} from whoishRisk's table."""
    out = {}
    with WHOISH.open(newline="") as f:
        for r in csv.DictReader(f):
            k = tuple(int(r[c]) for c in ("dm", "gdr", "smk", "age", "sbp", "chl"))
            for reg in list(r)[7:]:
                out[(reg.lower(),) + k] = BANDS.index(r[reg])
    return out
