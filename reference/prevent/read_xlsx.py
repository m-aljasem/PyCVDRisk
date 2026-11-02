"""Minimal .xlsx reader (stdlib only): dump every sheet as tab-separated text.
Usage: python reference/prevent/read_xlsx.py <file.xlsx> <out_dir>
"""

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group()
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def main(path, out):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for i, sh in enumerate(wb.find("m:sheets", NS)):
        xml = ET.fromstring(z.read("xl/" + target[sh.get(f"{{{NS['r']}}}id")].lstrip("/").removeprefix("xl/")))
        lines = []
        for row in xml.iter(f"{{{NS['m']}}}row"):
            cells = {}
            for c in row.findall("m:c", NS):
                v = c.find("m:v", NS)
                if v is None:
                    is_ = c.find("m:is", NS)
                    val = "".join(t.text or "" for t in is_.iter(f"{{{NS['m']}}}t")) if is_ is not None else ""
                else:
                    val = shared[int(v.text)] if c.get("t") == "s" else v.text
                cells[col_index(c.get("r"))] = val.replace("\t", " ").replace("\n", " ")
            if cells:
                lines.append("\t".join(cells.get(j, "") for j in range(max(cells) + 1)))
        (out / f"{i:02d}.tsv").write_text("\n".join(lines))
        print(f"{i:02d}", sh.get("name"), len(lines), "rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])
