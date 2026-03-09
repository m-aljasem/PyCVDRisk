# Source documents (not redistributed)

The publisher documents used to verify this score are not included in the repository. To rerun the checks
that read them, download each one from the link below and save it in this folder under the file name shown;
those checks are skipped when the file is absent. The SHA-256 is that of the copy used for the verification,
so you can confirm that you have the same file.

| File | Document | Where to get it | SHA-256 |
|---|---|---|---|
| `ecg_watson_calculator_page.pdf` | ecg.watson.jp Hisayama ASCVD calculator page, printed to PDF | https://ecg.watson.jp/medical/ASCVD/ | `ce897b03c3ce41205c71e4a72b934d5f97ed1226a9225060fa36e4286bd68344` |
| `honda2022.pdf` | Honda T et al. J Atheroscler Thromb 2022 (Hisayama ASCVD model) | https://doi.org/10.5551/jat.61960 | `4211287bb020380c2609a151fee10aba9bb5be281961acd5240d557a487e1866` |

Plain-text extractions of the documents above, made for searching; no code reads them. To recreate one,
extract the text of its document, for example with Ghostscript:
`gs -q -dNOPAUSE -dBATCH -sDEVICE=txtwrite -sOutputFile=NAME.txt NAME.pdf`.

| File | Text of | SHA-256 |
|---|---|---|
| `honda2022.txt` | `honda2022.pdf` | `9005476158dc81f19bd6140e029ee2b4db4759c62933bcb4eb343feded692672` |
