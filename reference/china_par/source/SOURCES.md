# Source documents (not redistributed)

The publisher documents used to verify this score are not included in the repository. To rerun the checks
that read them, download each one from the link below and save it in this folder under the file name shown;
those checks are skipped when the file is absent. The SHA-256 is that of the copy used for the verification,
so you can confirm that you have the same file.

| File | Document | Where to get it | SHA-256 |
|---|---|---|---|
| `eastern_china_application.pdf` | China-PAR in eastern China, Biomed Environ Sci 2019;32(2):87-95 | https://doi.org/10.3967/bes2019.014 | `81d96d8a07ccde75abcfb881d1474c969d1526ccdf0948a9d8a368d78dadb1e1` |
| `yang2016_cdtm.pdf` | Yang X et al. Chronic Dis Transl Med 2016 (risk stratification of ASCVD in Chinese adults) | https://doi.org/10.1016/j.cdtm.2016.10.001 | `0bd16e15d0402900978676c2e077dfe4ee60574bf66be3401f7d6bae38bf5eb4` |
| `yang2016_cdtm_supplement.docx` | Yang X et al. Chronic Dis Transl Med 2016, supplemental material | https://doi.org/10.1016/j.cdtm.2016.10.001 (supplementary material on the article page) | `141a513767c862fd09577a0e179a6f1379a6d92c66f8b8e4581af952991ec5da` |
| `yang2016_circulation.pdf` | Yang X et al. Circulation 2016;134:1430-40 (China-PAR) | https://doi.org/10.1161/CIRCULATIONAHA.116.022367 | `95013287a9f9bfd44aad3af9b6093e386b02415c95b81560363648fd0094b8c8` |

Plain-text extractions of the documents above, made for searching; no code reads them. To recreate one,
extract the text of its document, for example with Ghostscript:
`gs -q -dNOPAUSE -dBATCH -sDEVICE=txtwrite -sOutputFile=NAME.txt NAME.pdf`.

| File | Text of | SHA-256 |
|---|---|---|
| `eastern_china_application.txt` | `eastern_china_application.pdf` | `7a6286cbf93f16bb9f0c307b3af7cd1078d6ba11e2708c6171b61005ed488652` |
| `yang2016_cdtm.txt` | `yang2016_cdtm.pdf` | `9bd77ac809abe29d4bd698ad6517c06487c7e7516ea139a81b82b48f3ecf9989` |
| `yang2016_cdtm_supplement.txt` | `yang2016_cdtm_supplement.docx` | `8279ddef942a8f35d082f7382bdf56a85a96af6ad2bd68b597e2aa850791e652` |
| `yang2016_circulation.txt` | `yang2016_circulation.pdf` | `1e1be0b069699a5cfb1853e63333423ee265c4dc12543c0281cc33a959e6e850` |
