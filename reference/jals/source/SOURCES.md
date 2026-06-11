# Source documents (not redistributed)

The publisher documents used to verify this score are not included in the repository. To rerun the checks
that read them, download each one from the link below and save it in this folder under the file name shown;
those checks are skipped when the file is absent. The SHA-256 is that of the copy used for the verification,
so you can confirm that you have the same file.

| File | Document | Where to get it | SHA-256 |
|---|---|---|---|
| `harada2019.pdf` | Harada A et al. Hypertens Res 2019 (JALS score) | https://doi.org/10.1038/s41440-019-0220-z | `c686dcec0ff265645be300e3b249f1309579081613379ce9b41029a6ed1f14b6` |
| `supp1.pdf` | Harada A et al. Hypertens Res 2019, electronic supplementary material 1 | https://doi.org/10.1038/s41440-019-0220-z (supplementary material on the article page) | `2eec8ffa1bebc58186ba779ab60a7aa0452bd3cc4ed4668ef35d407d7b7417d9` |
| `supp2.pdf` | Harada A et al. Hypertens Res 2019, electronic supplementary material 2 | https://doi.org/10.1038/s41440-019-0220-z (supplementary material on the article page) | `7d43e337bbad6b99aae47309fea404359281b0e4db1d3e85444dba07d43f2c85` |
| `supp3.pdf` | Harada A et al. Hypertens Res 2019, electronic supplementary material 3 | https://doi.org/10.1038/s41440-019-0220-z (supplementary material on the article page) | `2db0acaeb0c86e1a7563238bbfc544db200552432abe842f0cdc8fd1d5ae88e5` |
| `supp4.pdf` | Harada A et al. Hypertens Res 2019, electronic supplementary material 4 | https://doi.org/10.1038/s41440-019-0220-z (supplementary material on the article page) | `f1323eeb6ea03fb115b63acbc58e60cc51cf2d94e30bc2fb8054d1c3ca8524ff` |

Plain-text extractions of the documents above, made for searching; no code reads them. To recreate one,
extract the text of its document, for example with Ghostscript:
`gs -q -dNOPAUSE -dBATCH -sDEVICE=txtwrite -sOutputFile=NAME.txt NAME.pdf`.

| File | Text of | SHA-256 |
|---|---|---|
| `harada2019.txt` | `harada2019.pdf` | `a295c6811549905502328c2be6431a75909ef7bcaf4e6b07de81fdbf2336c099` |
| `supp1.txt` | `supp1.pdf` | `a5241e268f5836bad550eed156857ae2590abe7bdbdb6ce4e84c1e5b27428ada` |
| `supp2.txt` | `supp2.pdf` | `b593fd2154e881737535a69d75c39b2f40f48c472bd86a1721f73d5e8f2171bf` |
| `supp3.txt` | `supp3.pdf` | `d43fc4747bf51bc699bbd6767e8772375cddb14cd14bb95de207e7bfa04cbc8b` |
| `supp4.txt` | `supp4.pdf` | `2a41ef8cf34f2911e6656d1ee5041166520aff815c042156b21ed1581e56fb3e` |
