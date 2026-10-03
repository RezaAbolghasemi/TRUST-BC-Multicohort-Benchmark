# LaTeX source of the v1.1 manuscript and supplement

Build (from this folder; figures are in `figs/`):

```bash
python make_numbers.py ../../results/SEER_v1.1 ../../supplementary/tables
python make_supp.py ../../results/SEER_v1.1 ../../supplementary/tables <SEER.csv> ../../results
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary && pdflatex supplementary      # uses xr to cite main.pdf pages
python to_docx.py main.tex main.docx main.aux main.bbl && python style_docx.py main.docx
```
`numbers.tex` and the `*_rows.tex` files are generated from the result files, so every
number in the manuscript can be traced to `results/` or `supplementary/tables/`.
