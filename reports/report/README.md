# LaTeX report (HCMUT template)

Academic report for Machine Learning Task 2 — Banking77.

## Build

```bash
cd reports/report
make          # XeLaTeX, two passes -> main.pdf
make clean    # remove aux/log/toc
```

Requires TeX Live with `xelatex` and Times New Roman (or change `\setmainfont` in `main.tex`).

## Layout

| File | Role |
| --- | --- |
| `main.tex` | Document root |
| `0.tex` | Title page + table of contents |
| `hcmut.sty` | Header/footer, geometry, shared packages |
| `sections/*.tex` | Body sections |
| `Images/bachkhoa_logo.png` | Logo for title page and header |
| `ML_Assignment_Report.md` | Markdown draft (not compiled) |

Figures are loaded from `../figures/` via `\graphicspath`.
