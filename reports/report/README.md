# LaTeX report (HCMUT template)

Academic report for Machine Learning Task 2 — three corpora (AG News, Rotten Tomatoes, Banking77) with TF-IDF, frozen BGE-small, and TextCNN. Numbers are synced from `outputs/full/`.

## Build

```bash
cd reports/report
make          # sync tables/figures from outputs/full, then XeLaTeX -> main.pdf
make sync     # only regenerate generated/*.tex and copy eda.png
make clean    # remove aux/log/toc
```

Uses `../../.venv/bin/python` by default for sync (`PYTHON=... make` to override). Requires TeX Live with `xelatex` and Times New Roman (or change `\setmainfont` in `main.tex`).

## Layout

| File | Role |
| --- | --- |
| `main.tex` | Document root |
| `0.tex` | Title page + TOC |
| `hcmut.sty` | Header/footer, geometry, packages |
| `sections/*.tex` | Narrative body |
| `generated/*.tex` | Auto-filled result tables from `sync_from_outputs.py` |
| `Images/bachkhoa_logo.png` | Logo |

Figures: `../figures/<dataset>/eda.png` (copied from outputs) and `pipeline_overview.png`.
