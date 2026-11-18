# Build the report: tables/figures from ../results, then two XeLaTeX passes.
PYTHON ?= ../../.venv/bin/python

main.pdf: main.tex titlepage.tex hcmut.sty sections/*.tex make_tables.py
	$(PYTHON) make_tables.py
	xelatex -interaction=nonstopmode -halt-on-error main
	xelatex -interaction=nonstopmode -halt-on-error main
	rm -f main.aux main.log main.out main.toc

clean:
	rm -rf generated main.aux main.log main.out main.toc
