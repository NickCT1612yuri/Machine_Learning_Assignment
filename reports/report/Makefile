# Build the HCMUT LaTeX assignment report (3-dataset hybrid pipeline).
# Prefer XeLaTeX for Vietnamese Unicode.

MAIN = main
LATEX = xelatex
FLAGS = -interaction=nonstopmode -halt-on-error
OUTROOT ?= ../../outputs/full
PYTHON ?= ../../.venv/bin/python

.PHONY: all clean sync

all: sync $(MAIN).pdf

sync:
	$(PYTHON) sync_from_outputs.py --root $(OUTROOT)

$(MAIN).pdf: $(MAIN).tex 0.tex hcmut.sty sections/*.tex generated/*.tex Images/bachkhoa_logo.png
	$(LATEX) $(FLAGS) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f *.aux sections/*.aux
