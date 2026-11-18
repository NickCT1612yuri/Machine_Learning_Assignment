# Build the Banking77 LaTeX assignment report (HCMUT template).
# Prefer XeLaTeX for Vietnamese Unicode; falls back documented in README.

MAIN = main
LATEX = xelatex
FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex 0.tex hcmut.sty sections/*.tex Images/bachkhoa_logo.png
	$(LATEX) $(FLAGS) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f *.aux sections/*.aux
