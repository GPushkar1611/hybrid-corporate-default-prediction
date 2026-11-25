.PHONY: all clean

all: report-template.pdf

report-template.pdf: main.tex body.tex references.bib assets/iith-logo.png
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=report-template main.tex

clean:
	latexmk -c -jobname=report-template main.tex
