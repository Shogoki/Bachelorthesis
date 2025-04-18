PY=python
PANDOC=pandoc

BASEDIR=$(CURDIR)
INPUTDIR=$(BASEDIR)/source
OUTPUTDIR=$(BASEDIR)/output
TEMPLATEDIR=$(INPUTDIR)/templates
STYLEDIR=$(BASEDIR)/style
EXPOSEINPUTDIR=$(BASEDIR)/expose_src
BIBFILE=$(INPUTDIR)/references.bib
E_BIBFILE=$(EXPOSEINPUTDIR)/references.bib

help:
	@echo ' 																	  '
	@echo 'Makefile for the Markdown thesis                                       '
	@echo '                                                                       '
	@echo 'Usage:                                                                 '
	@echo '   make html                        generate a web version             '
	@echo '   make pdf                         generate a PDF file  			  '
	@echo '   make docx	                       generate a Docx file 			  '
	@echo '   make tex	                       generate a Latex file 			  '
	@echo '                                                                       '
	@echo ' 																	  '
	@echo ' 																	  '
	@echo 'get local templates with: pandoc -D latex/html/etc	  				  '
	@echo 'or generic ones from: https://github.com/jgm/pandoc-templates		  '

pdf:
	./convert_svg.sh && \
	pandoc "$(INPUTDIR)"/*.md \
	-o "$(OUTPUTDIR)/thesis.pdf" \
	-H "$(STYLEDIR)/preamble.tex" \
	--template="$(STYLEDIR)/template.tex" \
	--bibliography="$(BIBFILE)" \
	--csl="$(STYLEDIR)/ref_format.csl" \
	--highlight-style pygments \
	-V fontsize=12pt \
	-V papersize=a4paper \
	-V documentclass=scrbook \
	-V include-after="\printEidesstattlicheErklaerung" \
	-N \
	--pdf-engine=pdflatex \
	--verbose  2>pandoc.log

pipeline:
	./convert_svg.sh && \
	pandoc "$(INPUTDIR)"/*.md \
	-o "$(OUTPUTDIR)/the owsis.pdf" \
	-H "$(STYLEDIR)/preamble.tex" \
	--template="$(STYLEDIR)/template.tex" \
	--bibliography="$(BIBFILE)" \
	--csl="$(STYLEDIR)/ref_format.csl" \
	--highlight-style pygments \
	-V fontsize=12pt \
	-V papersize=a4paper \
	-V documentclass=scrbook \
	-N \
	--pdf-engine=xelatex \
	--verbose && \
	python3 upload_thesis.py $(owncloud-pw)
pdf_cicd:
	./convert_svg.sh && \
	pandoc "$(INPUTDIR)"/*.md \
	-o "$(OUTPUTDIR)/thesis.pdf" \
	-H "$(STYLEDIR)/preamble.tex" \
	--template="$(STYLEDIR)/template.tex" \
	--bibliography="$(BIBFILE)" \
	--csl="$(STYLEDIR)/ref_format.csl" \
	--highlight-style pygments \
	-V fontsize=12pt \
	-V papersize=a4paper \
	-V documentclass=scrbook \
	-N \
	--pdf-engine=xelatex \
	--verbose

expose:
	pandoc "$(EXPOSEINPUTDIR)"/*.md \
		-o "$(OUTPUTDIR)/expose.pdf" \
		-H "$(STYLEDIR)/preamble.tex" \
		--template="$(STYLEDIR)/template.tex" \
		--bibliography="$(E_BIBFILE)" 2>pandoc.log \
		--csl="$(STYLEDIR)/ref_format.csl" \
		--highlight-style pygments \
		-V fontsize=12pt \
		-V papersize=a4paper \
		-V documentclass=article \
		-N \
		--pdf-engine=xelatex \
		--verbose

tex2:
	pandoc "$(INPUTDIR)"/*.md \
		-o "$(OUTPUTDIR)/thesis.tex" \
		-H "$(STYLEDIR)/preamble.tex" \
		--template="$(STYLEDIR)/template.tex" \
		--bibliography="$(BIBFILE)" \
		--csl="$(STYLEDIR)/ref_format.csl" \
		--highlight-style pygments \
		-V fontsize=12pt \
		-V papersize=a4paper \
		-V documentclass=article \
		-N \
		--pdf-engine=xelatex \
		--verbose  

tex:
	pandoc "$(INPUTDIR)"/*.md \
	-o "$(OUTPUTDIR)/thesis.tex" \
	-H "$(STYLEDIR)/preamble.tex" \
	--bibliography="$(BIBFILE)" \
	--template="$(STYLEDIR)/template.tex" \
	-V fontsize=12pt \
	-V papersize=a4paper \
	-V documentclass=article \
	-N \
	--csl="$(STYLEDIR)/ref_format.csl" \
	--latex-engine=xelatex

docx:
	pandoc "$(INPUTDIR)"/*.md \
	-o "$(OUTPUTDIR)/thesis.docx" \
	--bibliography="$(BIBFILE)" \
	--csl="$(STYLEDIR)/ref_format.csl" \
	--toc

html:
	pandoc "$(INPUTDIR)"/*.md \
	-o "$(OUTPUTDIR)/thesis.html" \
	--standalone \
	--template="$(STYLEDIR)/template.html" \
	--bibliography="$(BIBFILE)" \
	--csl="$(STYLEDIR)/ref_format.csl" \
	--include-in-header="$(STYLEDIR)/style.css" \
	--toc \
	--number-sections
	rm -rf "$(OUTPUTDIR)/source"
	mkdir "$(OUTPUTDIR)/source"
	cp -r "$(INPUTDIR)/figures" "$(OUTPUTDIR)/source/figures"

.PHONY: help pdf docx html tex
