# Build the print-ready PDF of the ELI5 book.
#
#   make pdf              US Letter  ->  build/zero-to-sglang-eli5-letter.pdf
#   make pdf PAPER=a4     A4         ->  build/zero-to-sglang-eli5-a4.pdf
#   make html             write build/book.html only (open it in a browser to debug the layout)
#   make deps             create .venv and install the two Python packages
#   make clean            remove build/ (keeps .venv)      make distclean   also remove .venv
#
# Needs: python3 (3.9+), and Chrome or Chromium (set CHROME=/path/to/chrome if it is not found).
# From the repository root the same targets work: `make pdf`.

PY      ?= python3
VENV    ?= .venv
PAPER   ?= letter
OUT     ?= build/zero-to-sglang-eli5-$(PAPER).pdf

.PHONY: pdf html deps clean distclean help
.DEFAULT_GOAL := help

help:
	@sed -n '1,11p' Makefile | sed 's/^# \{0,1\}//'

$(VENV)/.stamp: requirements-pdf.txt
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --quiet --disable-pip-version-check -r requirements-pdf.txt
	touch $@

deps: $(VENV)/.stamp

pdf: $(VENV)/.stamp
	$(VENV)/bin/python tools/build_pdf.py --paper $(PAPER) --out $(OUT)

html: $(VENV)/.stamp
	$(VENV)/bin/python tools/build_pdf.py --paper $(PAPER) --html-only

clean:
	rm -rf build

distclean: clean
	rm -rf $(VENV)
