# Convenience wrapper. `make pdf` builds the print-ready PDF of the ELI5 book.
# Options and details: eli5/Makefile   (e.g. `make pdf PAPER=a4`).
.PHONY: pdf html deps clean distclean
pdf html deps clean distclean:
	$(MAKE) -C eli5 $@
