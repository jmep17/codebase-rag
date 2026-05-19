PYTHON ?= python3
VENV ?= .venv
BIN_DIR ?= $(HOME)/.local/bin
COMMAND ?= codebase-rag
EXTRAS ?=

ifeq ($(strip $(EXTRAS)),)
INSTALL_TARGET := .
else
INSTALL_TARGET := .[$(EXTRAS)]
endif

.PHONY: help venv install install-global uninstall-global

help:
	@printf '%s\n' \
		'Targets:' \
		'  make install             Create .venv if needed and install editable package' \
		'  make install-global      Install a PATH wrapper at ~/.local/bin/codebase-rag' \
		'  make uninstall-global    Remove the PATH wrapper' \
		'' \
		'Options:' \
		'  EXTRAS=web,serve         Install optional extras, e.g. make install EXTRAS=web'

venv: $(VENV)/bin/python

$(VENV)/bin/python:
	$(PYTHON) -m venv "$(VENV)"

install: venv
	"$(VENV)/bin/python" -m pip install -e "$(INSTALL_TARGET)"

install-global: install
	mkdir -p "$(BIN_DIR)"
	printf '%s\n' '#!/bin/sh' 'exec "$(abspath $(VENV))/bin/python" -m codebase_rag "$$@"' > "$(BIN_DIR)/$(COMMAND)"
	chmod +x "$(BIN_DIR)/$(COMMAND)"
	@printf 'Installed %s\n' "$(BIN_DIR)/$(COMMAND)"
	@case ":$$PATH:" in \
		*:"$(BIN_DIR)":*) ;; \
		*) printf 'Add %s to PATH to run %s from any directory.\n' "$(BIN_DIR)" "$(COMMAND)" ;; \
	esac

uninstall-global:
	rm -f "$(BIN_DIR)/$(COMMAND)"
	@printf 'Removed %s\n' "$(BIN_DIR)/$(COMMAND)"
