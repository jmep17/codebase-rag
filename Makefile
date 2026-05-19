PYTHON ?= python3
VENV ?= .venv
BIN_DIR ?= $(HOME)/.local/bin
COMMAND ?= codebase-rag
EXTRAS ?=
SHELL_IMAGE ?= python:3.13-slim
SHELL_TIMEOUT ?= 30

ifeq ($(strip $(EXTRAS)),)
INSTALL_TARGET := .
else
INSTALL_TARGET := .[$(EXTRAS)]
endif

.PHONY: help venv install install-global uninstall-global chat-safe-shell

help:
	@printf '%s\n' \
		'Targets:' \
		'  make install             Create .venv if needed and install editable package' \
		'  make install-global      Install a PATH wrapper at ~/.local/bin/codebase-rag' \
		'  make uninstall-global    Remove the PATH wrapper' \
		'  make chat-safe-shell     Chat with shell enabled in Docker, network disabled' \
		'' \
		'Options:' \
		'  EXTRAS=web,serve         Install optional extras, e.g. make install EXTRAS=web' \
		'  SHELL_IMAGE=python:3.13  Docker image for make chat-safe-shell' \
		'  SHELL_TIMEOUT=60         Per-command timeout for make chat-safe-shell'

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

chat-safe-shell:
	"$(VENV)/bin/$(COMMAND)" chat --allow-shell --shell-runner "docker:$(SHELL_IMAGE)" --shell-network none --shell-timeout "$(SHELL_TIMEOUT)"
