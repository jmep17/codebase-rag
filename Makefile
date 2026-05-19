PYTHON ?= python3
VENV ?= .venv
BIN_DIR ?= $(HOME)/.local/bin
COMMAND ?= codebase-rag
EXTRAS ?=
DEV_EXTRAS ?= dev
TEST_DEPS ?= pytest
MODEL ?= qwen3:8b
HOST ?= 127.0.0.1
PORT ?= 8723
ROOT ?= .
SHELL_IMAGE ?= python:3.13-slim
SHELL_TIMEOUT ?= 30
CBR_OLLAMA_PORT ?= 11435

VENV_BIN := $(VENV)/bin
VENV_PY := $(VENV_BIN)/python
CLI := $(VENV_BIN)/$(COMMAND)

ifeq ($(strip $(EXTRAS)),)
INSTALL_TARGET := .
DEV_INSTALL_TARGET := .[$(DEV_EXTRAS)]
else
INSTALL_TARGET := .[$(EXTRAS)]
DEV_INSTALL_TARGET := .[$(DEV_EXTRAS),$(EXTRAS)]
endif

.DEFAULT_GOAL := help

.PHONY: help venv require-cli install install-dev hooks check fix test smoke doctor chat chat-tui cbr-browser serve install-global uninstall-global chat-safe-shell container-build container-up container-pull-model container-doctor container-index container-chat

help:
	@printf '%s\n' \
		'Setup:' \
		'  make install             Create .venv and install the editable package' \
		'  make install-dev         Install editable package plus dev tooling' \
		'  make hooks               Enable this repo'\''s committed git hooks' \
		'  make install-global      Install a PATH wrapper at ~/.local/bin/codebase-rag' \
		'  make uninstall-global    Remove the PATH wrapper' \
		'' \
		'Checks:' \
		'  make check               Run AST, ruff format --check, and ruff check' \
		'  make fix                 Run formatter and safe lint fixes' \
		'  make test                Run pytest tests' \
		'  make smoke               Check the main CLI help paths' \
		'  make doctor              Check local setup for ROOT and MODEL' \
		'' \
		'Run:' \
		'  make chat                Start terminal chat' \
		'  make chat-tui            Start Textual TUI chat' \
		'  make chat-safe-shell     Chat with Docker shell runner, network disabled' \
		'  make cbr-browser         Start the local browser app' \
		'  make serve               Start loopback HTTP/WebSocket server' \
		'  make container-build     Build the isolated codebase-rag container' \
		'  make container-up        Start containerized Ollama on loopback' \
		'  make container-pull-model Pull MODEL into the Ollama Docker volume' \
		'  make container-index     Index ROOT into the container state volume' \
		'  make container-chat      Chat read-only against ROOT in containers' \
		'' \
		'Options:' \
		'  EXTRAS=web,serve,tui     Install optional extras' \
		'  DEV_EXTRAS=dev           Extra used by make install-dev' \
		'  TEST_DEPS=pytest         Test-only packages for make install-dev' \
		'  MODEL=qwen2.5-coder:7b   Model for chat, serve, and doctor targets' \
		'  ROOT=/path/to/project    Project root for make doctor' \
		'  HOST=127.0.0.1 PORT=8723 Host/port for make cbr-browser/serve' \
		'  SHELL_IMAGE=python:3.13  Docker image for make chat-safe-shell' \
		'  SHELL_TIMEOUT=60         Per-command timeout for make chat-safe-shell' \
		'  CBR_OLLAMA_PORT=11435    Host loopback port for containerized Ollama'

venv: $(VENV_PY)

$(VENV_PY):
	$(PYTHON) -m venv "$(VENV)"

require-cli: venv
	@test -x "$(CLI)" || { printf '%s\n' 'Missing $(CLI). Run: make install'; exit 1; }

install: venv
	"$(VENV_PY)" -m pip install -e "$(INSTALL_TARGET)"

install-dev: venv
	"$(VENV_PY)" -m pip install -e "$(DEV_INSTALL_TARGET)" $(TEST_DEPS)

hooks: venv
	"$(VENV_PY)" scripts/install_hooks.py

check: venv
	"$(VENV_PY)" scripts/check_quality.py

fix: venv
	"$(VENV_PY)" scripts/check_quality.py --fix

test: venv
	@"$(VENV_PY)" -c 'import pytest' >/dev/null 2>&1 || { printf '%s\n' 'pytest is required. Run: make install-dev'; exit 1; }
	"$(VENV_PY)" -m pytest tests

smoke: require-cli
	"$(CLI)" --help >/dev/null
	"$(CLI)" chat --help >/dev/null
	"$(CLI)" serve --help >/dev/null
	"$(CLI)" browser --help >/dev/null
	@printf '%s\n' 'CLI smoke: OK'

doctor: require-cli
	"$(CLI)" doctor --root "$(ROOT)" --model "$(MODEL)"

chat: require-cli
	"$(CLI)" chat --model "$(MODEL)"

chat-tui: require-cli
	"$(CLI)" chat --model "$(MODEL)" --tui

cbr-browser: require-cli
	"$(CLI)" browser --host "$(HOST)" --port "$(PORT)" --model "$(MODEL)"

serve: require-cli
	"$(CLI)" serve --host "$(HOST)" --port "$(PORT)" --model "$(MODEL)"

install-global: install
	mkdir -p "$(BIN_DIR)"
	printf '%s\n' '#!/bin/sh' 'exec "$(abspath $(VENV_PY))" -m codebase_rag "$$@"' > "$(BIN_DIR)/$(COMMAND)"
	chmod +x "$(BIN_DIR)/$(COMMAND)"
	@printf 'Installed %s\n' "$(BIN_DIR)/$(COMMAND)"
	@case ":$$PATH:" in \
		*:"$(BIN_DIR)":*) ;; \
		*) printf 'Add %s to PATH to run %s from any directory.\n' "$(BIN_DIR)" "$(COMMAND)" ;; \
	esac

uninstall-global:
	rm -f "$(BIN_DIR)/$(COMMAND)"
	@printf 'Removed %s\n' "$(BIN_DIR)/$(COMMAND)"

chat-safe-shell: require-cli
	"$(CLI)" chat --model "$(MODEL)" --allow-shell --shell-runner "docker:$(SHELL_IMAGE)" --shell-network none --shell-timeout "$(SHELL_TIMEOUT)"

container-build:
	CBR_OLLAMA_PORT="$(CBR_OLLAMA_PORT)" docker compose build

container-up:
	CBR_OLLAMA_PORT="$(CBR_OLLAMA_PORT)" docker compose up -d ollama

container-pull-model: container-up
	CBR_OLLAMA_PORT="$(CBR_OLLAMA_PORT)" docker compose exec ollama ollama pull "$(MODEL)"

container-doctor:
	CBR_PROJECT_ROOT="$(abspath $(ROOT))" CBR_OLLAMA_PORT="$(CBR_OLLAMA_PORT)" docker compose run --rm codebase-rag doctor --root /work --model "$(MODEL)"

container-index:
	CBR_PROJECT_ROOT="$(abspath $(ROOT))" CBR_OLLAMA_PORT="$(CBR_OLLAMA_PORT)" docker compose run --rm codebase-rag index /work --db /data/codebase-rag/db

container-chat:
	CBR_PROJECT_ROOT="$(abspath $(ROOT))" CBR_OLLAMA_PORT="$(CBR_OLLAMA_PORT)" docker compose run --rm -it codebase-rag chat --root /work --db /data/codebase-rag/db --model "$(MODEL)" --read-only
