.PHONY: help init hooks fmt lint type test cov sec import-graph ingest plan gates smoke

PYTHON := python3
VENV := venv
PYTHONPATH := /Users/zachwade/StockMonitoring

help:
	@echo "Targets:"
	@echo "  init          - create venv and install dev deps"
	@echo "  hooks         - install pre-commit hooks"
	@echo "  fmt           - run black + ruff --fix"
	@echo "  lint          - run ruff (lint only)"
	@echo "  type          - run mypy"
	@echo "  test          - run pytest"
	@echo "  cov           - run tests with coverage report"
	@echo "  sec           - run bandit + pip-audit + safety"
	@echo "  gates         - run all quality gates"
	@echo "  import-graph  - build import graph (dot + json)"
	@echo "  ingest        - ingest review docs to issues.json"
	@echo "  plan          - produce wave_plan.md from issues.json"

init:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements-dev.txt

hooks:
	. $(VENV)/bin/activate && pre-commit install || true

fmt:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python -m ruff check src tests --fix && python -m black src tests

lint:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python -m ruff check src tests

type:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python -m mypy src tests

test:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python -m pytest

cov:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python -m pytest --cov

sec:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python -m bandit -r src tests || true
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python -m pip_audit -r requirements-dev.txt || true
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python -m safety check -r requirements-dev.txt || true

gates: fmt lint type test sec

import-graph:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python scripts/build_import_graph.py --root . --out graph

ingest:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python scripts/ingest_issues.py reviews --out issues.json

plan:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python scripts/wave_planner.py issues.json graph/import_graph.json --out wave_plan.md

smoke:
	. $(VENV)/bin/activate && PYTHONPATH=$(PYTHONPATH) python scripts/run_smoke_paper.py
