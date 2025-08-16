.PHONY: help init hooks fmt lint type test cov sec import-graph ingest plan gates smoke

PYTHON := python3

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
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements-dev.txt

hooks:
	. .venv/bin/activate && pre-commit install || true

fmt:
	. .venv/bin/activate && python -m ruff check . --fix && python -m black .

lint:
	. .venv/bin/activate && python -m ruff check .

type:
	. .venv/bin/activate && python -m mypy .

test:
	. .venv/bin/activate && python -m pytest

cov:
	. .venv/bin/activate && python -m pytest --cov

sec:
	. .venv/bin/activate && python -m bandit -r . || true
	. .venv/bin/activate && python -m pip_audit -r requirements-dev.txt || true
	. .venv/bin/activate && python -m safety check -r requirements-dev.txt || true

gates: fmt lint type test sec

import-graph:
	. .venv/bin/activate && python scripts/build_import_graph.py --root . --out graph

ingest:
	. .venv/bin/activate && python scripts/ingest_issues.py reviews --out issues.json

plan:
	. .venv/bin/activate && python scripts/wave_planner.py issues.json graph/import_graph.json --out wave_plan.md
