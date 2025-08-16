# Trading System Remediation Runbook

This runbook walks you end-to-end through standing up quality gates, ingesting review docs,
planning waves, restoring an MTP (Minimal Trading Path), and shipping fixes safely.

## Prereqs
- macOS with Homebrew
- GitHub repo + VS Code
- Python 3.11 recommended
- gh CLI (optional): `brew install gh`

## 0) Bootstrap this kit
```bash
unzip trading_remediation_bootstrap.zip -d your-repo/
cd your-repo
make init
```

## 1) Branch protection & CODEOWNERS
- In GitHub: Settings → Branches → Add rule for `main`:
  - Require PR, require status checks pass, require reviews.
- Commit CODEOWNERS (edit handles).

## 2) Pre-commit hooks
```bash
make hooks
```

## 3) CI
- Push branch; confirm GitHub Actions run `.github/workflows/ci.yml`.
- Ensure checks are required on main.

## 4) VS Code setup
- Open folder; allow recommended settings. Interpreter at `.venv/bin/python`.

## 5) Import graph
```bash
make import-graph
```
Outputs `graph/import_graph.json` and optionally `.dot`.

## 6) Ingest review docs
- Place your 50+ docs (txt/md) into `reviews/` at repo root.
```bash
make ingest
```
Produces `issues.json` (deduped).

## 7) Plan waves
```bash
make plan
```
Produces `wave_plan.md` with Wave 0/1/2 summaries. Edit as needed.

## 8) Smoke harness
- Edit `scripts/run_smoke_paper.py` to wire your real data/strategy/order router.
- Ensure `pytest` passes:
```bash
make test
```

## 9) Quality gates (local)
```bash
make gates
```

## 10) First PRs (Wave 0)
- Create a branch, fix a small coherent set, add tests, run `make gates`, open PR.
- Use PR template checklist fully.

## 11) Paper → Canary → Live
- Paper: run smoke hourly; validate logs/metrics.
- Canary: deploy with tiny sizes and kill-switch; monitor.
- Live: ramp cautiously; never disable alerts.

## 12) Docs & ADRs
- For contract/interface changes, copy `docs/ADR/template.md` and fill.
- Keep module READMEs updated.

## 13) Weekly Release Train
- Merge green PRs; tag release; write brief notes (changes, risks, mitigations).

## 14) Long tail remediation
- Continue Waves 1–2 in small PRs with tests and flags.
