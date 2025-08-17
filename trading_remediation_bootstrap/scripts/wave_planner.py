#!/usr/bin/env python3
# Standard library imports
import argparse
import json
import pathlib
from typing import Any


def mtp_score(issue: dict[str, Any]) -> int:
    sev = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}.get(issue["severity"], 1)
    mtb = {"High": 3, "Medium": 2, "Low": 1}.get(issue.get("mtb_relevance", "Medium"), 2)
    risk = {"High": 1, "Medium": 2, "Low": 3}.get(issue.get("risk", "Medium"), 2)
    return sev * 10 + mtb * 3 + risk


def group_by_module(issues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for it in issues:
        mod = (it.get("file") or "unknown").split("/")[0]
        groups.setdefault(mod, []).append(it)
    return groups


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("issues_json")
    ap.add_argument("graph_json")
    ap.add_argument("--out", default="wave_plan.md")
    args = ap.parse_args()
    issues = json.loads(pathlib.Path(args.issues_json).read_text())["issues"]
    # simple sort
    issues.sort(key=mtp_score, reverse=True)
    groups = group_by_module(issues)

    lines = []
    lines.append("# Wave Plan\n")
    lines.append("## Wave 0 — MTP Blockers\n")
    lines.append("| # | File | Severity | Summary |\n|---|---|---|---|\n")
    for i, it in enumerate(issues[:30], 1):
        lines.append(
            f"| {i} | {it.get('file','')} | {it['severity']} | {it['summary'].replace('|','/')} |\n"
        )
    lines.append("\n## Wave 1 — Stability & Observability (next 50)\n")
    lines.append("| # | File | Severity | Summary |\n|---|---|---|---|\n")
    for i, it in enumerate(issues[30:80], 31):
        lines.append(
            f"| {i} | {it.get('file','')} | {it['severity']} | {it['summary'].replace('|','/')} |\n"
        )
    lines.append("\n## Wave 2+ — Long Tail\n")
    lines.append("Remaining issues grouped by top-level module:\n")
    for mod, lst in groups.items():
        lines.append(f"- **{mod}**: {len(lst)} issues\n")

    pathlib.Path(args.out).write_text("".join(lines))
    print(f"Wrote {args.out}")
