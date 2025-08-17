#!/usr/bin/env python3
# Standard library imports
import argparse
import json
from pathlib import Path
import re
from typing import Any

ISSUE_RE = re.compile(r"(?i)severity\s*:\s*(critical|high|medium|low)")
FILE_RE = re.compile(r"(?i)(file|path)\s*:\s*([\w./\\-]+)")
LINE_RE = re.compile(r"(?i)(line|lines)\s*:\s*(\d+)(?:\D+(\d+))?")
CAT_RE = re.compile(r"(?i)(category)\s*:\s*([\w-]+)")


def parse_text(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    chunks = re.split(r"\n\s*[-*#]+\s*|\n\s*\n", text)
    for idx, ch in enumerate(chunks):
        ch = ch.strip()
        if not ch or len(ch) < 10:
            continue
        sev_m = ISSUE_RE.search(ch)
        sev = sev_m.group(1).capitalize() if sev_m else "Low"
        file_m = FILE_RE.search(ch)
        path = file_m.group(2).strip() if file_m else ""
        line_m = LINE_RE.search(ch)
        ls, le = (
            (int(line_m.group(2)), int(line_m.group(3)))
            if line_m and line_m.group(3)
            else (int(line_m.group(2)), int(line_m.group(2))) if line_m else (0, 0)
        )
        cat_m = CAT_RE.search(ch)
        cat = cat_m.group(2).capitalize() if cat_m else "Bug"
        summary = ch.splitlines()[0][:200]
        items.append(
            {
                "id": f"docchunk-{idx}",
                "file": path,
                "line_start": ls,
                "line_end": le,
                "severity": sev,
                "category": cat,
                "summary": summary,
                "details": ch[:2000],
                "duplicates": [],
                "root_cause_hypothesis": "",
                "proposed_fix": "",
                "dependencies": [],
                "mtb_relevance": "Medium",
                "risk": "Medium",
                "status": "Todo",
                "owner": "",
                "test_plan": [],
                "acceptance_criteria": [],
                "pr_links": [],
            }
        )
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reviews_dir", type=str, help="Directory with .md/.txt files")
    ap.add_argument("--out", type=str, default="issues.json")
    args = ap.parse_args()
    reviews = Path(args.reviews_dir)
    all_items = []
    for p in reviews.rglob("*"):
        if p.suffix.lower() in {".md", ".txt"} and p.is_file():
            try:
                all_items.extend(parse_text(p.read_text(encoding="utf-8", errors="ignore")))
            except Exception as e:
                print(f"Failed to parse {p}: {e}")
    # naive dedupe by (file, line_start, summary)
    seen = set()
    deduped = []
    for it in all_items:
        key = (it["file"], it["line_start"], it["summary"][:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    Path(args.out).write_text(json.dumps({"issues": deduped}, indent=2))
    print(f"Wrote {args.out} with {len(deduped)} issues (from {len(all_items)} raw)")
