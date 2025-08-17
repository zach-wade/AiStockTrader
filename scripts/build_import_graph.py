#!/usr/bin/env python3
# Standard library imports
import argparse
import ast
import json
from pathlib import Path

# Third-party imports
import networkx as nx


def find_py_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part in {".venv", "venv", "build", "dist", ".git"} for part in p.parts):
            continue
        yield p


def module_name(root: Path, file: Path) -> str:
    rel = file.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def build_graph(root: Path) -> nx.DiGraph:
    g = nx.DiGraph()
    files = list(find_py_files(root))
    modules = {f: module_name(root, f) for f in files}
    for f in files:
        mod = modules[f]
        g.add_node(mod, path=str(f))
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    g.add_edge(mod, n.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    g.add_edge(mod, node.module.split(".")[0])
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--out", type=str, default="graph")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = build_graph(root)
    data = nx.readwrite.json_graph.node_link_data(g)
    (out / "import_graph.json").write_text(json.dumps(data, indent=2))
    try:
        dot = "digraph G { rankdir=LR; node [shape=box,fontsize=10];\n"
        for n in g.nodes():
            dot += f'"{n}" ;\n'
        for u, v in g.edges():
            dot += f'"{u}" -> "{v}";\n'
        dot += "}\n"
        (out / "import_graph.dot").write_text(dot)
    except Exception:
        pass
    print(f"Wrote {out/'import_graph.json'} (and .dot if graphviz available)")
