# Standard library imports
import pathlib
import subprocess
import sys


def test_paper_smoke():
    script = pathlib.Path("scripts/run_smoke_paper.py")
    assert script.exists(), "Missing smoke harness"
    # Using subprocess with explicit executable and arguments is safe
    res = subprocess.run(  # noqa: S603
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, res.stderr
    assert "SMOKE" in res.stdout
