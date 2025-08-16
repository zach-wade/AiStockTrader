import subprocess, sys, pathlib

def test_paper_smoke():
    script = pathlib.Path("scripts/run_smoke_paper.py")
    assert script.exists(), "Missing smoke harness"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "SMOKE" in res.stdout
