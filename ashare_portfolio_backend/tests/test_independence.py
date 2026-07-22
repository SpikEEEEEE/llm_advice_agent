from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_source_and_metadata_have_no_legacy_project_dependency():
    root = Path(__file__).resolve().parents[1]
    forbidden = "stock" + "bench"
    files = [
        *root.joinpath("app").rglob("*.py"),
        root / "pyproject.toml",
        root / "requirements.txt",
        root / ".env.example",
    ]
    matches = [
        str(path.relative_to(root))
        for path in files
        if forbidden in path.read_text(encoding="utf-8").lower()
    ]
    assert matches == []


def test_application_imports_from_an_isolated_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        "from app.main import app; "
        "assert app.title == 'Independent A-share Portfolio Advisor'"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
