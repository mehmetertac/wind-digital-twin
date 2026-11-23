"""Reproducibility contract: Python pin and hashed lock pins for key wheels."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.lock"
PYPROJECT = ROOT / "pyproject.toml"

LOCKED_PACKAGES = ("numpy", "pandas", "pyarrow", "lightgbm", "shap")


def _pinned_versions(lock_text: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lock_text.splitlines():
        m = re.match(r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)==([^\s\\]+)", line)
        if m:
            versions[m.group(1).lower()] = m.group(2)
    return versions


def test_requires_python_pin():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.11,<3.13"


def test_lockfile_pins_core_wheels():
    assert LOCK.is_file(), "requirements.lock missing"
    text = LOCK.read_text(encoding="utf-8")
    assert "--hash=sha256:" in text
    pinned = _pinned_versions(text)
    for name in LOCKED_PACKAGES:
        assert name in pinned, f"{name} not pinned with ==" in LOCK
        assert pinned[name], f"empty version for {name}"