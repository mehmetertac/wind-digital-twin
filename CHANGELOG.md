# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CARE-to-Compare: Zenodo v6 WF-A downloader, vendored `CAREScore`, channel map, loader, hybrid/ablation scoring scripts, and tests.
- `requirements.lock` (hashed universal lock for base + `dev` + `serve` + `export`).
- CI **repro** job: clean clone, wheels-only `pip install`, import smoke on `ubuntu-latest` and `windows-latest`.

### Changed

- `requires-python` pinned to `>=3.11,<3.13`; README quickstart documents wheels-only install.
- Cap `scipy`, pin `scikit-learn>=1.9` for committed joblib artifacts, and cap `lightgbm<4.6` (4.7.0 native predict fails on Windows in CI/repro).

## [1.0.1] - 2026-11-05

### Fixed

- CI: pin `pandas<3` so `export_models.py` parquet writes work with `pyarrow` on GitHub Actions.

### Added

- `CHANGELOG.md` and portfolio release polish (demo GIF refresh, GitHub release notes).

## [1.0.0] - 2026-10-17

### Added

- Hybrid drivetrain digital twin: lumped thermal ODE, learned residual, hybrid anomaly detection.
- Week 11 **FastAPI + Streamlit** serving with P10 SHAP attributions in API and dashboard.
- Capstone README v1.0.0: stack diagram, RMSE table, RUL narrative, limitations.
- End-to-end twin pass on synthetic EDP; headline figures under `docs/figures/`.
- Docker service stack and GitHub Actions CI.

[Unreleased]: https://github.com/mehmetertac/wind-digital-twin/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/mehmetertac/wind-digital-twin/releases/tag/v1.0.1
[1.0.0]: https://github.com/mehmetertac/wind-digital-twin/releases/tag/v1.0.0
