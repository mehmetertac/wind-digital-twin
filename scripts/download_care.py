#!/usr/bin/env python3
"""Download CARE-to-Compare (Zenodo) — Wind Farm A via HTTP range into a zip."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_digital_twin.config import (  # noqa: E402
    CARE_WFA_EXPECTED_DATASETS,
    CARE_ZENODO_RECORD,
    CARE_ZIP_MD5,
    CARE_ZIP_URL,
    DATA_RAW_CARE_WFA,
)

MANUAL_STEPS = """
CARE-to-Compare (Wind Farm A = real EDP Portugal)
=================================================
Zenodo record v6: https://zenodo.org/records/{record}

python scripts/download_care.py --check
python scripts/download_care.py --wind-farm A

Data lands in: {out_dir}

Wind Farm A: 22 event CSVs, event_info.csv, feature_description.csv (~0.8 GB extracted).
Paper: arXiv:2404.10320 — CARE score baselines (AE ~0.66, random 0.5, IF below random over 95 events).

Mendeley filtered EDP (second route): https://data.mendeley.com/datasets/zjxjnjp3xs/2
"""


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable stream over HTTP Range requests."""

    def __init__(self, url: str, size: int | None = None) -> None:
        super().__init__()
        self.url = url
        self._pos = 0
        self._size = size
        if self._size is None:
            self._size = self._probe_size()

    def _probe_size(self) -> int:
        req = Request(self.url, method="HEAD")
        with urlopen(req, timeout=120) as resp:
            length = resp.headers.get("Content-Length")
            if not length:
                raise RuntimeError("Server did not return Content-Length for CARE zip")
            return int(length)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = (self._size or 0) + offset
        else:
            raise ValueError(whence)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._size is not None and self._pos >= self._size:
            return b""
        if size < 0:
            size = (self._size or 0) - self._pos
        end = self._pos + size - 1
        if self._size is not None:
            end = min(end, self._size - 1)
        req = Request(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urlopen(req, timeout=300) as resp:
            chunk = resp.read()
        self._pos += len(chunk)
        return chunk


def _zip_prefix_for_farm(farm: str) -> str:
    if farm.upper() == "A":
        return "CARE_To_Compare/Wind Farm A/"
    if farm.upper() == "B":
        return "CARE_To_Compare/Wind Farm B/"
    if farm.upper() == "C":
        return "CARE_To_Compare/Wind Farm C/"
    raise ValueError(f"Unknown wind farm {farm}")


def extract_wind_farm(
    out_dir: Path,
    farm: str,
    *,
    url: str = CARE_ZIP_URL,
    use_local_zip: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _zip_prefix_for_farm(farm)

    if use_local_zip:
        zf = zipfile.ZipFile(use_local_zip)
        stream = None
    else:
        stream = HttpRangeFile(url)
        zf = zipfile.ZipFile(stream)

    members = [n for n in zf.namelist() if n.startswith(prefix) and not n.endswith("/")]
    manifest: dict = {
        "zenodo_record": CARE_ZENODO_RECORD,
        "zip_url": url,
        "prefix": prefix,
        "members": [],
    }
    try:
        for name in members:
            info = zf.getinfo(name)
            rel = name[len(prefix) :]
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            manifest["members"].append(
                {
                    "archive_path": name,
                    "file": str(rel),
                    "size": info.file_size,
                    "crc32": info.CRC,
                }
            )
    finally:
        zf.close()
        if stream is not None:
            stream.close()

    manifest_path = out_dir / ".care_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def check_wfa(out_dir: Path) -> bool:
    datasets = _datasets_dir(out_dir)
    ok = True
    for name in ("event_info.csv", "feature_description.csv"):
        p = datasets / name if (datasets / name).exists() else out_dir / name
        if p.exists():
            print(f"  OK  {p.name}")
        else:
            print(f"  MISSING  {name}")
            ok = False
    csv_count = len([p for p in datasets.glob("*.csv") if p.stem.isdigit()])
    if csv_count >= CARE_WFA_EXPECTED_DATASETS:
        print(f"  OK  {csv_count} event CSVs")
    else:
        print(f"  MISSING  event CSVs ({csv_count}/{CARE_WFA_EXPECTED_DATASETS})")
        ok = False
    manifest = out_dir / ".care_manifest.json"
    if manifest.exists():
        print(f"  OK  {manifest.name}")
    return ok


def _datasets_dir(out_dir: Path) -> Path:
    d = out_dir / "datasets"
    return d if d.is_dir() else out_dir


def verify_manifest_md5(manifest: dict) -> None:
    """Optional: user can md5 full zip after --full download."""
    if manifest.get("zip_md5") and manifest["zip_md5"] != CARE_ZIP_MD5:
        print(f"  WARN  zip MD5 {manifest['zip_md5']} != expected {CARE_ZIP_MD5}")


def download_full_zip(dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5()
    req = Request(CARE_ZIP_URL)
    with urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    digest = h.hexdigest()
    if digest != CARE_ZIP_MD5:
        print(f"  WARN  downloaded MD5 {digest} != catalog {CARE_ZIP_MD5}")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_RAW_CARE_WFA)
    parser.add_argument("--wind-farm", default="A", help="A (default), B, or C")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--instructions", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download full zip to data/raw/care/CARE_To_Compare.zip then extract",
    )
    parser.add_argument(
        "--from-zip",
        type=Path,
        help="Extract wind farm from an existing local CARE_To_Compare.zip",
    )
    args = parser.parse_args()

    if args.instructions:
        print(
            MANUAL_STEPS.format(
                record=CARE_ZENODO_RECORD,
                out_dir=args.out_dir,
            )
        )
        return 0

    if args.check:
        print(f"Checking CARE WF-A in {args.out_dir}...")
        return 0 if check_wfa(args.out_dir) else 1

    farm = args.wind_farm.upper()
    if args.full:
        zip_path = PROJECT_ROOT / "data" / "raw" / "care" / "CARE_To_Compare.zip"
        if not zip_path.exists():
            print(f"Downloading full zip to {zip_path} (several GB)...")
            try:
                download_full_zip(zip_path)
            except (HTTPError, URLError) as exc:
                print(f"Download failed: {exc}")
                return 1
        manifest = extract_wind_farm(args.out_dir, farm, use_local_zip=zip_path)
        manifest["zip_md5"] = CARE_ZIP_MD5
        (args.out_dir / ".care_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    elif args.from_zip:
        if not args.from_zip.exists():
            print(f"Missing zip: {args.from_zip}")
            return 1
        extract_wind_farm(args.out_dir, farm, use_local_zip=args.from_zip)
    else:
        print(f"Extracting Wind Farm {farm} from Zenodo (HTTP range)...")
        try:
            extract_wind_farm(args.out_dir, farm)
        except (HTTPError, URLError, zipfile.BadZipFile, RuntimeError) as exc:
            print(f"Range extract failed: {exc}")
            print("Try: python scripts/download_care.py --full")
            return 1

    ok = check_wfa(args.out_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())