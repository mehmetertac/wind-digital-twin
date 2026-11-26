"""Tests for CARE zip extraction helpers."""

from __future__ import annotations

import io
import sys
import tempfile
import threading
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_care import HttpRangeFile, extract_wind_farm  # noqa: E402


def _make_mini_care_zip(path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CARE_To_Compare/Wind Farm A/event_info.csv", "asset;event_id;event_label\n")
        zf.writestr("CARE_To_Compare/Wind Farm A/feature_description.csv", "sensor_name;statistics_type\n")
        zf.writestr("CARE_To_Compare/Wind Farm A/datasets/1.csv", "time_stamp;train_test\n")
    path.write_bytes(buf.getvalue())


def test_extract_wind_farm_from_local_zip():
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "care.zip"
        out = Path(tmp) / "out"
        _make_mini_care_zip(zpath)
        manifest = extract_wind_farm(out, "A", use_local_zip=zpath)
        assert (out / "event_info.csv").exists()
        assert len(manifest["members"]) == 3


def test_http_range_file_byte_ranges():
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "care.zip"
        _make_mini_care_zip(zpath)
        data = zpath.read_bytes()
        size = len(data)

        class RangeHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    part = rng.replace("bytes=", "").split("-")
                    start = int(part[0])
                    end = int(part[1]) if part[1] else size - 1
                    chunk = data[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(size))
                    self.end_headers()
                    self.wfile.write(data)

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{port}/care.zip"
            stream = HttpRangeFile(url, size=size)
            with zipfile.ZipFile(stream) as zf:
                names = zf.namelist()
            assert any("Wind Farm A" in n for n in names)
        finally:
            server.shutdown()