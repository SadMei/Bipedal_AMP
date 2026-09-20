"""Fetch only selected official 100STYLE files, using HTTP ZIP range requests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import requests


RECORD_URL = "https://zenodo.org/api/records/8127870"
MOVEMENTS = ("FW", "BW", "SW", "FR", "BR", "SR", "ID", "TR1")


class RemoteZipReader(io.RawIOBase):
    def __init__(self, url: str, size: int):
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()
        self.cache_start, self.cache = 0, b""
        self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = origin + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 64 * 1024**2:
            raise ValueError("Unexpected large ZIP read")
        start, stop = self.position, self.position + size
        if not (self.cache_start <= start and stop <= self.cache_start + len(self.cache)):
            request_stop = min(self.size, start + max(size, 65536))
            headers = {"Range": f"bytes={start}-{request_stop - 1}", "Accept-Encoding": "identity"}
            with self.session.get(self.url, headers=headers, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{request_stop - 1}/{self.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"Server did not honor requested range: {response.headers}")
                data = response.raw.read(request_stop - start + 1)
                if len(data) != request_stop - start:
                    raise RuntimeError("Incomplete or oversized range response")
            self.cache_start, self.cache = start, data
            self.transferred += len(data)
        result = self.cache[start - self.cache_start : stop - self.cache_start]
        self.position = stop
        return result

    def close(self):
        self.session.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    response = requests.get(RECORD_URL, timeout=(10, 30))
    response.raise_for_status()
    record = response.json()
    archive = next(item for item in record["files"] if item["key"] == "100STYLE.zip")
    requested = {f"Neutral_{movement}.bvh" for movement in MOVEMENTS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with RemoteZipReader(archive["links"]["self"], archive["size"]) as remote:
        with zipfile.ZipFile(remote) as source:
            selected = []
            for item in source.infolist():
                name = Path(item.filename).name
                if item.is_dir() or "__MACOSX" in item.filename or name.startswith("."):
                    continue
                if name in requested or name.lower() in {"frame_cuts.csv", "dataset_list.csv", "license.txt", "license"}:
                    selected.append(item)
            missing = requested - {Path(item.filename).name for item in selected}
            if missing:
                raise ValueError(f"Missing requested motions: {sorted(missing)}")
            priority = {f"Neutral_{movement}.bvh": index for index, movement in enumerate(MOVEMENTS)}
            selected.sort(key=lambda item: priority.get(Path(item.filename).name, -1))
            manifest_files = []
            for item in selected:
                target = args.output_dir / Path(item.filename).name
                if target.exists():
                    data = target.read_bytes()
                    if len(data) != item.file_size or zipfile.crc32(data) != item.CRC:
                        raise ValueError(f"Existing file differs from official archive: {target}")
                else:
                    print(f"[100STYLE] Fetching {target.name} ({item.compress_size / 1024**2:.2f} MiB compressed)", flush=True)
                    data = source.read(item)
                    with target.open("xb") as output:
                        output.write(data)
                manifest_files.append({"file": target.name, "archive_path": item.filename,
                                       "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                print(f"[100STYLE] {target.name}: {len(data):,} bytes", flush=True)
        manifest = {
            "source": "https://www.ianxmason.com/100style/", "record": RECORD_URL,
            "archive_url": archive["links"]["self"], "archive_checksum": archive.get("checksum"),
            "license": "CC BY 4.0", "credit": "The 100STYLE Dataset - Ian Mason",
            "citation": "Mason, Starke, Komura (2022), DOI:10.1145/3522618",
            "files": manifest_files, "downloaded_bytes_this_run": remote.transferred,
        }
        manifest_path = args.output_dir / "download_manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Downloaded {remote.transferred / 1024**2:.1f} MiB from the 1.37 GiB archive", flush=True)


if __name__ == "__main__":
    main()
