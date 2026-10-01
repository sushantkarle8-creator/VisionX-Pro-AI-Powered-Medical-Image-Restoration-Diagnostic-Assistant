"""Download official weights atomically and record SHA-256 for corruption detection."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from visionx.models import CHECKPOINTS, DEFAULT_WEIGHTS, UPSTREAM_COMMIT


def download(method, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = CHECKPOINTS[method]
    url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/" + name
    destination = directory / name
    temporary = destination.with_suffix(".part")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "VisionX-Pro/2.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            expected = int(response.headers.get("Content-Length", 0))
            size = 0
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size < 1_000_000 or (expected and size != expected):
            raise ValueError("Incomplete checkpoint download.")
        temporary.replace(destination)
        record = {"url": url, "bytes": size, "sha256": digest.hexdigest(),
                  "architecture_commit": UPSTREAM_COMMIT,
                  "checksum_scope": "Locally recorded after HTTPS download; not an independent publisher signature."}
        destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Downloaded {method}: {size:,} bytes; SHA-256 {digest.hexdigest()}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*CHECKPOINTS, "all"], default="swinir_denoise")
    parser.add_argument("--directory", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    for selected in CHECKPOINTS if args.model == "all" else [args.model]:
        download(selected, args.directory)
