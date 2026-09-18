"""Download official DriveLM-nuScenes v1.1 mirrors and verify against HF metadata.

Requires gdown. No Hugging Face token is needed for the public Google Drive
mirrors linked by the authors in challenge/README.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import urllib.request
import zipfile

import gdown


FILES = {
    "v1_1_train_nus.json": "1CvTPwChKvfnvrZ1Wr0ZNVqtibkkNeGgt",
    "v1_1_val_nus_q_only.json": "1fsVP7jOpvChcpoXVdypaZ4HREX1gA7As",
    "drivelm_nus_imgs_train.zip": "1DeosPGYeM2gXSChjMODGsQChZyYDmaUz",
    "drivelm_nus_imgs_val.zip": "18f8ygNxGZWat-crUjroYuQbd39Sk9xCo",
}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archives = root / "archives"
    archives.mkdir(exist_ok=True)
    metadata_url = "https://huggingface.co/api/datasets/OpenDriveLab/DriveLM/tree/main"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = {item["path"]: item for item in json.load(response)}
    manifest = {}
    for name, file_id in FILES.items():
        info = metadata[name]
        path = (archives if name.endswith(".zip") else root) / name
        expected_hash = info.get("lfs", {}).get("oid")
        # HF hides LFS hashes behind asterisks for accounts without gate access.
        # Still validate the public file size, parse JSON, and check ZIP CRCs;
        # record our SHA-256 without claiming an official checksum comparison.
        if expected_hash and not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            expected_hash = None
        url = "https://drive.google.com/uc?id=" + file_id
        print(f"Downloading {name} ({info['size'] / 1e9:.3f} GB)", flush=True)
        if not path.exists() or path.stat().st_size != info["size"]:
            result = gdown.download(url, str(path), quiet=True, resume=True, use_cookies=False)
            if result is None:
                raise RuntimeError(f"Download failed: {name}")
        if path.stat().st_size != info["size"]:
            raise ValueError(f"Size differs from official metadata: {name}")
        actual_hash = digest(path)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"SHA-256 differs from official metadata: {name}")
        if name.endswith(".json"):
            with path.open() as stream:
                json.load(stream)
        manifest[name] = {
            "source": url,
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
            "official_sha256_verified": bool(expected_hash),
        }
        (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Verified {name}", flush=True)

    for name in FILES:
        if not name.endswith(".zip"):
            continue
        print(f"Extracting {name}", flush=True)
        with zipfile.ZipFile(archives / name) as archive:
            for member in archive.infolist():
                relative = Path(member.filename)
                # Official val ZIP uses val_data/CAM_*, while its annotations
                # refer to nuscenes/samples/CAM_* like the training split.
                if relative.parts and relative.parts[0] == "val_data":
                    relative = Path("nuscenes/samples", *relative.parts[1:])
                target = (root / relative).resolve()
                if root not in target.parents:
                    raise ValueError(f"Unsafe ZIP path: {member.filename}")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"ZIP symlink is unsupported: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        print(f"Extracted {name} (ZIP CRC verified while reading)", flush=True)
    print(f"Dataset ready at {root}", flush=True)


if __name__ == "__main__":
    main()
