"""
Downloading Global Wheat Head Dataset 2020 (official Zenodo release).
ZIP contains images + CSV annotations (not Parquet).
"""
import os
import zipfile
from urllib.request import Request, urlopen

# Direct Zenodo file link (GWHD 2020 codalab official)
ZENODO_URL = (
    "https://zenodo.org/records/4298502/files/global-wheat-codalab-official.zip"
)
TARGET_DIR = "data/raw"
ZIP_NAME = "global-wheat-codalab-official.zip"
# Extract into a subfolder so data/raw stays organized
EXTRACT_DIR = os.path.join(TARGET_DIR, "gwhd2020")

CHUNK_BYTES = 1024 * 1024  # 1 MiB


def download_file(url: str, dest_path: str) -> None:
    """Stream download to disk with progress (MB written)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Agriculture-Decision-System/1.0)"},
    )
    with urlopen(req) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        with open(dest_path, "wb") as out:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = min(100, int(100 * downloaded / total))
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    print(f"\r  Progress: {mb_done:.1f} / {mb_total:.1f} MB ({pct}%)", end="")
                else:
                    print(f"\r  Downloaded: {downloaded / (1024 * 1024):.1f} MB", end="")
    print()


def main() -> None:
    os.makedirs(TARGET_DIR, exist_ok=True)
    zip_path = os.path.join(TARGET_DIR, ZIP_NAME)

    print(f"Starting download from Zenodo…")
    print(f"  URL: {ZENODO_URL}")
    print(f"  Save: {zip_path}")
    download_file(ZENODO_URL, zip_path)

    print(f"Extracting to {EXTRACT_DIR}…")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)

    print(f"\nDone. Dataset is under: {EXTRACT_DIR}")
    print(f"ZIP kept at: {zip_path} (delete manually if you need disk space.)")


if __name__ == "__main__":
    main()
