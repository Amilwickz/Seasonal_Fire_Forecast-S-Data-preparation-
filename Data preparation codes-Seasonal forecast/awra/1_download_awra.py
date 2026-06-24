"""
1_download_awra.py
==================
Downloads all AWRA-L v7 NetCDF files (7 variables x 27 years = 189 files)
from NCI THREDDS to local scratch storage.

Run this on a Gadi LOGIN NODE (has internet access) or via copyq PBS job.
DO NOT run on normal compute nodes (no internet).

Usage:
    python 1_download_awra.py

Files saved to:
    /scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/13.AWRA-L/0.raw_nc/
"""

import os
import requests
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

THREDDS_BASE = (
    "https://thredds.nci.org.au/thredds/fileServer/iu04/"
    "australian-water-outlook/historical/v1/AWRALv7"
)

DOWNLOAD_DIR = (
    "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/"
    "13.AWRA-L/0.raw_nc"
)

VARIABLES  = ["s0", "sd", "ss", "qtot", "etot", "e0", "dd"]
YEAR_START = 2000
YEAR_END   = 2026

CHUNK_SIZE = 1024 * 1024  # 1 MB read chunks

# ── Download function ─────────────────────────────────────────────────────────

def download_file(url: str, dest_path: Path) -> bool:
    """
    Download a file with resume support.
    If dest_path already exists and is the correct size, skip it.
    Returns True if downloaded or already complete, False on error.
    """
    # Check if file already fully downloaded via Content-Length header
    try:
        head = requests.head(url, timeout=30)
        remote_size = int(head.headers.get('Content-Length', -1))
    except Exception as e:
        print(f"    HEAD request failed: {e}")
        remote_size = -1

    if dest_path.exists():
        local_size = dest_path.stat().st_size
        if remote_size > 0 and local_size == remote_size:
            print(f"    Already complete ({local_size / 1e6:.1f} MB) — skipping.")
            return True
        elif local_size > 0:
            print(f"    Partial file found ({local_size / 1e6:.1f} MB) — re-downloading.")
            dest_path.unlink()

    # Stream download
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get('Content-Length', 0))
            downloaded = 0
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            size_mb = downloaded / 1e6
            print(f"    Downloaded {size_mb:.1f} MB  → {dest_path.name}")
            return True
    except Exception as e:
        print(f"    ERROR downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # remove partial file
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    download_dir = Path(DOWNLOAD_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Build full file list
    tasks = []
    for var in VARIABLES:
        var_dir = download_dir / var
        var_dir.mkdir(exist_ok=True)
        for year in range(YEAR_START, YEAR_END + 1):
            filename = f"{var}_{year}.nc"
            url      = f"{THREDDS_BASE}/{filename}"
            dest     = var_dir / filename
            tasks.append((var, year, url, dest))

    total  = len(tasks)
    done   = 0
    failed = []

    print(f"{'='*60}")
    print(f"  AWRA-L v7 Download")
    print(f"  Files to download : {total}")
    print(f"  Destination       : {DOWNLOAD_DIR}")
    print(f"{'='*60}\n")

    for var, year, url, dest in tasks:
        done += 1
        print(f"[{done:>3}/{total}]  {var}_{year}.nc")
        success = download_file(url, dest)
        if not success:
            failed.append(f"{var}_{year}.nc")

    print(f"\n{'='*60}")
    print(f"  Download complete")
    print(f"  Successful : {total - len(failed)}/{total}")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for f in failed:
            print(f"    {f}")
        print(f"\n  Re-run this script to retry failed files.")
    else:
        print(f"  All files downloaded successfully.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
