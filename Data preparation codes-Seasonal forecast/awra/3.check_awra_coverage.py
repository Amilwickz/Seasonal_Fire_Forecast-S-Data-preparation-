"""
check_awra_coverage.py
======================
Checks the last available month in:
  1. Downloaded raw AWRA-L NetCDF files  (0.raw_nc/<var>/<var>_2026.nc)
  2. Processed monthly cube files        (2.monthly_cubes/AWRA_<VAR>_monthly_500m.nc)
"""

import xarray as xr
from pathlib import Path

RAW_DIR      = Path("/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/13.AWRA-L/0.raw_nc")
PROCESSED_DIR = Path("/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/13.AWRA-L/2.monthly_cubes")
VARIABLES    = ["s0", "sd", "ss", "qtot", "etot", "e0", "dd"]

print(f"\n{'='*60}")
print(f"  AWRA-L Coverage Check")
print(f"{'='*60}\n")

# ── 1. Raw downloaded files ───────────────────────────────────────
print("[ Downloaded raw files — last time step in 2026 file ]\n")
for var in VARIABLES:
    nc_path = RAW_DIR / var / f"{var}_2026.nc"
    if not nc_path.exists():
        print(f"  {var:<6}  NOT FOUND : {nc_path}")
        continue
    try:
        ds = xr.open_dataset(str(nc_path))
        last_time = str(ds["time"].values[-1])[:10]   # YYYY-MM-DD
        ds.close()
        print(f"  {var:<6}  last date : {last_time}")
    except Exception as e:
        print(f"  {var:<6}  ERROR : {e}")

# ── 2. Processed monthly cube files ──────────────────────────────
print(f"\n[ Processed monthly cubes — last band ]\n")
for var in VARIABLES:
    cube_path = PROCESSED_DIR / f"AWRA_{var.upper()}_monthly_500m.nc"
    if not cube_path.exists():
        print(f"  {var:<6}  NOT FOUND : {cube_path}")
        continue
    try:
        ds = xr.open_dataset(str(cube_path))
        last_band = str(ds["band"].values[-1])   # e.g. '2026_04_01'
        n_bands   = len(ds["band"])
        ds.close()
        print(f"  {var:<6}  last band : {last_band}   ({n_bands} months total)")
    except Exception as e:
        print(f"  {var:<6}  ERROR : {e}")

print(f"\n{'='*60}\n")
