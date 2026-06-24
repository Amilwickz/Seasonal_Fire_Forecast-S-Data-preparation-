"""
2_Extend_ignitions_to_2026_04.py
=================================
Extends the existing Ignitions_monthly_005_Aus.nc (currently up to July 2024)
by appending monthly ignition data from August 2024 through April 2026.

Source GeoPackage:
  fired_au_2024_2026_05_2024_to_2026_events.gpkg  (Sinusoidal CRS)

Output:
  /scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/
  14.FireIgnitions/Ignitions_monthly_005_Aus_to_2026_04.nc

RAM strategy:
  - One month processed at a time (one 6800×9000 int32 slice ≈ 245 MB)
  - The existing NetCDF is opened read-only and copied slice-by-slice
  - No full arrays held in memory simultaneously

Run on GADI NCI:
  python 2_Extend_ignitions_to_2026_04.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.spatial import KDTree
from netCDF4 import Dataset, date2num
import xarray as xr
from datetime import datetime
import os

# =============================================================================
# Paths
# =============================================================================

BASE_DIR   = "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/14.FireIgnitions"

EXISTING_NC = os.path.join(BASE_DIR, "Ignitions_monthly_005_Aus.nc")

GPKG_PATH   = (
    "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/"
    "14.FireIgnitions/0.Raw_data/firedpy_au_2024_2026_05/"
    "firedpy_au_2024_2026_05/outputs/shapefiles/"
    "fired_au_2024_2026_05_2024_to_2026_events.gpkg"
)

OUTPUT_NC   = os.path.join(BASE_DIR, "Ignitions_monthly_005_Aus_to_2026_04.nc")

# The months we need to append (first month after existing data → April 2026)
# Adjust EXTEND_FROM if your existing file already includes some of 2024
EXTEND_FROM = pd.Timestamp("2024-08-01")   # first NEW month to append
EXTEND_TO   = pd.Timestamp("2026-04-01")   # last  NEW month to append (inclusive)

# =============================================================================
# Step 1 – Inspect the existing NetCDF to get grid and last timestamp
# =============================================================================

print("=" * 65)
print("Step 1: Reading existing NetCDF metadata ...")
print("=" * 65)

with Dataset(EXISTING_NC, "r") as nc_in:
    # Grid
    lats_grid = nc_in.variables["lat"][:]          # shape (6800,)
    lons_grid = nc_in.variables["lon"][:]          # shape (9000,)
    n_lat      = len(lats_grid)
    n_lon      = len(lons_grid)

    # Time axis
    time_var   = nc_in.variables["time"]
    time_units = time_var.units                     # e.g. 'days since 2000-01-01'
    time_cal   = getattr(time_var, "calendar", "gregorian")
    existing_times = time_var[:]                    # numeric values
    n_existing = len(existing_times)

    # Reconstruct datetime index for the existing file
    existing_dts = pd.to_datetime(
        [pd.Timestamp("2000-01-01") + pd.Timedelta(days=float(t))
         for t in existing_times]
    ).to_period("M").to_timestamp()

    last_existing = existing_dts.max()
    print(f"  Grid size       : {n_lat} lat × {n_lon} lon")
    print(f"  Existing months : {n_existing}  ({existing_dts.min().date()} → {last_existing.date()})")
    print(f"  Time units      : {time_units}")

# =============================================================================
# Step 2 – Build the list of new months to append
# =============================================================================

print("\nStep 2: Building new month list ...")

new_months = pd.date_range(start=EXTEND_FROM, end=EXTEND_TO, freq="MS")
print(f"  Months to append: {len(new_months)}")
print(f"  {new_months[0].date()}  →  {new_months[-1].date()}")

# =============================================================================
# Step 3 – Load and prepare the GeoPackage (once, outside the month loop)
# =============================================================================

print("\nStep 3: Loading GeoPackage and converting coordinates ...")

# Load only the columns we need
cols_needed = ["ig_date", "ig_month", "ig_year", "ig_utm_x", "ig_utm_y"]
gdf = gpd.read_file(GPKG_PATH, columns=cols_needed + ["geometry"])
gdf = gdf[cols_needed].copy()   # drop geometry — we only need the centroid coords

print(f"  Total events in GeoPackage : {len(gdf):,}")

# Convert ig_date to datetime, derive month_year
gdf["ig_date"]    = pd.to_datetime(gdf["ig_date"], errors="coerce")
gdf["month_year"] = gdf["ig_date"].dt.to_period("M").dt.to_timestamp()

# Keep only rows that fall within the new months we need
gdf = gdf[gdf["month_year"].isin(new_months)].copy()
print(f"  Events in target period    : {len(gdf):,}")

if len(gdf) == 0:
    print("  WARNING: No events found in the target period. Check EXTEND_FROM/EXTEND_TO.")

# Sinusoidal → WGS84
print("  Reprojecting Sinusoidal → WGS84 ...")
transformer = Transformer.from_proj(
    proj_from="+proj=sinu +R=6371007.181 +nadgrids=@null +wktext",
    proj_to="EPSG:4326",
    always_xy=True
)

lons_evt, lats_evt = transformer.transform(
    gdf["ig_utm_x"].values,
    gdf["ig_utm_y"].values
)
gdf["longitude"] = np.round(lons_evt, 4)
gdf["latitude"]  = np.round(lats_evt, 4)

# =============================================================================
# Step 4 – Snap events to the existing grid using KDTree
# =============================================================================

print("\nStep 4: Snapping events to grid with KDTree ...")

# Build grid point array — (lat, lon) pairs matching the existing grid
lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)  # both (6800, 9000)
grid_points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])  # (61_200_000, 2)

# Build KDTree — this is the memory-heavy step (~1.5 GB for 61 M points).
# If RAM is very tight, build on a subset and use searchsorted instead.
print("  Building KDTree (may take 1–2 min and ~1.5 GB RAM) ...")
kdtree = KDTree(grid_points)

print("  Querying nearest grid cell for each event ...")
query_pts = np.column_stack([gdf["latitude"].values, gdf["longitude"].values])
_, indices = kdtree.query(query_pts, k=1, workers=-1)   # workers=-1 = all CPUs

# Recover (lat_idx, lon_idx) from flat index
gdf["lat_idx"] = indices // n_lon
gdf["lon_idx"] = indices %  n_lon

# Free KDTree and grid_points — no longer needed
del kdtree, grid_points, lon_mesh, lat_mesh, query_pts
print("  KDTree freed from memory.")

# =============================================================================
# Step 5 – Create the output NetCDF (copy existing + append new months)
# =============================================================================

print("\nStep 5: Creating output NetCDF ...")
print(f"  Output : {OUTPUT_NC}")

with Dataset(EXISTING_NC, "r") as nc_in, \
     Dataset(OUTPUT_NC,   "w", format="NETCDF4") as nc_out:

    # ── Dimensions ────────────────────────────────────────────────────────────
    nc_out.createDimension("time", None)      # unlimited
    nc_out.createDimension("lat",  n_lat)
    nc_out.createDimension("lon",  n_lon)

    # ── Coordinate variables ──────────────────────────────────────────────────
    v_time = nc_out.createVariable(
        "time", "f8", ("time",), zlib=True, complevel=5, shuffle=True)
    v_time.units    = time_units
    v_time.calendar = time_cal

    v_lat = nc_out.createVariable(
        "lat", np.float32, ("lat",), zlib=True, complevel=5, shuffle=True)
    v_lon = nc_out.createVariable(
        "lon", np.float32, ("lon",), zlib=True, complevel=5, shuffle=True)

    v_lat[:] = lats_grid
    v_lon[:] = lons_grid

    # ── Ignition variable ─────────────────────────────────────────────────────
    v_ign = nc_out.createVariable(
        "ignition", np.int32, ("time", "lat", "lon"),
        zlib=True, complevel=5, shuffle=True,
        chunksizes=(1, n_lat, n_lon)          # one month per chunk
    )

    # ── Global attributes ─────────────────────────────────────────────────────
    nc_out.title       = "Combined Monthly Fire Ignition Data 2000 to 2026-04"
    nc_out.institution = "WSU"
    nc_out.source      = "GeoPackage conversion (FIREDpy)"
    nc_out.history     = (
        f"Original file extended to 2026-04 on {datetime.now().strftime('%Y-%m-%d')} "
        "using fired_au_2024_2026_05_2024_to_2026_events.gpkg"
    )
    nc_out.references  = "WSU Research"
    nc_out.comment     = "Spatial resolution of 0.005 degrees."
    nc_out.CRS         = "EPSG:4326 - WGS84"

    # ── Copy existing months slice-by-slice (RAM-safe) ─────────────────────────
    print(f"\n  Copying {n_existing} existing months ...")
    v_ign_in = nc_in.variables["ignition"]

    for t in range(n_existing):
        v_time[t]      = existing_times[t]
        v_ign[t, :, :] = v_ign_in[t, :, :]   # one 245 MB slice at a time
        if (t + 1) % 12 == 0 or t == n_existing - 1:
            print(f"    Copied {t + 1}/{n_existing} months", flush=True)

    # ── Append new months ─────────────────────────────────────────────────────
    print(f"\n  Appending {len(new_months)} new months ...")

    for m_idx, month_ts in enumerate(new_months):
        out_t = n_existing + m_idx

        # Convert timestamp to numeric days since epoch
        month_dt = month_ts.to_pydatetime()
        time_num  = date2num(month_dt, units=time_units, calendar=time_cal)
        v_time[out_t] = time_num

        # Build empty slice
        slice_2d = np.zeros((n_lat, n_lon), dtype=np.int32)

        # Fill ignition cells for this month
        mask = gdf["month_year"] == month_ts
        sub  = gdf[mask]
        if len(sub) > 0:
            slice_2d[sub["lat_idx"].values, sub["lon_idx"].values] = 1

        v_ign[out_t, :, :] = slice_2d

        print(f"    [{m_idx + 1}/{len(new_months)}] {month_ts.strftime('%Y-%m')} "
              f"— {mask.sum():>5} events → {int(slice_2d.sum()):>5} ignition cells",
              flush=True)

print("\n" + "=" * 65)
print("Pipeline complete.")
print(f"Output NetCDF : {OUTPUT_NC}")
total_months = n_existing + len(new_months)
print(f"Total months  : {total_months}  "
      f"({existing_dts.min().strftime('%Y-%m')} → {new_months[-1].strftime('%Y-%m')})")
print("=" * 65)
