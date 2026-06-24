"""
Llightning_nc_005_Aus_RFS_pipeline.py
==============================================
Single end-to-end script: reads raw WWLLN lightning strike CSV events,
bins them to a 0.05° monthly grid for the OBSERVED period (2004-Aug →
2023-May), then infills the GAP periods (2000-Jan → 2004-Jul and
2023-Jun → 2026-May) using trend-adjusted climatological infilling.

No intermediate NetCDF is created.  The final output is written directly to:
  /scratch/ey42/aw1142/Seasonal_forecast_data/15.Lightning/
  3.Lightning_data_2000_2026_RFS/Llightning_nc_005_Aus_RFS_pipeline.nc

PIPELINE OVERVIEW
-----------------
PHASE 1 — Grid setup
  Load the reference lat/lon grid from locations_layer.nc.
  Build a KDTree over all grid-cell centroids for fast spatial snapping.

PHASE 2 — Bin observed strikes (2004-Aug → 2023-May)
  Read the raw WWLLN CSV in chunks (memory-safe).
  For each chunk, snap each strike to the nearest 0.05° cell and
  accumulate into monthly_lightning[band, lat, lon] (int32 accumulator).
  Only strikes inside the observed window are processed.

PHASE 3 — Compute detrended spatial pattern templates
  From the fully accumulated observed array, compute:
    a. Australia-wide total per month  → fit OLS linear trend
    b. Detrended spatial grid per month → averaged by calendar month
       to give 12 pattern templates (one per Jan–Dec)
  These templates capture WHERE lightning occurs in each season.
  Memory: the full observed array is int32 ≈ 54 GB — too large to keep
  in RAM alongside the pattern accumulators.  We therefore process it
  BAND BY BAND from the in-memory array (already built in Phase 2) rather
  than writing and re-reading a file.

PHASE 4 — Compute trend-adjusted infill totals
  For each gap month, compute:
    total_adj(t) = (intercept + slope * x(t)) + detrended_clim_total(M)
  clipped to ≥ 0, where x(t) is months since 2004-Aug-01.

PHASE 5 — Write output NetCDF band by band
  For each month in 2000-Jan → 2026-May:
    • Observed month  → write the accumulated int32 grid (cast to int16)
    • Gap month       → scale the spatial pattern template by the
                        ratio (total_adj / pattern_sum), clip, int16
  Band-by-band writing keeps peak RAM to ~1 grid (~490 MB) during output.

MEMORY MANAGEMENT
-----------------
  The dominant RAM consumer is the observed accumulator:
    226 bands × 6800 × 9000 × int32 = 49.6 GB
  The 12 pattern templates add:
    12 × 6800 × 9000 × float64 = 4.4 GB
  Total peak: ~54 GB.  Request ≥ 80 GB on Gadi to be safe.

  If your job only has 32 GB, set LOWMEM = True below.  In that mode
  the observed array is NOT kept in RAM; instead the script writes a
  temporary NetCDF after Phase 2 and reads bands back in Phase 5.
  This adds ~30 min of extra I/O but keeps peak RAM under 10 GB.

OUTPUTS
-------
  Llightning_nc_005_Aus_RFS_pipeline.nc
    Dimensions : band=317, lat=6800, lon=9000
    Variable   : lightning (int16, zlib compressed)
    Coordinates: band = ['2000_01_01', ..., '2026_05_01']
    Attributes : full provenance, trend parameters, infill method

RUNTIME ESTIMATE (Gadi normal queue, 8 CPUs)
--------------------------------------------
  Phase 2 (CSV binning)   :  60–90 min
  Phase 3 (pattern build) :   5–10 min  (all in RAM)
  Phase 5 (write output)  :  20–30 min
  Total                   : ~90–130 min
"""

import os
import sys
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
from scipy import stats
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")

t_start = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  — edit only this section
# ══════════════════════════════════════════════════════════════════════════════

# Input files
CSV_PATH  = ("/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/15.Lightning/"
             "1.All_WWLLN_Aus_csv/1_All_WWLLN_Aus.csv")
LOC_NC    = ("/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/"
             "0.Lat_Lon_layer/locations_layer.nc")

# Output
OUT_DIR   = ("/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/15.Lightning/"
             "3.Lightning_data_2000_2026_RFS")
NC_OUT    = os.path.join(OUT_DIR, "Llightning_nc_005_Aus_RFS_pipeline.nc")

# Observed data window (must match the WWLLN CSV coverage)
OBS_START = "2004-08-01"
OBS_END   = "2023-05-31"

# Full extended window
EXT_START = "2000-01"
EXT_END   = "2026-05"

# CSV reading
CSV_CHUNKSIZE = 1_000_000   # rows per chunk; lower if CSV read fails

# Low-memory mode: set True if available RAM < 60 GB
# Uses a temporary NetCDF instead of keeping observed array in RAM.
LOWMEM = False
TMP_NC = os.path.join(OUT_DIR, "_tmp_observed_bands.nc")   # deleted at end

# Month names for logging
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  GRID SETUP
# ══════════════════════════════════════════════════════════════════════════════
def elapsed():
    return f"{(time.time() - t_start)/60:.1f} min"

print("=" * 65)
print("  Lightning RFS Pipeline — start", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
print("=" * 65)

os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n[Phase 1] Loading reference grid from {LOC_NC} …")
loc_ds    = xr.open_dataset(LOC_NC)
lat_vals  = loc_ds["lat"].values        # shape (nlat,)
lon_vals  = loc_ds["lon"].values        # shape (nlon,)
loc_ds.close()

nlat, nlon = len(lat_vals), len(lon_vals)
print(f"  Grid : {nlat} lats × {nlon} lons  "
      f"(lat {lat_vals[0]:.3f}→{lat_vals[-1]:.3f}, "
      f"lon {lon_vals[0]:.3f}→{lon_vals[-1]:.3f})")

# Build flat (lat, lon) coordinate array for KDTree
lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)   # both (nlat, nlon)
flat_coords = np.column_stack([lat_grid.ravel(),
                               lon_grid.ravel()])       # (nlat*nlon, 2)
tree = cKDTree(flat_coords)
print(f"  KDTree built over {len(flat_coords):,} grid cells  [{elapsed()}]")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  BIN OBSERVED STRIKES → monthly_obs[band, lat, lon]
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[Phase 2] Binning WWLLN strikes → monthly grid  [{elapsed()}]")

obs_date_range = pd.date_range(start=OBS_START, end=OBS_END, freq="MS")
n_obs          = len(obs_date_range)

# Date string → band index lookup (fast)
obs_period_index = {p: i
                    for i, p in enumerate(
                        obs_date_range.to_period("M").astype(str)
                    )}

print(f"  Observed bands : {n_obs}  "
      f"({obs_date_range[0].date()} → {obs_date_range[-1].date()})")
print(f"  CSV chunk size : {CSV_CHUNKSIZE:,} rows")

# Allocate accumulator
# int32: max value per pixel ≈ few thousand; safe up to ~2.1 billion
print(f"  Allocating accumulator "
      f"({n_obs} × {nlat} × {nlon} × int32 = "
      f"{n_obs * nlat * nlon * 4 / 1e9:.1f} GB) …")
monthly_obs = np.zeros((n_obs, nlat, nlon), dtype=np.int32)

total_strikes = 0
chunk_count   = 0

for chunk in pd.read_csv(
        CSV_PATH,
        usecols=["Date", "Lat", "Lon"],
        parse_dates=["Date"],
        chunksize=CSV_CHUNKSIZE):

    # Filter to observed window
    chunk = chunk[(chunk["Date"] >= OBS_START) & (chunk["Date"] <= OBS_END)]
    if chunk.empty:
        continue

    # Calendar-month period string → band index
    periods = chunk["Date"].dt.to_period("M").astype(str)
    band_idx = periods.map(obs_period_index)

    # Drop any rows whose period isn't in our index (shouldn't happen)
    valid = band_idx.notna()
    chunk    = chunk[valid]
    band_idx = band_idx[valid].astype(int).values

    # Snap to nearest grid cell
    coords = chunk[["Lat", "Lon"]].to_numpy()
    _, flat_idx = tree.query(coords)          # flat index into (nlat*nlon)

    lat_idx = flat_idx // nlon
    lon_idx = flat_idx  % nlon

    # Accumulate — numpy.add.at is safe for repeated indices
    np.add.at(monthly_obs,
              (band_idx, lat_idx, lon_idx),
              1)

    total_strikes += len(chunk)
    chunk_count   += 1
    if chunk_count % 10 == 0:
        print(f"    chunk {chunk_count}  |  {total_strikes:,} strikes so far  [{elapsed()}]")

print(f"  Binning complete: {total_strikes:,} strikes in {chunk_count} chunks  [{elapsed()}]")
print(f"  observed max pixel value : {monthly_obs.max()}")
print(f"  observed total           : {monthly_obs.sum():,}")

# If LOWMEM: flush to temp file and free RAM now
if LOWMEM:
    print(f"\n  LOWMEM mode: writing temp file {TMP_NC} …")
    ds_tmp = xr.Dataset(
        {"lightning": (["band", "lat", "lon"], monthly_obs)},
        coords={
            "band": obs_date_range.strftime("%Y_%m_%d"),
            "lat" : lat_vals,
            "lon" : lon_vals,
        }
    )
    ds_tmp.to_netcdf(TMP_NC, format="NETCDF4",
                     encoding={"lightning": {"zlib": True, "complevel": 1}})
    ds_tmp.close()
    # Build a re-reader to use in Phase 5
    ds_tmp_r = xr.open_dataset(TMP_NC, engine="netcdf4")
    da_tmp_r = ds_tmp_r["lightning"]
    obs_date_to_bi = {d: i for i, d in enumerate(obs_date_range)}
    # Free the big array
    del monthly_obs
    import gc; gc.collect()
    print(f"  Temp file written; observed array freed from RAM  [{elapsed()}]")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DETRENDED SPATIAL PATTERN TEMPLATES  (12, one per calendar month)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[Phase 3] Building detrended spatial pattern templates  [{elapsed()}]")

# Australia-wide totals for each observed band
if LOWMEM:
    print("  LOWMEM: reading totals from temp file …")
    obs_totals = np.array([
        float(da_tmp_r.isel(band=i).values.sum())
        for i in range(n_obs)
    ])
else:
    obs_totals = monthly_obs.reshape(n_obs, -1).sum(axis=1).astype(np.float64)

x_obs = np.arange(n_obs, dtype=np.float64)
slope_px, intercept_px, *_ = stats.linregress(x_obs, obs_totals)
trend_pct = slope_px * 120 / obs_totals.mean() * 100
print(f"  OLS trend : slope={slope_px:.2f} flashes/month, "
      f"intercept={intercept_px:.2f}, {trend_pct:+.1f}%/decade")

# Accumulate detrended grids by calendar month
pat_sum   = {m: np.zeros((nlat, nlon), dtype=np.float64) for m in range(1, 13)}
pat_count = {m: 0 for m in range(1, 13)}

for i, d in enumerate(obs_date_range):
    m  = d.month
    xi = float(i)
    trend_i = intercept_px + slope_px * xi

    if LOWMEM:
        grid = da_tmp_r.isel(band=i).values.astype(np.float64)
    else:
        grid = monthly_obs[i].astype(np.float64)

    tot_i = obs_totals[i]
    if tot_i > 0:
        # Proportionally remove the trend contribution from each pixel
        detrended = grid - trend_i * (grid / tot_i)
    else:
        detrended = grid.copy()

    detrended = np.clip(detrended, 0.0, None)
    pat_sum[m]   += detrended
    pat_count[m] += 1

    if (i + 1) % 40 == 0 or (i + 1) == n_obs:
        print(f"    {i+1}/{n_obs} bands processed  [{elapsed()}]")

# Average
pat_mean = {}
for m in range(1, 13):
    if pat_count[m] > 0:
        pat_mean[m] = (pat_sum[m] / pat_count[m]).astype(np.float32)
    else:
        pat_mean[m] = np.zeros((nlat, nlon), dtype=np.float32)
    print(f"  {MONTH_NAMES[m-1]:>3}: {pat_count[m]} obs, "
          f"pattern sum = {pat_mean[m].sum():.0f}")

del pat_sum   # free ~4 GB
print(f"  Pattern templates ready  [{elapsed()}]")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  INFILL TOTALS FOR GAP MONTHS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[Phase 4] Computing trend-adjusted infill totals  [{elapsed()}]")

full_index  = pd.date_range(EXT_START, EXT_END, freq="MS")
n_full      = len(full_index)

# Detrended climatological totals (Australia-wide) per calendar month
obs_det     = obs_totals - (intercept_px + slope_px * x_obs)
clim_det    = {m: 0.0 for m in range(1, 13)}
clim_count  = {m: 0   for m in range(1, 13)}
for i, d in enumerate(obs_date_range):
    clim_det[d.month]   += obs_det[i]
    clim_count[d.month] += 1
for m in range(1, 13):
    if clim_count[m] > 0:
        clim_det[m] /= clim_count[m]

# Reference origin for x-axis (= first observed month)
obs_origin = obs_date_range[0]

def month_x(d):
    """Integer x-position of date d relative to OLS origin."""
    return (d.year - obs_origin.year) * 12 + (d.month - obs_origin.month)

# Build total and source flag arrays across the full index
full_totals = np.zeros(n_full, dtype=np.float64)
full_source = np.empty(n_full, dtype=object)

obs_set = set(obs_date_range)

for bi, d in enumerate(full_index):
    if d in obs_set:
        full_source[bi] = "observed"
        full_totals[bi] = obs_totals[obs_date_range.get_loc(d)]
    else:
        full_source[bi] = "infilled"
        xi              = month_x(d)
        trend_val       = intercept_px + slope_px * xi
        total_adj       = trend_val + clim_det[d.month]
        full_totals[bi] = max(0.0, total_adj)

n_obs_bands  = (full_source == "observed").sum()
n_fill_bands = (full_source == "infilled").sum()
print(f"  Full index : {n_full} months  "
      f"({n_obs_bands} observed, {n_fill_bands} infilled)")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  WRITE OUTPUT NetCDF  band by band
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[Phase 5] Writing output NetCDF  [{elapsed()}]")
print(f"  Output : {NC_OUT}")

if os.path.exists(NC_OUT):
    os.remove(NC_OUT)
    print("  Removed existing output file.")

full_band_strs = [d.strftime("%Y_%m_%d") for d in full_index]

# Index lookup: observed date → position in monthly_obs (or tmp file)
obs_date_to_idx = {d: i for i, d in enumerate(obs_date_range)}

with nc4.Dataset(NC_OUT, "w", format="NETCDF4") as dst:

    # ── Dimensions ────────────────────────────────────────────────────────────
    dst.createDimension("band", n_full)
    dst.createDimension("lat",  nlat)
    dst.createDimension("lon",  nlon)

    # ── Coordinate variables ──────────────────────────────────────────────────
    v_band      = dst.createVariable("band", str, ("band",))
    v_band[:]   = np.array(full_band_strs, dtype=object)
    v_band.long_name = "time (monthly, encoded as YYYY_MM_DD)"

    v_lat       = dst.createVariable("lat", "f4", ("lat",))
    v_lat[:]    = lat_vals.astype(np.float32)
    v_lat.units          = "degrees_north"
    v_lat.long_name      = "latitude"
    v_lat.standard_name  = "latitude"
    v_lat.axis           = "Y"

    v_lon       = dst.createVariable("lon", "f4", ("lon",))
    v_lon[:]    = lon_vals.astype(np.float32)
    v_lon.units          = "degrees_east"
    v_lon.long_name      = "longitude"
    v_lon.standard_name  = "longitude"
    v_lon.axis           = "X"

    # ── Lightning variable ────────────────────────────────────────────────────
    # Chunk (1, 256, 256): optimises time-slice reads (how models use this)
    v_light = dst.createVariable(
        "lightning", "i2",
        ("band", "lat", "lon"),
        chunksizes=(1, min(256, nlat), min(256, nlon)),
        zlib=True, complevel=4,
        fill_value=np.int16(-1)
    )
    v_light.long_name   = "Monthly lightning flash count"
    v_light.units       = "flash count per 0.05-degree pixel per month"
    v_light.valid_min   = np.int16(0)
    v_light.valid_max   = np.int16(32767)
    v_light.coordinates = "lat lon"
    v_light.cell_methods = "time: sum"

    # ── Global attributes ─────────────────────────────────────────────────────
    dst.title            = ("Australian Monthly Lightning Flash Count "
                            "2000-Jan to 2026-May (RFS Pipeline)")
    dst.institution      = "WSU Seasonal Forecast Group"
    dst.source           = ("WWLLN strike data binned to 0.05-degree grid; "
                            "gap months infilled via trend-adjusted climatology")
    dst.observed_period  = f"{obs_date_range[0].date()} to {obs_date_range[-1].date()}"
    dst.extended_period  = f"{full_index[0].date()} to {full_index[-1].date()}"
    dst.infill_method    = (
        "Trend-adjusted climatological infilling. "
        "OLS trend fitted to Australia-wide monthly totals over observed period. "
        "Spatial pattern = detrended calendar-month mean of observed grids. "
        "Gap month total = trend_extrapolation + detrended_climatological_mean. "
        "Infilled grid = spatial_pattern × (gap_total / pattern_sum)."
    )
    dst.trend_slope_flashes_per_month = float(slope_px)
    dst.trend_intercept_flashes       = float(intercept_px)
    dst.trend_pct_per_decade          = float(trend_pct)
    dst.n_observed_bands              = int(n_obs_bands)
    dst.n_infilled_bands              = int(n_fill_bands)
    dst.total_observed_strikes        = int(total_strikes)
    dst.Conventions                   = "CF-1.8"
    dst.history = (
        f"Created {datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')} UTC "
        f"by Llightning_nc_005_Aus_RFS_pipeline.py"
    )

    # ── Write bands ───────────────────────────────────────────────────────────
    print(f"  Writing {n_full} bands …")
    n_written_obs  = 0
    n_written_fill = 0

    for bi, d in enumerate(full_index):

        if full_source[bi] == "observed":
            # Copy directly from the in-memory accumulator (or temp file)
            idx_obs = obs_date_to_idx[d]
            if LOWMEM:
                grid_i = da_tmp_r.isel(band=idx_obs).values.astype(np.int16)
            else:
                grid_i = monthly_obs[idx_obs].astype(np.int16)
            # Clip to int16 range (observed values should be well within range)
            grid_out = np.clip(grid_i, 0, 32767).astype(np.int16)
            n_written_obs += 1

        else:
            # Infilled: scale spatial pattern to the target total
            m       = d.month
            pat     = pat_mean[m]            # float32 (nlat, nlon)
            pat_sum = float(pat.sum())
            tot     = full_totals[bi]

            if pat_sum > 0.0 and tot > 0.0:
                grid_f = pat * (tot / pat_sum)
            else:
                grid_f = np.zeros((nlat, nlon), dtype=np.float32)

            grid_out = np.clip(np.round(grid_f), 0, 32767).astype(np.int16)
            n_written_fill += 1

        v_light[bi, :, :] = grid_out

        if (bi + 1) % 20 == 0 or (bi + 1) == n_full:
            pct = (bi + 1) / n_full * 100
            print(f"  Band {bi+1:3d}/{n_full}  {d.strftime('%Y-%m')}  "
                  f"[obs={n_written_obs} fill={n_written_fill}]  "
                  f"{pct:.0f}%  [{elapsed()}]")

# ── Clean up temp file ─────────────────────────────────────────────────────
if LOWMEM and os.path.exists(TMP_NC):
    ds_tmp_r.close()
    os.remove(TMP_NC)
    print("  Temp file removed.")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[Verification]  [{elapsed()}]")
size_gb = os.path.getsize(NC_OUT) / 1e9
print(f"  File size : {size_gb:.2f} GB")

ds_v  = nc4.Dataset(NC_OUT, "r")
dims_v = {k: len(v) for k, v in ds_v.dimensions.items()}
print(f"  Dimensions: { {k:v for k,v in dims_v.items()} }")

band_v = np.array([str(b) for b in ds_v["band"][:]])
print(f"  Band[0]   : {band_v[0]}   "
      f"sum = {ds_v['lightning'][0,:,:].data.sum():,}")
print(f"  Band[-1]  : {band_v[-1]}  "
      f"sum = {ds_v['lightning'][-1,:,:].data.sum():,}")

# First observed band
first_obs_bi = int(full_index.get_loc(obs_date_range[0]))
print(f"  Band[{first_obs_bi}] : {band_v[first_obs_bi]}  "
      f"sum = {ds_v['lightning'][first_obs_bi,:,:].data.sum():,}  "
      f"← first observed")

# Check observed band matches accumulator
if not LOWMEM:
    expected = int(monthly_obs[0].sum())
    got      = int(ds_v["lightning"][first_obs_bi, :, :].data.sum())
    match    = "✓" if expected == got else f"✗ (expected {expected})"
    print(f"  Observed band[0] sum check : {match}")

ds_v.close()

print(f"\n{'='*65}")
print(f"  Pipeline complete  [{elapsed()}]")
print(f"  Output : {NC_OUT}")
print(f"  Bands  : {n_written_obs} observed + {n_written_fill} infilled = "
      f"{n_written_obs + n_written_fill} total")
print(f"{'='*65}")
