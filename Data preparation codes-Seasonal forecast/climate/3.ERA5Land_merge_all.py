"""
3.ERA5Land_merge_all.py  — memory-efficient version
================================================================
Merges all interim files into 7 final netCDFs using netCDF4
direct write, one month at a time. Never loads more than one
month into RAM — safe to run on the login node.

Output format matches BARRA pipeline exactly:
  dims       : (band, lat, lon)
  band       : ['2000_01_01', ..., '2026_04_01']
  dtype      : float32
  compression: zlib level 5
  chunks     : (1, 6800, 9000)
================================================================
"""

import os
import glob
import gc
import numpy as np
import xarray as xr
import netCDF4 as nc4
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# USER SETTINGS
# ================================================================
INTERIM_DIR = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/24.ERA5_climate_data/interim'
OUT_DIR     = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/24.ERA5_climate_data'
LOG_PATH    = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/24.ERA5_climate_data/3_merge.log'
REF_GRID_PATH = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc'

VARS = {
    'tmean' : ('Monthly mean 2m temperature',            'degC'),
    'tmax'  : ('Monthly maximum 2m temperature',         'degC'),
    'tmin'  : ('Monthly minimum 2m temperature',         'degC'),
    'precip': ('Monthly total precipitation',            'mm'),
    'wind'  : ('Monthly mean 10m wind speed',            'm/s'),
    'vp9am' : ('Monthly mean vapour pressure 9am AEST',  'hPa'),
    'vp3pm' : ('Monthly mean vapour pressure 3pm AEST',  'hPa'),
}

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

log("=" * 70)
log("Script 3: Memory-efficient merge (one month at a time)")
log("=" * 70)

# Load reference grid
ref_ds  = xr.open_dataset(REF_GRID_PATH)
ref_lat = ref_ds['lat'].values.astype(np.float64)
ref_lon = ref_ds['lon'].values.astype(np.float64)
ref_ds.close()
n_lat = len(ref_lat)
n_lon = len(ref_lon)
log(f"Ref grid: {n_lat} lat x {n_lon} lon")

# ================================================================
# Build the master band list (chronological order)
# Use tmean as the reference variable to discover all months
# ================================================================
log("Building master band list from tmean interim files ...")

year_files = sorted(glob.glob(os.path.join(INTERIM_DIR, 'tmean', 'tmean_[0-9][0-9][0-9][0-9].nc')))
cds_files  = sorted(glob.glob(os.path.join(INTERIM_DIR, 'tmean', 'tmean_cds_*.nc')))

if not year_files:
    log("ERROR: No interim year files found. Did Script 1 complete?")
    raise SystemExit(1)

# Collect all band names from all files
all_band_info = []   # list of (band_name, source_file_path, band_index_in_file)

for yf in year_files:
    ds = xr.open_dataset(yf, engine='netcdf4')
    bands = ds['band'].values.tolist()
    ds.close()
    for bi, band in enumerate(bands):
        all_band_info.append((band, yf, bi))

for cf in cds_files:
    ds = xr.open_dataset(cf, engine='netcdf4')
    bands = ds['band'].values.tolist()
    ds.close()
    for bi, band in enumerate(bands):
        all_band_info.append((band, cf, bi))

# Sort chronologically by band name (YYYY_MM_DD format sorts correctly)
all_band_info.sort(key=lambda x: x[0])
all_bands = [x[0] for x in all_band_info]
n_bands   = len(all_bands)

log(f"Total months found: {n_bands}")
log(f"Date range: {all_bands[0]} → {all_bands[-1]}")

# ================================================================
# Process each variable
# ================================================================
for var_name, (long_name, units) in VARS.items():

    out_path = os.path.join(OUT_DIR, f'era5land_{var_name}_monthly_005_Aus.nc')

    log(f"\n{'─'*60}")
    log(f"Variable: {var_name}")
    log(f"Output  : {out_path}")
    log(f"{'─'*60}")

    if os.path.exists(out_path):
        log(f"  Already exists — skipping. Delete to rerun.")
        continue

    # Get the source file paths for this variable
    # (same structure as tmean but different var folder)
    var_year_files = sorted(glob.glob(
        os.path.join(INTERIM_DIR, var_name, f'{var_name}_[0-9][0-9][0-9][0-9].nc')))
    var_cds_files  = sorted(glob.glob(
        os.path.join(INTERIM_DIR, var_name, f'{var_name}_cds_*.nc')))

    if not var_year_files:
        log(f"  WARNING: No interim files for {var_name} — skipping.")
        continue

    # Build band→file mapping for this variable
    band_to_file = {}
    for yf in var_year_files:
        ds = xr.open_dataset(yf, engine='netcdf4')
        bands = ds['band'].values.tolist()
        ds.close()
        for bi, band in enumerate(bands):
            band_to_file[band] = (yf, bi)
    for cf in var_cds_files:
        ds = xr.open_dataset(cf, engine='netcdf4')
        bands = ds['band'].values.tolist()
        ds.close()
        for bi, band in enumerate(bands):
            band_to_file[band] = (cf, bi)

    # ── Create output netCDF4 file with pre-allocated dimensions ──
    log(f"  Creating output file ...")
    tmp_path = out_path + '.tmp'

    root = nc4.Dataset(tmp_path, 'w', format='NETCDF4')

    # Dimensions
    root.createDimension('band', n_bands)
    root.createDimension('lat',  n_lat)
    root.createDimension('lon',  n_lon)

    # Coordinate variables
    v_band = root.createVariable('band', str,  ('band',))
    v_lat  = root.createVariable('lat',  'f8', ('lat',))
    v_lon  = root.createVariable('lon',  'f8', ('lon',))

    v_lat.units     = 'degrees_north'
    v_lat.long_name = 'latitude'
    v_lon.units     = 'degrees_east'
    v_lon.long_name = 'longitude'

    v_lat[:]  = ref_lat
    v_lon[:]  = ref_lon
    v_band[:] = np.array(all_bands, dtype=object)

    # Main data variable — chunked and compressed
    v_data = root.createVariable(
        var_name, 'f4', ('band', 'lat', 'lon'),
        zlib=True, complevel=5,
        chunksizes=(1, n_lat, n_lon),
        fill_value=np.float32(np.nan)
    )
    v_data.long_name = long_name
    v_data.units     = units

    # Global attributes
    root.description = f'ERA5-Land {long_name}, resampled to 0.005deg reference grid'
    root.source      = 'ERA5-Land ECMWF/Copernicus. NCI zz93 + CDS API gap fill.'
    root.CRS         = 'EPSG:4326 - WGS84'
    root.resolution  = '0.005 degrees (~500m)'
    root.created     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    root.institution = 'WSU'

    # ── Write one month at a time ──────────────────────────
    log(f"  Writing {n_bands} months (one at a time) ...")

    prev_file = None
    prev_ds   = None

    for bi, band in enumerate(all_bands):
        if band not in band_to_file:
            log(f"  WARNING: {band} not found for {var_name} — writing NaN")
            v_data[bi, :, :] = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            continue

        src_file, src_idx = band_to_file[band]

        # Only reopen if different file from previous month
        if src_file != prev_file:
            if prev_ds is not None:
                prev_ds.close()
            prev_ds   = xr.open_dataset(src_file, engine='netcdf4')
            prev_file = src_file

        data = prev_ds[var_name].isel(band=src_idx).values.astype(np.float32)
        v_data[bi, :, :] = data

        if (bi + 1) % 12 == 0 or bi == n_bands - 1:
            log(f"  Written {bi+1}/{n_bands} months ({band})")

        del data
        gc.collect()

    if prev_ds is not None:
        prev_ds.close()

    root.close()

    # Rename tmp → final
    os.rename(tmp_path, out_path)
    fsize = os.path.getsize(out_path) / 1e9
    log(f"  Done: {fsize:.2f} GB — {out_path}")
    gc.collect()

# ================================================================
# Final summary
# ================================================================
log("\n" + "=" * 70)
log("MERGE COMPLETE — Final output files:")
total_gb = 0.0
for var_name in VARS:
    out_path = os.path.join(OUT_DIR, f'era5land_{var_name}_monthly_005_Aus.nc')
    if os.path.exists(out_path):
        fsize = os.path.getsize(out_path) / 1e9
        total_gb += fsize
        log(f"  {os.path.basename(out_path):<45}  {fsize:.2f} GB")
    else:
        log(f"  era5land_{var_name}_monthly_005_Aus.nc  MISSING")
log(f"\n  Total: {total_gb:.2f} GB")
log("=" * 70)
log("Your 7 ERA5-Land climate cubes are ready for the ML model.")
log("=" * 70)
