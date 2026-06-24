"""
1.ERA5Land_process_zz93.py
================================================================
Reads hourly ERA5-Land from /g/data/zz93 (NCI replica),
computes 7 monthly climate variables, regrids to the
reference 500m grid (locations_layer.nc), and saves
per-variable per-year intermediate netCDF files.

Variables:
  tmean  - Monthly mean 2m temperature         (°C)
  tmax   - Monthly maximum 2m temperature      (°C)
  tmin   - Monthly minimum 2m temperature      (°C)
  precip - Monthly total precipitation         (mm)
  wind   - Monthly mean 10m wind speed         (m/s)
  vp9am  - Vapour pressure ~9am AEST (23 UTC)  (hPa)
  vp3pm  - Vapour pressure ~3pm AEST (05 UTC)  (hPa)

Run via PBS job: 4.PBS_ERA5Land.sh
================================================================
"""

import os
import sys
import glob
import gc
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import calendar
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# USER SETTINGS
# ================================================================
YEAR_START    = 2000
YEAR_END      = 2025          # NCI zz93 confirmed up to Jan 2026;
                               # 2026 partial handled by Script 2

ZZ93_BASE     = '/g/data/zz93/era5-land/reanalysis'
REF_GRID_PATH = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc'
INTERIM_DIR   = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/24.ERA5_climate_data/interim'
LOG_PATH      = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/24.ERA5_climate_data/1_process_zz93.log'

# Australia bounding box — with 0.2° buffer for safe interpolation
LAT_MIN, LAT_MAX =  -44.5,  -9.5   # ref grid: -44.0 to -10.005
LON_MIN, LON_MAX = 109.5,  155.5   # ref grid: 110.0 to 154.995

VARS = ['tmean', 'tmax', 'tmin', 'precip', 'wind', 'vp9am', 'vp3pm']

# ================================================================
# LOGGING
# ================================================================
os.makedirs(INTERIM_DIR, exist_ok=True)
for v in VARS:
    os.makedirs(os.path.join(INTERIM_DIR, v), exist_ok=True)

def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

log("=" * 70)
log("Script 1: ERA5-Land zz93 processing started")
log(f"Period : {YEAR_START}–{YEAR_END}")
log("=" * 70)

# ================================================================
# STEP 1 — Load reference grid EXACTLY from locations_layer.nc
# ================================================================
log("Loading reference grid from locations_layer.nc ...")
ref_ds  = xr.open_dataset(REF_GRID_PATH)
ref_lat = ref_ds['lat'].values.astype(np.float64)   # S→N, -44.0 to -10.005
ref_lon = ref_ds['lon'].values.astype(np.float64)   # W→E, 110.0 to 154.995
ref_ds.close()

log(f"Ref grid: lat {ref_lat[0]:.4f}→{ref_lat[-1]:.4f} "
    f"({len(ref_lat)} pts), "
    f"lon {ref_lon[0]:.4f}→{ref_lon[-1]:.4f} "
    f"({len(ref_lon)} pts)")

# ================================================================
# HELPER: regrid a 2D ERA5 field onto the reference grid
# ================================================================
def regrid(data2d, src_lat, src_lon):
    """
    Bilinear interpolation of a 2D field (src_lat x src_lon)
    onto the reference 500m grid.

    src_lat MUST be ascending for RegularGridInterpolator.
    ERA5 latitude is N→S (descending), so we flip before calling.
    """
    # ERA5 latitude is descending — flip to ascending
    if src_lat[0] > src_lat[-1]:
        src_lat  = src_lat[::-1]
        data2d   = data2d[::-1, :]

    interp = RegularGridInterpolator(
        (src_lat, src_lon),
        data2d,
        method      = 'linear',
        bounds_error= False,
        fill_value  = np.nan
    )

    # Build target meshgrid — shape (6800, 9000)
    lon_grid, lat_grid = np.meshgrid(ref_lon, ref_lat)
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    result = interp(pts).reshape(len(ref_lat), len(ref_lon))
    return result.astype(np.float32)

# ================================================================
# HELPER: vapour pressure from dewpoint (Magnus formula)
# ================================================================
def dewpoint_to_vp(td_kelvin):
    """ Returns vapour pressure in hPa. """
    td_c = td_kelvin - 273.15
    return 6.1078 * np.exp(17.27 * td_c / (td_c + 237.3))

# ================================================================
# HELPER: find monthly files for a variable/year
# ================================================================
def get_files(var_folder, year):
    pattern = os.path.join(ZZ93_BASE, var_folder, str(year), '*.nc')
    return sorted(glob.glob(pattern))

# ================================================================
# HELPER: save one interim file  (1 year, 1 variable)
# ================================================================
def save_interim(data_3d, band_names, var_name, long_name, units, year):
    """
    data_3d   : np.ndarray shape (n_months, 6800, 9000) float32
    band_names: list of strings e.g. ['2000_01_01', '2000_02_01', ...]
    """
    out_path = os.path.join(INTERIM_DIR, var_name, f'{var_name}_{year}.nc')

    ds_out = xr.Dataset(
        {var_name: xr.DataArray(
            data_3d,
            dims   = ['band', 'lat', 'lon'],
            attrs  = {'long_name': long_name, 'units': units}
        )},
        coords = {
            'band': band_names,
            'lat' : ref_lat,
            'lon' : ref_lon,
        }
    )
    ds_out.attrs = {
        'source'    : 'ERA5-Land, ECMWF/Copernicus via NCI /g/data/zz93',
        'resolution': 'Resampled from 0.1° to 0.005° (500m) reference grid',
        'created'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    enc = {var_name: {
        'zlib'      : True,
        'complevel' : 5,
        'chunksizes': (1, len(ref_lat), len(ref_lon)),
        'dtype'     : 'float32'
    }}
    ds_out.to_netcdf(out_path, encoding=enc)
    log(f"    Saved interim: {out_path}")

# ================================================================
# STEP 2 — Main year loop
# ================================================================
for year in range(YEAR_START, YEAR_END + 1):

    log(f"\n{'─'*60}")
    log(f"Year: {year}")
    log(f"{'─'*60}")

    # Check if all variables already done for this year (resume support)
    already_done = all(
        os.path.exists(os.path.join(INTERIM_DIR, v, f'{v}_{year}.nc'))
        for v in VARS
    )
    if already_done:
        log(f"  Year {year} already complete — skipping.")
        continue

    # ── Get file lists ─────────────────────────────────────
    files_2t = get_files('2t',  year)
    files_tp = get_files('tp',  year)
    files_u  = get_files('u10', year)
    files_v  = get_files('v10', year)
    files_2d = get_files('2d',  year)

    if not files_2t:
        log(f"  WARNING: No 2t files for {year} — skipping year.")
        continue

    n_months = len(files_2t)
    log(f"  Found {n_months} monthly files for 2t")

    # Pre-allocate output arrays (n_months, 6800, 9000) float32
    arr_tmean  = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
    arr_tmax   = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
    arr_tmin   = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
    arr_precip = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
    arr_wind   = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
    arr_vp9am  = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
    arr_vp3pm  = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
    band_names = []

    # ── Month loop ─────────────────────────────────────────
    for mi, f2t in enumerate(files_2t):

        # Derive month from filename: 2t_era5-land_oper_sfc_20000101-20000131.nc
        basename   = os.path.basename(f2t)
        date_part  = basename.split('_sfc_')[1].split('-')[0]  # '20000101'
        yr_str     = date_part[:4]
        mo_str     = date_part[4:6]
        band_name  = f"{yr_str}_{mo_str}_01"
        band_names.append(band_name)
        month_int  = int(mo_str)
        n_days     = calendar.monthrange(year, month_int)[1]

        log(f"  Month {band_name} ({n_days} days)...")

        # ── Load and subset 2t ─────────────────────────────
        log(f"    Loading 2t ...")
        ds2t = xr.open_dataset(f2t, engine='netcdf4')

        # Subset to Australia + buffer BEFORE loading into RAM
        ds2t_aus = ds2t.sel(
            latitude = slice(LAT_MAX, LAT_MIN),   # descending N→S
            longitude= slice(LON_MIN, LON_MAX)
        )
        t2m_all  = ds2t_aus['t2m'].values          # (744, ~351, ~461) float64
        src_lat  = ds2t_aus['latitude'].values.astype(np.float64)
        src_lon  = ds2t_aus['longitude'].values.astype(np.float64)
        ds2t.close(); ds2t_aus.close()

        # tmean — mean of all hourly values, convert K→°C
        tmean_2d = t2m_all.mean(axis=0) - 273.15
        arr_tmean[mi] = regrid(tmean_2d, src_lat.copy(), src_lon)

        # tmax — max of all hourly values, convert K→°C
        tmax_2d  = t2m_all.max(axis=0) - 273.15
        arr_tmax[mi]  = regrid(tmax_2d,  src_lat.copy(), src_lon)

        # tmin — min of all hourly values, convert K→°C
        tmin_2d  = t2m_all.min(axis=0) - 273.15
        arr_tmin[mi]  = regrid(tmin_2d,  src_lat.copy(), src_lon)

        del t2m_all, tmean_2d, tmax_2d, tmin_2d
        gc.collect()

        # ── Load and subset tp ─────────────────────────────
        # Match file by month
        f_tp = [f for f in files_tp if f'_{yr_str}{mo_str}' in f]
        if f_tp:
            log(f"    Loading tp ...")
            ds_tp     = xr.open_dataset(f_tp[0], engine='netcdf4')
            ds_tp_aus = ds_tp.sel(
                latitude = slice(LAT_MAX, LAT_MIN),
                longitude= slice(LON_MIN, LON_MAX)
            )
            tp_all    = ds_tp_aus['tp'].values    # (744, lat, lon) metres
            ds_tp.close(); ds_tp_aus.close()

            # Monthly total precip in mm = sum of all hourly values × 1000
            # ERA5-Land tp is hourly accumulation in metres
            precip_2d = tp_all.sum(axis=0) * 1000.0
            # Clip negatives (occasional ERA5 artefact)
            precip_2d = np.clip(precip_2d, 0, None)
            arr_precip[mi] = regrid(precip_2d, src_lat.copy(), src_lon)

            del tp_all, precip_2d
            gc.collect()
        else:
            log(f"    WARNING: No tp file for {band_name}")

        # ── Load u10 and v10 for wind ──────────────────────
        f_u = [f for f in files_u if f'_{yr_str}{mo_str}' in f]
        f_v = [f for f in files_v if f'_{yr_str}{mo_str}' in f]
        if f_u and f_v:
            log(f"    Loading u10/v10 ...")
            ds_u     = xr.open_dataset(f_u[0], engine='netcdf4')
            ds_u_aus = ds_u.sel(
                latitude = slice(LAT_MAX, LAT_MIN),
                longitude= slice(LON_MIN, LON_MAX)
            )
            u_all    = ds_u_aus['u10'].values
            ds_u.close(); ds_u_aus.close()

            ds_v     = xr.open_dataset(f_v[0], engine='netcdf4')
            ds_v_aus = ds_v.sel(
                latitude = slice(LAT_MAX, LAT_MIN),
                longitude= slice(LON_MIN, LON_MAX)
            )
            v_all    = ds_v_aus['v10'].values
            ds_v.close(); ds_v_aus.close()

            # Wind speed scalar at each hour, then monthly mean
            wind_all  = np.sqrt(u_all**2 + v_all**2)   # (744, lat, lon)
            wind_2d   = wind_all.mean(axis=0)
            arr_wind[mi] = regrid(wind_2d, src_lat.copy(), src_lon)

            del u_all, v_all, wind_all, wind_2d
            gc.collect()
        else:
            log(f"    WARNING: No u10/v10 files for {band_name}")

        # ── Load 2d (dewpoint) for VP 9am and 3pm ─────────
        f_2d = [f for f in files_2d if f'_{yr_str}{mo_str}' in f]
        if f_2d:
            log(f"    Loading 2d (dewpoint) ...")
            ds_2d     = xr.open_dataset(f_2d[0], engine='netcdf4')
            ds_2d_aus = ds_2d.sel(
                latitude = slice(LAT_MAX, LAT_MIN),
                longitude= slice(LON_MIN, LON_MAX)
            )
            # Select only hours 23 UTC (9am AEST) and 05 UTC (3pm AEST)
            times      = ds_2d_aus['time']
            mask_9am   = times.dt.hour == 23    # 23:00 UTC = 09:00 AEST
            mask_3pm   = times.dt.hour == 5     # 05:00 UTC = 15:00 AEST

            d2m_9am    = ds_2d_aus['d2m'].isel(time=mask_9am).values  # (n_days, lat, lon)
            d2m_3pm    = ds_2d_aus['d2m'].isel(time=mask_3pm).values
            ds_2d.close(); ds_2d_aus.close()

            # Monthly mean VP at 9am AEST
            vp9am_2d   = dewpoint_to_vp(d2m_9am.mean(axis=0))
            arr_vp9am[mi] = regrid(vp9am_2d, src_lat.copy(), src_lon)

            # Monthly mean VP at 3pm AEST
            vp3pm_2d   = dewpoint_to_vp(d2m_3pm.mean(axis=0))
            arr_vp3pm[mi] = regrid(vp3pm_2d, src_lat.copy(), src_lon)

            del d2m_9am, d2m_3pm, vp9am_2d, vp3pm_2d
            gc.collect()
        else:
            log(f"    WARNING: No 2d files for {band_name}")

        log(f"    Month {band_name} done.")

    # ── Save all 7 interim files for this year ─────────────
    log(f"  Saving interim files for {year}...")
    save_interim(arr_tmean,  band_names, 'tmean',  'Monthly mean 2m temperature',     '°C',   year)
    save_interim(arr_tmax,   band_names, 'tmax',   'Monthly maximum 2m temperature',  '°C',   year)
    save_interim(arr_tmin,   band_names, 'tmin',   'Monthly minimum 2m temperature',  '°C',   year)
    save_interim(arr_precip, band_names, 'precip', 'Monthly total precipitation',     'mm',   year)
    save_interim(arr_wind,   band_names, 'wind',   'Monthly mean 10m wind speed',     'm/s',  year)
    save_interim(arr_vp9am,  band_names, 'vp9am',  'Monthly mean vapour pressure 9am AEST', 'hPa', year)
    save_interim(arr_vp3pm,  band_names, 'vp3pm',  'Monthly mean vapour pressure 3pm AEST', 'hPa', year)

    # Free year arrays
    del arr_tmean, arr_tmax, arr_tmin, arr_precip, arr_wind, arr_vp9am, arr_vp3pm
    gc.collect()
    log(f"  Year {year} complete.")

# ── 2026: process only January (confirmed available on zz93)
year = 2026
log(f"\n{'─'*60}")
log(f"Year: {year} (January only from zz93)")
log(f"{'─'*60}")

already_done_2026 = all(
    os.path.exists(os.path.join(INTERIM_DIR, v, f'{v}_{year}.nc'))
    for v in VARS
)
if not already_done_2026:
    files_2t = get_files('2t', year)
    # Only process January (first file)
    if files_2t:
        files_2t = files_2t[:1]   # Jan 2026 only; Feb+ handled by Script 2
        files_tp = get_files('tp',  year)[:1]
        files_u  = get_files('u10', year)[:1]
        files_v  = get_files('v10', year)[:1]
        files_2d = get_files('2d',  year)[:1]

        # Re-run the same month loop with just these files
        # (code is identical to above — factored via the same loop body)
        n_months   = 1
        arr_tmean  = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
        arr_tmax   = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
        arr_tmin   = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
        arr_precip = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
        arr_wind   = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
        arr_vp9am  = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
        arr_vp3pm  = np.full((n_months, len(ref_lat), len(ref_lon)), np.nan, dtype=np.float32)
        band_names = ['2026_01_01']

        mi = 0
        f2t = files_2t[0]
        yr_str, mo_str = '2026', '01'
        month_int = 1
        n_days = 31

        log(f"  Processing 2026_01_01 ...")

        ds2t     = xr.open_dataset(f2t, engine='netcdf4')
        ds2t_aus = ds2t.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
        t2m_all  = ds2t_aus['t2m'].values
        src_lat  = ds2t_aus['latitude'].values.astype(np.float64)
        src_lon  = ds2t_aus['longitude'].values.astype(np.float64)
        ds2t.close(); ds2t_aus.close()

        arr_tmean[mi]  = regrid(t2m_all.mean(axis=0) - 273.15, src_lat.copy(), src_lon)
        arr_tmax[mi]   = regrid(t2m_all.max(axis=0)  - 273.15, src_lat.copy(), src_lon)
        arr_tmin[mi]   = regrid(t2m_all.min(axis=0)  - 273.15, src_lat.copy(), src_lon)
        del t2m_all; gc.collect()

        if files_tp:
            ds_tp = xr.open_dataset(files_tp[0], engine='netcdf4')
            ds_tp_aus = ds_tp.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
            tp_all = ds_tp_aus['tp'].values
            ds_tp.close(); ds_tp_aus.close()
            arr_precip[mi] = regrid(np.clip(tp_all.sum(axis=0)*1000, 0, None), src_lat.copy(), src_lon)
            del tp_all; gc.collect()

        if files_u and files_v:
            ds_u = xr.open_dataset(files_u[0], engine='netcdf4')
            ds_u_aus = ds_u.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
            u_all = ds_u_aus['u10'].values; ds_u.close(); ds_u_aus.close()
            ds_v = xr.open_dataset(files_v[0], engine='netcdf4')
            ds_v_aus = ds_v.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
            v_all = ds_v_aus['v10'].values; ds_v.close(); ds_v_aus.close()
            arr_wind[mi] = regrid(np.sqrt(u_all**2 + v_all**2).mean(axis=0), src_lat.copy(), src_lon)
            del u_all, v_all; gc.collect()

        if files_2d:
            ds_2d = xr.open_dataset(files_2d[0], engine='netcdf4')
            ds_2d_aus = ds_2d.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
            times = ds_2d_aus['time']
            d2m_9am = ds_2d_aus['d2m'].isel(time=times.dt.hour == 23).values
            d2m_3pm = ds_2d_aus['d2m'].isel(time=times.dt.hour == 5).values
            ds_2d.close(); ds_2d_aus.close()
            arr_vp9am[mi] = regrid(dewpoint_to_vp(d2m_9am.mean(axis=0)), src_lat.copy(), src_lon)
            arr_vp3pm[mi] = regrid(dewpoint_to_vp(d2m_3pm.mean(axis=0)), src_lat.copy(), src_lon)
            del d2m_9am, d2m_3pm; gc.collect()

        save_interim(arr_tmean,  band_names, 'tmean',  'Monthly mean 2m temperature',     '°C',   year)
        save_interim(arr_tmax,   band_names, 'tmax',   'Monthly maximum 2m temperature',  '°C',   year)
        save_interim(arr_tmin,   band_names, 'tmin',   'Monthly minimum 2m temperature',  '°C',   year)
        save_interim(arr_precip, band_names, 'precip', 'Monthly total precipitation',     'mm',   year)
        save_interim(arr_wind,   band_names, 'wind',   'Monthly mean 10m wind speed',     'm/s',  year)
        save_interim(arr_vp9am,  band_names, 'vp9am',  'Monthly mean vapour pressure 9am AEST', 'hPa', year)
        save_interim(arr_vp3pm,  band_names, 'vp3pm',  'Monthly mean vapour pressure 3pm AEST', 'hPa', year)
        log(f"  2026 January complete.")
    else:
        log(f"  WARNING: No 2026 files found on zz93.")
else:
    log(f"  Year 2026 already complete — skipping.")

log("\n" + "=" * 70)
log("Script 1 complete. Run Script 2 next for CDS gap months.")
log("=" * 70)
