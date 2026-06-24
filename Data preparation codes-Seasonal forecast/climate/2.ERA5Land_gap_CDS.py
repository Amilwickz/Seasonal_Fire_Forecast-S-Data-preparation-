"""
2.ERA5Land_gap_CDS.py
================================================================
Downloads gap months (Feb-Apr 2026) from Copernicus CDS API,
processes them identically to Script 1, saves interim files.

NOTE: New CDS API delivers files as ZIP archives containing
      data_0.nc inside. This script handles that automatically.
================================================================
"""

import os
import sys
import gc
import zipfile
import shutil
import numpy as np
import xarray as xr
import cdsapi
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import calendar
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# USER SETTINGS
# ================================================================
GAP_MONTHS = [
    (2026, 2),
    (2026, 3),
    (2026, 4),
    (2026, 5),
]

REF_GRID_PATH = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc'
INTERIM_DIR   = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/24.ERA5_climate_data/interim'
DOWNLOAD_DIR  = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/24.ERA5_climate_data/cds_downloads'
LOG_PATH      = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/24.ERA5_climate_data/2_gap_CDS.log'

AUS_AREA = [-9.5, 109.5, -44.5, 155.5]   # [N, W, S, E]

VARS = ['tmean', 'tmax', 'tmin', 'precip', 'wind', 'vp9am', 'vp3pm']

# ================================================================
# SETUP
# ================================================================
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
for v in VARS:
    os.makedirs(os.path.join(INTERIM_DIR, v), exist_ok=True)

def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

log("=" * 70)
log("Script 2: CDS gap months processing started")
log(f"Gap months: {GAP_MONTHS}")
log("=" * 70)

# ================================================================
# Load reference grid
# ================================================================
log("Loading reference grid ...")
ref_ds  = xr.open_dataset(REF_GRID_PATH)
ref_lat = ref_ds['lat'].values.astype(np.float64)
ref_lon = ref_ds['lon'].values.astype(np.float64)
ref_ds.close()
log(f"Ref grid: {len(ref_lat)} lat x {len(ref_lon)} lon")

# ================================================================
# HELPERS
# ================================================================

def unzip_cds(filepath):
    """New CDS API wraps netCDF inside a ZIP. Detect and extract."""
    if not zipfile.is_zipfile(filepath):
        return filepath
    log(f"    Detected ZIP archive - extracting ...")
    with zipfile.ZipFile(filepath, 'r') as z:
        names = z.namelist()
        log(f"    ZIP contents: {names}")
        nc_names = [n for n in names if n.endswith('.nc')]
        if not nc_names:
            raise RuntimeError(f"No .nc file inside ZIP: {filepath}")
        tmp_path = filepath + '.tmp.nc'
        with z.open(nc_names[0]) as src, open(tmp_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    os.remove(filepath)
    os.rename(tmp_path, filepath)
    log(f"    Extracted successfully.")
    return filepath


def download_var(c, variable_list, year, month, days_list,
                 hours_list, out_path, label):
    """Download from CDS if not present, then unzip."""
    if os.path.exists(out_path):
        log(f"  {label} already downloaded.")
        return
    log(f"  Downloading {label} ...")
    c.retrieve('reanalysis-era5-land', {
        'variable'   : variable_list,
        'year'       : [str(year)],
        'month'      : [f'{month:02d}'],
        'day'        : days_list,
        'time'       : hours_list,
        'data_format': 'netcdf',
        'area'       : AUS_AREA,
    }, out_path)
    unzip_cds(out_path)
    log(f"  {label} ready.")


def regrid(data2d, src_lat, src_lon):
    """Bilinear interpolation onto the 500m reference grid."""
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data2d  = data2d[::-1, :]
    interp = RegularGridInterpolator(
        (src_lat, src_lon), data2d,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    lon_grid, lat_grid = np.meshgrid(ref_lon, ref_lat)
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return interp(pts).reshape(len(ref_lat), len(ref_lon)).astype(np.float32)


def dewpoint_to_vp(td_kelvin):
    """Dewpoint (K) to vapour pressure (hPa)."""
    td_c = td_kelvin - 273.15
    return 6.1078 * np.exp(17.27 * td_c / (td_c + 237.3))


def get_coord(ds, candidates):
    """Return first matching coordinate name."""
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"None of {candidates} in {list(ds.coords)}")


def save_interim(data2d, band_name, var_name, long_name, units):
    """Save single month as compressed interim netCDF."""
    out_path = os.path.join(INTERIM_DIR, var_name,
                            f'{var_name}_cds_{band_name}.nc')
    ds_out = xr.Dataset(
        {var_name: xr.DataArray(
            data2d[np.newaxis, :, :],
            dims=['band', 'lat', 'lon'],
            attrs={'long_name': long_name, 'units': units}
        )},
        coords={'band': [band_name], 'lat': ref_lat, 'lon': ref_lon}
    )
    ds_out.attrs = {
        'source' : 'ERA5-Land, ECMWF/Copernicus CDS API',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    enc = {var_name: {
        'zlib': True, 'complevel': 5,
        'chunksizes': (1, len(ref_lat), len(ref_lon)),
        'dtype': 'float32'
    }}
    ds_out.to_netcdf(out_path, encoding=enc)
    log(f"    Saved: {os.path.basename(out_path)}")

# ================================================================
# Connect to CDS
# ================================================================
log("Connecting to CDS API ...")
try:
    c = cdsapi.Client()
    log("CDS connected.")
except Exception as e:
    log(f"ERROR: {e}")
    sys.exit(1)

# ================================================================
# Main gap loop
# ================================================================
for (year, month) in GAP_MONTHS:

    band_name = f"{year}_{month:02d}_01"
    log(f"\n{'─'*60}")
    log(f"Processing gap month: {band_name}")
    log(f"{'─'*60}")

    already_done = all(
        os.path.exists(os.path.join(INTERIM_DIR, v,
                                    f'{v}_cds_{band_name}.nc'))
        for v in VARS
    )
    if already_done:
        log(f"  Already complete - skipping.")
        continue

    n_days    = calendar.monthrange(year, month)[1]
    days_list = [f"{d:02d}" for d in range(1, n_days + 1)]
    all_hours = [f"{h:02d}:00" for h in range(24)]

    f_2t   = os.path.join(DOWNLOAD_DIR, f'2t_{year}{month:02d}.nc')
    f_tp   = os.path.join(DOWNLOAD_DIR, f'tp_{year}{month:02d}.nc')
    f_wind = os.path.join(DOWNLOAD_DIR, f'wind_{year}{month:02d}.nc')
    f_2d   = os.path.join(DOWNLOAD_DIR, f'2d_{year}{month:02d}.nc')

    # Downloads
    download_var(c, ['2m_temperature'],
                 year, month, days_list, all_hours, f_2t, '2t')

    download_var(c, ['total_precipitation'],
                 year, month, days_list, all_hours, f_tp, 'tp')

    download_var(c, ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                 year, month, days_list, all_hours, f_wind, 'u10/v10')

    download_var(c, ['2m_dewpoint_temperature'],
                 year, month, days_list, ['23:00', '05:00'], f_2d, '2d')

    # Process 2t
    log(f"  Processing 2t ...")
    ds2t    = xr.open_dataset(f_2t, engine='netcdf4')
    t_var   = 't2m' if 't2m' in ds2t.data_vars else list(ds2t.data_vars)[0]
    lat_nm  = get_coord(ds2t, ['latitude', 'lat'])
    lon_nm  = get_coord(ds2t, ['longitude', 'lon'])
    t2m     = ds2t[t_var].values
    src_lat = ds2t[lat_nm].values.astype(np.float64)
    src_lon = ds2t[lon_nm].values.astype(np.float64)
    ds2t.close()

    save_interim(regrid(t2m.mean(axis=0) - 273.15, src_lat.copy(), src_lon),
                 band_name, 'tmean', 'Monthly mean 2m temperature', 'degC')
    save_interim(regrid(t2m.max(axis=0)  - 273.15, src_lat.copy(), src_lon),
                 band_name, 'tmax',  'Monthly maximum 2m temperature', 'degC')
    save_interim(regrid(t2m.min(axis=0)  - 273.15, src_lat.copy(), src_lon),
                 band_name, 'tmin',  'Monthly minimum 2m temperature', 'degC')
    del t2m; gc.collect()

    # Process tp
    log(f"  Processing tp ...")
    ds_tp  = xr.open_dataset(f_tp, engine='netcdf4')
    tp_var = 'tp' if 'tp' in ds_tp.data_vars else list(ds_tp.data_vars)[0]
    tp     = ds_tp[tp_var].values
    ds_tp.close()
    save_interim(regrid(np.clip(tp.sum(axis=0) * 1000.0, 0, None),
                        src_lat.copy(), src_lon),
                 band_name, 'precip', 'Monthly total precipitation', 'mm')
    del tp; gc.collect()

    # Process wind
    log(f"  Processing wind ...")
    ds_w  = xr.open_dataset(f_wind, engine='netcdf4')
    u_var = 'u10' if 'u10' in ds_w.data_vars else [v for v in ds_w.data_vars if 'u' in v.lower()][0]
    v_var = 'v10' if 'v10' in ds_w.data_vars else [v for v in ds_w.data_vars if 'v' in v.lower()][0]
    u     = ds_w[u_var].values
    v     = ds_w[v_var].values
    ds_w.close()
    save_interim(regrid(np.sqrt(u**2 + v**2).mean(axis=0), src_lat.copy(), src_lon),
                 band_name, 'wind', 'Monthly mean 10m wind speed', 'm/s')
    del u, v; gc.collect()

    # Process dewpoint -> vapour pressure
    log(f"  Processing dewpoint -> vapour pressure ...")
    ds_2d   = xr.open_dataset(f_2d, engine='netcdf4')
    d_var   = 'd2m' if 'd2m' in ds_2d.data_vars else list(ds_2d.data_vars)[0]
    t_coord = get_coord(ds_2d, ['time', 'valid_time'])
    hours   = ds_2d[t_coord].dt.hour.values
    d2m     = ds_2d[d_var].values
    ds_2d.close()

    save_interim(regrid(dewpoint_to_vp(d2m[hours == 23].mean(axis=0)),
                        src_lat.copy(), src_lon),
                 band_name, 'vp9am', 'Monthly mean vapour pressure 9am AEST', 'hPa')
    save_interim(regrid(dewpoint_to_vp(d2m[hours == 5].mean(axis=0)),
                        src_lat.copy(), src_lon),
                 band_name, 'vp3pm', 'Monthly mean vapour pressure 3pm AEST', 'hPa')
    del d2m; gc.collect()

    # Remove raw downloads
    for fpath in [f_2t, f_tp, f_wind, f_2d]:
        if os.path.exists(fpath):
            os.remove(fpath)
            log(f"  Removed: {os.path.basename(fpath)}")

    log(f"  Gap month {band_name} complete.")

log("\n" + "=" * 70)
log("Script 2 complete. Run Script 3 to merge all interim files.")
log("=" * 70)
