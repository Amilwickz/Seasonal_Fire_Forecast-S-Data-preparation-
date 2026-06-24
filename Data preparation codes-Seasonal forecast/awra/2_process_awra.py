"""
2_process_awra.py  (v3 — reads local files, no internet needed)
===============================================================
Reads downloaded AWRA-L v7 NetCDF files from scratch storage,
aggregates daily → monthly means, resamples to 500m locations_layer.nc
grid (nearest-neighbour), and writes one NetCDF cube per variable.

Peak memory: ~2–4 GB. Runs on normal PBS compute nodes.

Usage (called by PBS array job):
    python 2_process_awra.py <variable>
    e.g. python 2_process_awra.py s0

Variables: s0, sd, ss, qtot, etot, e0, dd
"""

import sys
import os
import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.spatial import cKDTree
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR = (
    "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/"
    "13.AWRA-L/0.raw_nc"
)

LOCATIONS_FILE = (
    "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/"
    "0.Lat_Lon_layer/locations_layer.nc"
)

OUTPUT_DIR = (
    "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/"
    "13.AWRA-L/2.monthly_cubes"
)

VALID_VARIABLES = ["s0", "sd", "ss", "qtot", "etot", "e0", "dd"]

YEAR_START = 2000
YEAR_END   = 2026
MONTH_END  = 5      # Stop at April 2026 (last complete month)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_band_list() -> list:
    """
    All complete months 2000-01 to 2026-04 as 'YYYY_MM_01' strings.
    Total = 26 full years (2000-2025) x 12 + 4 months (2026) = 316 months.
    """
    bands = []
    for year in range(YEAR_START, YEAR_END + 1):
        m_end = MONTH_END if year == YEAR_END else 12
        for month in range(1, m_end + 1):
            bands.append(f"{year}_{month:02d}_01")
    return bands


def build_src_kdtree(src_lat, src_lon):
    """
    Build kd-tree from the AWRA source grid.
    Returns (tree, ascending_lat_array).
    """
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
    lat_2d, lon_2d = np.meshgrid(src_lat, src_lon, indexing='ij')
    coords = np.column_stack((lat_2d.ravel(), lon_2d.ravel()))
    return cKDTree(coords), src_lat


def resample_slice(data_2d, src_lat_asc, src_tree, target_coords, target_shape):
    """
    Nearest-neighbour resample one 2D field onto the target grid.
    data_2d      : (n_src_lat, n_src_lon)
    Returns      : (n_tgt_lat, n_tgt_lon) float32
    """
    # Flip source to ascending latitude order if needed
    if src_lat_asc[0] != src_lat_asc[0]:   # always ensure ascending
        pass
    data_2d = np.where(np.abs(data_2d) > 1e10, np.nan, data_2d)

    _, idx = src_tree.query(target_coords, k=1)
    resampled = data_2d.ravel()[idx].reshape(target_shape).astype(np.float32)
    return resampled


def init_netcdf(output_path, var_name, lat_vals, lon_vals, all_bands):
    """
    Create output NetCDF with unlimited band dimension.
    Returns open netCDF4.Dataset (caller must close).
    """
    ds = nc.Dataset(output_path, 'w', format='NETCDF4')

    ds.createDimension('band', None)           # unlimited
    ds.createDimension('lat',  len(lat_vals))
    ds.createDimension('lon',  len(lon_vals))

    v_lat  = ds.createVariable('lat',  'f4', ('lat',))
    v_lon  = ds.createVariable('lon',  'f4', ('lon',))
    v_band = ds.createVariable('band', str,  ('band',))

    v_lat[:]  = lat_vals.astype(np.float32)
    v_lon[:]  = lon_vals.astype(np.float32)
    for i, b in enumerate(all_bands):
        v_band[i] = b

    v_lat.units     = 'degrees_north'
    v_lat.long_name = 'latitude'
    v_lon.units     = 'degrees_east'
    v_lon.long_name = 'longitude'
    v_band.long_name = 'Month identifier (YYYY_MM_01)'

    chunk_lat = min(256, len(lat_vals))
    chunk_lon = min(256, len(lon_vals))

    v_data = ds.createVariable(
        var_name.upper(), 'f4', ('band', 'lat', 'lon'),
        fill_value=-9999.0,
        zlib=True, complevel=4,
        chunksizes=(1, chunk_lat, chunk_lon)
    )
    v_data.units        = 'mm'
    v_data.long_name    = f'{var_name.upper()} monthly mean'
    v_data.missing_value = np.float32(-9999.0)

    ds.description       = (
        f'AWRA-L v7 {var_name.upper()} monthly mean resampled to '
        f'500m grid (nearest-neighbour). Coverage: 2000-01 to 2026-04.'
    )
    ds.source            = (
        'NCI THREDDS: thredds.nci.org.au/thredds/fileServer/iu04/'
        'australian-water-outlook/historical/v1/AWRALv7'
    )
    ds.variable          = var_name.upper()
    ds.units             = 'mm'
    ds.native_resolution = '0.05 degrees (~5km)'
    ds.resampled_res     = '~500m (locations_layer.nc grid)'
    ds.resampling_method = 'nearest-neighbour'
    ds.CRS               = 'EPSG:4326 - WGS84'
    ds.time_coverage     = '2000-01-01 to 2026-04-30'
    ds.band_format       = 'YYYY_MM_01 (first day of each month)'
    ds.created_by        = '2_process_awra.py v3'

    return ds


# ── Main processing function ──────────────────────────────────────────────────

def process_variable(var_name):

    print(f"\n{'='*60}")
    print(f"  Processing variable : {var_name.upper()}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"AWRA_{var_name.upper()}_monthly_500m.nc")

    if os.path.exists(output_path):
        print(f"  Output already exists — delete to reprocess:\n  {output_path}")
        return

    # ── Load target grid ──────────────────────────────────────────────────────
    print(f"  Loading locations layer ...")
    loc_ds  = xr.open_dataset(LOCATIONS_FILE)
    lat_tgt = loc_ds.lat.values
    lon_tgt = loc_ds.lon.values
    loc_ds.close()

    lat_tgt_2d, lon_tgt_2d = np.meshgrid(lat_tgt, lon_tgt, indexing='ij')
    target_shape  = lat_tgt_2d.shape
    target_coords = np.column_stack((lat_tgt_2d.ravel(), lon_tgt_2d.ravel()))

    print(f"  Target grid : {target_shape[0]} lat × {target_shape[1]} lon "
          f"= {target_shape[0]*target_shape[1]:,} cells")

    all_bands = build_band_list()
    print(f"  Total months: {len(all_bands)}  ({all_bands[0]} → {all_bands[-1]})\n")

    # ── Initialise output NetCDF ──────────────────────────────────────────────
    print(f"  Creating output file: {output_path}")
    out_nc   = init_netcdf(output_path, var_name, lat_tgt, lon_tgt, all_bands)
    v_data   = out_nc.variables[var_name.upper()]
    band_idx = 0
    src_tree    = None
    src_lat_asc = None

    # ── Loop over years ───────────────────────────────────────────────────────
    for year in range(YEAR_START, YEAR_END + 1):

        nc_path = Path(RAW_DIR) / var_name / f"{var_name}_{year}.nc"
        m_end   = MONTH_END if year == YEAR_END else 12

        if not nc_path.exists():
            print(f"  [{year}] WARNING: file not found: {nc_path}")
            print(f"  [{year}] Filling {m_end} months with NaN.")
            nan_slice = np.full(target_shape, np.nan, dtype=np.float32)
            for _ in range(m_end):
                fill = np.where(np.isnan(nan_slice), -9999.0, nan_slice)
                v_data[band_idx, :, :] = fill
                band_idx += 1
            continue

        print(f"  [{year}] Reading {nc_path.name} ...")

        try:
            ds_year = xr.open_dataset(str(nc_path))
        except Exception as e:
            print(f"  [{year}] ERROR opening file: {e} — filling with NaN.")
            band_idx += m_end
            continue

        # Build source kd-tree once from the first successfully opened file
        if src_tree is None:
            src_lat = ds_year['latitude'].values
            src_lon = ds_year['longitude'].values
            print(f"  Building source kd-tree "
                  f"({len(src_lat)} lat × {len(src_lon)} lon) ...")
            src_tree, src_lat_asc = build_src_kdtree(src_lat, src_lon)
            print(f"  kd-tree ready.\n")

        # Subset time to complete months only
        da_year = ds_year[var_name].sel(
            time=ds_year['time'].dt.month.isin(range(1, m_end + 1))
        )

        # Daily → monthly mean (all computation is local, fast)
        print(f"  [{year}] Aggregating to monthly means ...")
        da_monthly = da_year.resample(time='ME').mean()
        monthly_np = da_monthly.values   # (n_months, 681, 841)
        ds_year.close()

        print(f"  [{year}] Resampling {monthly_np.shape[0]} months → writing ...")

        for m_i in range(monthly_np.shape[0]):
            # Flip source to ascending lat before resampling
            data_2d = monthly_np[m_i]
            if src_lat_asc[0] > src_lat_asc[-1]:
                data_2d = data_2d[::-1, :]

            resampled = resample_slice(
                data_2d, src_lat_asc, src_tree,
                target_coords, target_shape
            )

            # Flip target lat axis if descending
            if lat_tgt[0] > lat_tgt[-1]:
                resampled = resampled[::-1, :]

            # Replace NaN with fill value
            resampled = np.where(np.isnan(resampled), -9999.0, resampled)

            v_data[band_idx, :, :] = resampled
            band_idx += 1

        out_nc.sync()
        print(f"  [{year}] Done. {band_idx}/{len(all_bands)} months written.\n")

    out_nc.close()
    print(f"\n  ✓ Complete: {output_path}")
    print(f"    Total bands written: {band_idx}/{len(all_bands)}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 2_process_awra.py <variable>")
        print(f"  Valid: {', '.join(VALID_VARIABLES)}")
        sys.exit(1)

    var_arg = sys.argv[1].lower().strip()
    if var_arg not in VALID_VARIABLES:
        print(f"ERROR: '{var_arg}' not valid. Choose: {', '.join(VALID_VARIABLES)}")
        sys.exit(1)

    process_variable(var_arg)
    print("All done.\n")
