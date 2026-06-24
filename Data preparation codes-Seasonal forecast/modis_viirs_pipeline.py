#!/usr/bin/env python3
"""
MODIS + VIIRS Merged Burned Area Pipeline  (memory-efficient version)
======================================================================
Produces MODIS_VIIRS_BA_005_Aus.nc — a single continuous data cube:
  - MODIS  : 2000-11  to  2012-01   (read from existing MODIS_BA_nc_005_Aus.nc)
  - VIIRS  : 2012-02  to  END_DATE  (downloaded from NASA FIRMS, processed fresh)

Memory strategy:
  - MODIS bands are saved one at a time as small monthly NetCDF files
    (same format as VIIRS monthly files) — never loaded all at once
  - VIIRS months are also processed one at a time
  - Final cube is built by concatenating the small monthly files

This keeps peak memory well under 4GB regardless of how many months exist.

To configure: edit only the CONFIG section below.
"""

import os
import sys
import time
import logging
import requests
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import date, timedelta
from io import StringIO
from scipy.spatial import cKDTree

# =============================================================================
# CONFIG — only edit this section
# =============================================================================

MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "f7bdb20777e25d33a1a6760f6ac4ec73")

BASE_DIR       = Path("/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/10.MODIS_VIIRS_BA")
REFERENCE_GRID = Path("/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc")
MODIS_CUBE     = BASE_DIR / "0.Raw_MODIS" / "MODIS_BA_nc_005_Aus.nc"

# MODIS contributes bands up to and including this month
MODIS_END = "2012-01"

# VIIRS period starts the month after MODIS ends
VIIRS_START_DATE = date(2012, 2, 1)

# VIIRS end date (last complete month you want)
# Test (covers ~4 VIIRS months):  END_DATE = date(2012, 5, 31)
# Full build:                      END_DATE = date(2026, 4, 30)
# Future update: change this to last day of the new month and resubmit
END_DATE = date(2026, 5, 31)

# Output
OUTPUT_CUBE = BASE_DIR / "MODIS_VIIRS_BA_005_Aus.nc"

# =============================================================================
# DO NOT EDIT BELOW
# =============================================================================

RAW_DIR      = BASE_DIR / "0.Raw_VIIRS"
MONTHLY_DIR  = BASE_DIR / "1.Monthly_nc"   # stores BOTH modis and viirs monthly files

BBOX             = "110.0,-44.0,155.0,-10.0"
UTC_OFFSET_HOURS = 10
REQUEST_DELAY    = 1.5
MAX_RETRIES      = 3
COMPRESSION      = {"zlib": True, "complevel": 5, "dtype": "int8"}
FIRMS_URL        = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

SENSORS = [
    ("SNPP",   "SNPP",   date(2012, 1, 20)),
    ("NOAA20", "NOAA20", date(2018, 4,  1)),
    ("NOAA21", "NOAA21", date(2024, 1, 17)),
]
SENSOR_SOURCES = {
    "SNPP":   {"SP": "VIIRS_SNPP_SP",   "NRT": "VIIRS_SNPP_NRT"},
    "NOAA20": {"SP": "VIIRS_NOAA20_SP", "NRT": "VIIRS_NOAA20_NRT"},
    "NOAA21": {"SP": None,              "NRT": "VIIRS_NOAA21_NRT"},
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_existing_monthly_files():
    """Return set of labels (e.g. '2012-01') already saved in 1.Monthly_nc/"""
    existing = set()
    if MONTHLY_DIR.exists():
        for f in MONTHLY_DIR.glob("BA_????-??.nc"):
            label = f.stem.replace("BA_", "")
            existing.add(label)
    return existing


def save_monthly_nc(layer, lats, lons, label, source, logger):
    """
    Save one binary monthly layer as a small compressed NetCDF.
    Filename: BA_YYYY-MM.nc  (works for both MODIS and VIIRS months)
    """
    out_path = MONTHLY_DIR / f"BA_{label}.nc"
    da = xr.DataArray(
        layer.astype(np.int8),
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
        name="burned_area",
        attrs={
            "long_name": "Binary burned area",
            "month":     label,
            "source":    source,
            "units":     "1=burned 0=unburned",
        }
    )
    ds = xr.Dataset({"burned_area": da})
    ds.attrs = {
        "title":       f"Monthly Binary Burned Area (500m) — {label}",
        "institution": "WSU",
        "source":      source,
        "CRS":         "EPSG:4326 - WGS84",
    }
    ds.to_netcdf(out_path, encoding={"burned_area": COMPRESSION})
    logger.info(f"  Saved -> {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    ds.close()


# ---------------------------------------------------------------------------
# PHASE 1 — MODIS: save each band as a tiny monthly NC (one at a time)
# ---------------------------------------------------------------------------

def phase1_extract_modis(logger):
    """
    Read MODIS cube one band at a time and save each as BA_YYYY-MM.nc.
    Skips bands already saved. Keeps peak memory to one 6800x9000 layer (~58MB).
    Returns (lats, lons) from the MODIS grid.
    """
    logger.info("=" * 65)
    logger.info("PHASE 1 — Extracting MODIS bands (one at a time)")
    logger.info(f"  Source : {MODIS_CUBE}")
    logger.info(f"  Saving bands up to : {MODIS_END}")
    logger.info("=" * 65)

    if not MODIS_CUBE.exists():
        logger.error(f"MODIS cube not found: {MODIS_CUBE}")
        sys.exit(1)

    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    existing = get_existing_monthly_files()

    # Open dataset in lazy mode (does NOT load all data into memory)
    ds        = xr.open_dataset(MODIS_CUBE)
    lats      = ds["burned_area"].lat.values.copy()
    lons      = ds["burned_area"].lon.values.copy()
    all_bands = ds["burned_area"].band.values.tolist()

    keep_bands = [b for b in all_bands if b <= MODIS_END]
    logger.info(f"  MODIS cube covers : {all_bands[0]} -> {all_bands[-1]}  ({len(all_bands)} bands)")
    logger.info(f"  Bands to extract  : {keep_bands[0]} -> {keep_bands[-1]}  ({len(keep_bands)} bands)")

    saved   = 0
    skipped = 0

    for label in keep_bands:
        if label in existing:
            logger.info(f"  [{label}] Already exists — skipping.")
            skipped += 1
            continue

        # Load ONE band at a time — peak memory = one 6800x9000 int8 array (~58MB)
        data  = ds["burned_area"].sel(band=label).values
        layer = (data > 0).astype(np.int8)   # ensure binary

        save_monthly_nc(layer, lats, lons, label, "MODIS", logger)
        saved += 1

        # Explicitly free memory
        del data, layer

    ds.close()
    logger.info(f"  Phase 1 done. Saved: {saved}  Skipped: {skipped}")
    return lats, lons


# ---------------------------------------------------------------------------
# PHASE 2 — DOWNLOAD VIIRS DAILY CSVs
# ---------------------------------------------------------------------------

def get_firms_availability(logger):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{MAP_KEY}/all"
    logger.info("Querying FIRMS data availability ...")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df    = pd.read_csv(StringIO(r.text))
            avail = {}
            for _, row in df.iterrows():
                avail[row["data_id"]] = (
                    pd.to_datetime(row["min_date"]).date(),
                    pd.to_datetime(row["max_date"]).date(),
                )
            for k, (mn, mx) in avail.items():
                if "VIIRS" in k:
                    logger.info(f"  {k:<22}  {mn}  ->  {mx}")
            return avail
        except Exception as e:
            logger.warning(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(10)
    logger.error("Could not reach FIRMS API. Exiting.")
    sys.exit(1)


def pick_source(sensor_name, day, availability):
    sp_key = f"VIIRS_{sensor_name}_SP"
    if sensor_name != "NOAA21" and sp_key in availability:
        sp_min, sp_max = availability[sp_key]
        if sp_min <= day <= sp_max:
            return SENSOR_SOURCES[sensor_name]["SP"]
    nrt_key = f"VIIRS_{sensor_name}_NRT"
    if nrt_key in availability:
        nrt_min, nrt_max = availability[nrt_key]
        if nrt_min <= day <= nrt_max:
            return SENSOR_SOURCES[sensor_name]["NRT"]
    return None


def download_one_day(sensor_name, day, source_id, out_dir, logger):
    date_str = day.strftime("%Y%m%d")
    out_path = out_dir / f"VIIRS_{sensor_name}_{date_str}.csv"
    if out_path.exists():
        return True

    url = f"{FIRMS_URL}/{MAP_KEY}/{source_id}/{BBOX}/1/{day.strftime('%Y-%m-%d')}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 400:
                logger.warning(f"  {sensor_name} {date_str}: 400 (out of range)")
                return False
            r.raise_for_status()
            if "text/html" in r.headers.get("Content-Type", ""):
                logger.error(f"  {sensor_name} {date_str}: got HTML — check MAP_KEY")
                return False
            df = pd.read_csv(StringIO(r.text))
            df.to_csv(out_path, index=False)
            logger.info(f"  Downloaded {sensor_name} {date_str}: {len(df):>6,} hotspots")
            return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"  {sensor_name} {date_str}: attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
    logger.error(f"  {sensor_name} {date_str}: all retries failed")
    return False


def phase2_download_viirs(logger):
    logger.info("=" * 65)
    logger.info("PHASE 2 — Downloading VIIRS daily CSVs")
    logger.info(f"  Range : {VIIRS_START_DATE}  ->  {END_DATE}")
    logger.info("=" * 65)

    availability = get_firms_availability(logger)

    for sensor_name, folder, sensor_start in SENSORS:
        if END_DATE < sensor_start:
            logger.info(f"Skipping {sensor_name} — before sensor start ({sensor_start})")
            continue
        out_dir         = RAW_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        effective_start = max(VIIRS_START_DATE, sensor_start)
        logger.info(f"--- {sensor_name}  ({effective_start} -> {END_DATE}) ---")
        current = effective_start
        while current <= END_DATE:
            source_id = pick_source(sensor_name, current, availability)
            if source_id:
                download_one_day(sensor_name, current, source_id, out_dir, logger)
                time.sleep(REQUEST_DELAY)
            else:
                logger.warning(f"  {sensor_name} {current}: no FIRMS source covers this date")
            current += timedelta(days=1)

    logger.info("Phase 2 complete.")


# ---------------------------------------------------------------------------
# PHASE 3 — BUILD REFERENCE GRID TREE
# ---------------------------------------------------------------------------

def phase3_build_tree(lats, lons, logger):
    logger.info("=" * 65)
    logger.info("PHASE 3 — Building reference grid lookup tree")
    logger.info("=" * 65)

    lat_size = len(lats)
    lon_size = len(lons)
    logger.info(f"  Grid : {lat_size} x {lon_size} = {lat_size * lon_size:,} cells")

    lat_2d, lon_2d = np.meshgrid(lats, lons, indexing="ij")
    coords_flat    = np.column_stack((lat_2d.ravel(), lon_2d.ravel()))

    logger.info("  Building cKDTree (~30 sec) ...")
    tree = cKDTree(coords_flat)
    logger.info("  cKDTree ready.")
    return tree, lat_size, lon_size


# ---------------------------------------------------------------------------
# PHASE 4 — PROCESS VIIRS INTO MONTHLY BINARY LAYERS
# ---------------------------------------------------------------------------

def get_viirs_months():
    months = []
    y, m   = VIIRS_START_DATE.year, VIIRS_START_DATE.month
    while (y, m) <= (END_DATE.year, END_DATE.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def load_month_hotspots(year, month, logger):
    month_start = date(year, month, 1)
    month_end   = (date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
    check_start = month_start - timedelta(days=1)
    check_end   = month_end   + timedelta(days=1)

    all_dfs = []
    for sensor_name, folder, sensor_start in SENSORS:
        if check_end < sensor_start:
            continue
        sensor_dir = RAW_DIR / folder
        current    = max(check_start, sensor_start)
        while current <= check_end:
            fpath = sensor_dir / f"VIIRS_{sensor_name}_{current.strftime('%Y%m%d')}.csv"
            if fpath.exists():
                try:
                    df = pd.read_csv(fpath, usecols=["latitude", "longitude", "acq_date", "acq_time"])
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    logger.warning(f"  Could not read {fpath.name}: {e}")
            current += timedelta(days=1)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)

    combined["acq_time_str"] = combined["acq_time"].astype(int).astype(str).str.zfill(4)
    combined["utc_dt"]       = pd.to_datetime(
        combined["acq_date"] + " " + combined["acq_time_str"],
        format="%Y-%m-%d %H%M", utc=True
    )
    combined["aedt_dt"]    = combined["utc_dt"] + pd.Timedelta(hours=UTC_OFFSET_HOURS)
    combined["aedt_year"]  = combined["aedt_dt"].dt.year
    combined["aedt_month"] = combined["aedt_dt"].dt.month
    combined["aedt_date"]  = combined["aedt_dt"].dt.date

    mask     = (combined["aedt_year"] == year) & (combined["aedt_month"] == month)
    combined = combined[mask][["latitude", "longitude", "aedt_date"]].rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )
    if combined.empty:
        return None

    logger.info(f"  {len(combined):>7,} hotspot observations loaded")
    return combined


def make_binary_layer(df, tree, lat_size, lon_size, logger):
    coords       = df[["lat", "lon"]].values
    _, flat_idx  = tree.query(coords, k=1)
    df           = df.copy()
    df["cell"]   = flat_idx
    unique_cells = df.drop_duplicates(subset=["cell", "aedt_date"])["cell"].values
    layer        = np.zeros(lat_size * lon_size, dtype=np.int8)
    layer[unique_cells] = 1
    layer        = layer.reshape(lat_size, lon_size)
    logger.info(f"  {int(layer.sum()):,} unique 500m cells burned this month")
    return layer


def phase4_process_viirs(lats, lons, tree, lat_size, lon_size, logger):
    logger.info("=" * 65)
    logger.info("PHASE 4 — Processing VIIRS monthly binary layers")
    logger.info("=" * 65)

    existing       = get_existing_monthly_files()
    all_months     = get_viirs_months()
    missing_months = [(y, m) for y, m in all_months
                      if f"{y}-{m:02d}" not in existing]

    logger.info(f"  Total VIIRS months  : {len(all_months)}")
    logger.info(f"  Already processed   : {len(all_months) - len(missing_months)}  (skipping)")
    logger.info(f"  To process now      : {len(missing_months)}")

    for i, (year, month) in enumerate(missing_months, 1):
        label = f"{year}-{month:02d}"
        logger.info(f"[{i}/{len(missing_months)}] {label} ...")
        df    = load_month_hotspots(year, month, logger)
        if df is None or df.empty:
            logger.info(f"  No data for {label} — saving empty layer.")
            layer = np.zeros((lat_size, lon_size), dtype=np.int8)
        else:
            layer = make_binary_layer(df, tree, lat_size, lon_size, logger)
        save_monthly_nc(layer, lats, lons, label, "VIIRS", logger)
        del layer

    logger.info("Phase 4 complete.")


# ---------------------------------------------------------------------------
# PHASE 5 — BUILD FINAL MERGED CUBE (streaming, low memory)
# ---------------------------------------------------------------------------

def phase5_build_merged_cube(lats, lons, logger):
    """
    Collect all BA_YYYY-MM.nc files (MODIS + VIIRS), sort chronologically,
    and write the final cube one band at a time using netCDF4 directly.
    This avoids loading all bands into memory simultaneously.
    """
    logger.info("=" * 65)
    logger.info("PHASE 5 — Building final merged MODIS + VIIRS cube")
    logger.info("=" * 65)

    all_files = sorted(MONTHLY_DIR.glob("BA_????-??.nc"))
    if not all_files:
        logger.error("No monthly files found — cannot build cube.")
        return

    all_labels = [f.stem.replace("BA_", "") for f in all_files]

    # Separate MODIS and VIIRS labels for reporting
    modis_labels = [l for l in all_labels if l <= MODIS_END]
    viirs_labels = [l for l in all_labels if l > MODIS_END]

    logger.info(f"  MODIS bands  : {len(modis_labels)}  ({modis_labels[0]} -> {modis_labels[-1]})")
    logger.info(f"  VIIRS months : {len(viirs_labels)}  ({viirs_labels[0] if viirs_labels else 'none'} -> {viirs_labels[-1] if viirs_labels else 'none'})")
    logger.info(f"  Total bands  : {len(all_labels)}  ({all_labels[0]} -> {all_labels[-1]})")
    logger.info(f"  Writing -> {OUTPUT_CUBE}")

    # Use netCDF4 directly to write one band at a time (avoids loading all into RAM)
    import netCDF4 as nc4

    lat_size = len(lats)
    lon_size = len(lons)
    n_bands  = len(all_labels)

    with nc4.Dataset(OUTPUT_CUBE, "w", format="NETCDF4") as ncout:
        # Dimensions
        ncout.createDimension("band", n_bands)
        ncout.createDimension("lat",  lat_size)
        ncout.createDimension("lon",  lon_size)

        # Coordinate variables
        v_band = ncout.createVariable("band", str,  ("band",))
        v_lat  = ncout.createVariable("lat",  "f4", ("lat",))
        v_lon  = ncout.createVariable("lon",  "f4", ("lon",))

        v_lat[:]  = lats
        v_lon[:]  = lons
        for i, label in enumerate(all_labels):
            v_band[i] = label

        v_lat.units         = "degrees_north"
        v_lat.long_name     = "Latitude"
        v_lon.units         = "degrees_east"
        v_lon.long_name     = "Longitude"

        # Main data variable — chunked and compressed
        v_ba = ncout.createVariable(
            "burned_area", "i1", ("band", "lat", "lon"),
            zlib=True, complevel=5,
            chunksizes=(1, 500, 500),
            fill_value=0
        )
        v_ba.long_name  = "Binary burned area"
        v_ba.units      = "1=burned 0=unburned"
        v_ba.comment    = (
            "MODIS period (up to 2012-01): burned_cells resampled to 500m grid. "
            "VIIRS period (2012-02 onwards): 1=any VIIRS hotspot in 500m cell this month. "
            "UTC+10. Sensors merged and deduplicated at 500m/daily resolution."
        )

        # Global attributes
        ncout.title       = "MODIS + VIIRS Monthly Binary Burned Area Data Cube (500m)"
        ncout.institution = "WSU"
        ncout.source      = (
            "MODIS burned area (2000-11 to 2012-01) merged with "
            "NASA FIRMS VIIRS SNPP + NOAA-20 + NOAA-21 hotspots (2012-02 onwards)"
        )
        ncout.history     = "Created using modis_viirs_pipeline.py"
        ncout.CRS         = "EPSG:4326 - WGS84"
        ncout.band_start  = all_labels[0]
        ncout.band_end    = all_labels[-1]
        ncout.modis_end   = MODIS_END
        ncout.viirs_start = viirs_labels[0] if viirs_labels else "N/A"

        # Write one band at a time — peak memory = one 6800x9000 int8 array (~58MB)
        for i, (f, label) in enumerate(zip(all_files, all_labels)):
            ds            = xr.open_dataset(f)
            layer         = ds["burned_area"].values.astype(np.int8)
            v_ba[i, :, :] = layer
            ds.close()
            del layer
            if (i + 1) % 20 == 0 or (i + 1) == n_bands:
                logger.info(f"  Written {i + 1}/{n_bands} bands ...")

    logger.info(f"  Final cube saved. Size: {OUTPUT_CUBE.stat().st_size / 1e6:.0f} MB")
    logger.info("Phase 5 complete.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    logger = setup_logging()

    logger.info("=" * 65)
    logger.info("MODIS + VIIRS MERGED BURNED AREA PIPELINE")
    logger.info(f"  MODIS source : {MODIS_CUBE}")
    logger.info(f"  MODIS period : up to {MODIS_END}")
    logger.info(f"  VIIRS period : {VIIRS_START_DATE}  ->  {END_DATE}")
    logger.info(f"  Output       : {OUTPUT_CUBE}")
    logger.info("=" * 65)

    # Phase 1: Extract MODIS bands one at a time -> BA_YYYY-MM.nc files
    lats, lons = phase1_extract_modis(logger)

    # Phase 2: Download VIIRS daily CSVs (needs internet -> copyq)
    phase2_download_viirs(logger)

    # Phase 3: Build cKDTree once (uses ~2GB RAM, freed after this phase)
    tree, lat_size, lon_size = phase3_build_tree(lats, lons, logger)

    # Phase 4: Process VIIRS months one at a time -> BA_YYYY-MM.nc files
    phase4_process_viirs(lats, lons, tree, lat_size, lon_size, logger)

    # Phase 5: Stream all monthly files into final cube (one band at a time)
    phase5_build_merged_cube(lats, lons, logger)

    logger.info("=" * 65)
    logger.info("PIPELINE COMPLETE.")
    logger.info(f"Output -> {OUTPUT_CUBE}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
