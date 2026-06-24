import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from gstools import SRF, Gaussian
from scipy.stats import weibull_min, norm
from joblib import Parallel, delayed
import dask.array as da

# --- Config ---
MAX_TSF0 = 600
n_simulations = 20
tile_size = 2000
n_jobs = 16  # Match PBS ncpus

output_path = "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/21.TSF/TSF_nc_cube_005_Aus_Gaussian_Copula_native_vege.nc"
checkpoint_dir = "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/21.TSF/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Input Files ---
modis_path = "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/10.MODIS_VIIRS_BA/MODIS_VIIRS_BA_005_Aus.nc"
pyro_path = "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/17.pyroregions/pyroregions.nc"
weibull_csv = "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/21.TSF/0.Weibull_veriogram/5_weibull_params_NATIVE_VEG_ONLY.csv"
variogram_csv = "/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/21.TSF/0.Weibull_veriogram/1_Variogram_ranges_pyroregion_all.csv"

# --- Load datasets lazily ---
burned_da = xr.open_dataset(modis_path, chunks={'lat': tile_size, 'lon': tile_size})["burned_area"].astype(np.uint8)
pyro_da = xr.open_dataset(pyro_path)["pyroregion"]
lat, lon = burned_da.lat, burned_da.lon
n_time = burned_da.sizes["band"]
nlat, nlon = len(lat), len(lon)

# --- Load Weibull and variogram parameters ---
weibull_df = pd.read_csv(weibull_csv)
variogram_df = pd.read_csv(variogram_csv)
param_df = pd.merge(weibull_df, variogram_df, on="pyroregion_id")
param_df["pyroregion_id"] = param_df["pyroregion_id"].astype(int)
param_lookup = param_df.set_index("pyroregion_id").to_dict("index")


# --- KEY OPTIMISATION: SRF on bounding box only, not full tile ---
def simulate_tsf0(pyro_np, first_fire, param_lookup, seed):
    """
    Simulate TSF0 using Gaussian copula with Weibull marginals.
    Critically, the spatial random field is simulated only over the
    bounding box of each pyroregion, not the full tile, cutting
    computation dramatically for small/fragmented regions.
    """
    np.random.seed(seed)
    ny, nx = pyro_np.shape
    tsf0 = np.full((ny, nx), np.nan, dtype=np.float32)

    for pid in np.unique(pyro_np):
        if pid == -1 or np.isnan(float(pid)) or int(pid) not in param_lookup:
            continue

        p = param_lookup[int(pid)]
        rho, lam, range_km = p["shape_rho"], p["scale_lambda"], p["range_km"]
        mask = (pyro_np == pid)

        if np.sum(mask) < 10:
            continue

        # --- Bounding box of this pyroregion within the tile ---
        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        bb_h = r1 - r0
        bb_w = c1 - c0

        # Simulate Gaussian field only over bounding box
        model = Gaussian(dim=2, var=1.0, len_scale=range_km)
        srf = SRF(model, seed=seed)
        grid_x, grid_y = np.meshgrid(np.arange(bb_w), np.arange(bb_h))
        field_bb = srf((grid_x, grid_y)).reshape(bb_h, bb_w)

        # Extract values only at mask positions within bounding box
        local_mask = mask[r0:r1, c0:c1]
        z_vals = field_bb[local_mask]
        u_vals = norm.cdf(z_vals)
        d_vals = first_fire[mask]

        # Vectorised truncated Weibull inversion
        cdf_d = weibull_min.cdf(d_vals, rho, scale=lam)
        u_adj = np.clip(u_vals * (1 - cdf_d) + cdf_d, 0.0, 1.0 - 1e-9)
        tsf_vals = weibull_min.ppf(u_adj, rho, scale=lam)
        tsf_vals = np.clip(tsf_vals + d_vals, d_vals, MAX_TSF0)

        tsf0[mask] = tsf_vals

    return tsf0


def checkpoint_path(i, j):
    return os.path.join(checkpoint_dir, f"tile_{i}_{j}.npy")


def tile_is_done(i, j):
    return os.path.exists(checkpoint_path(i, j))


# --- Main tile loop with checkpointing ---
lat_starts = list(range(0, nlat, tile_size))
lon_starts = list(range(0, nlon, tile_size))
total_tiles = len(lat_starts) * len(lon_starts)

print(f"Grid: {nlat}×{nlon}, tile_size={tile_size}, total tiles={total_tiles}")

for i in tqdm(lat_starts, desc="Tiles (lat)"):
    for j in lon_starts:
        i2 = min(i + tile_size, nlat)
        j2 = min(j + tile_size, nlon)

        # --- Resume: skip completed tiles ---
        if tile_is_done(i, j):
            print(f"  Skipping tile ({i},{j}) — checkpoint found.")
            continue

        burned_tile = burned_da.isel(
            band=slice(0, n_time), lat=slice(i, i2), lon=slice(j, j2)
        ).compute()
        pyro_tile = pyro_da.isel(lat=slice(i, i2), lon=slice(j, j2)).compute()

        burned_np = burned_tile.values
        pyro_np = pyro_tile.values.astype(float)

        # Skip ocean/empty tiles
        if np.all(np.isnan(burned_np[0])):
            # Write a sentinel so we skip on resume too
            np.save(checkpoint_path(i, j), np.array(["SKIP"]))
            continue

        th, tw = i2 - i, j2 - j

        # Compute first-fire timestep per pixel
        first_fire = np.argmax(burned_np == 1, axis=0).astype(np.float32)
        never_burned = ~np.any(burned_np == 1, axis=0)
        first_fire[never_burned] = n_time

        # --- Parallel simulations (n_jobs matches PBS ncpus) ---
        tsf0_stack = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(simulate_tsf0)(pyro_np, first_fire, param_lookup, seed)
            for seed in range(n_simulations)
        )

        tsf0_mean = np.nanmean(tsf0_stack, axis=0)  # (th, tw)

        # --- Build TSF time series for this tile ---
        tsf_tile = np.full((n_time, th, tw), np.nan, dtype=np.float32)
        tsf_tile[0] = tsf0_mean

        for t in range(1, n_time):
            burned_mask = burned_np[t] == 1
            prev = tsf_tile[t - 1]
            tsf_tile[t] = np.where(burned_mask, 0.0, prev + 1.0)

        # --- Save checkpoint ---
        np.save(checkpoint_path(i, j), tsf_tile)
        print(f"  Tile ({i},{j}) done and saved.")
        del tsf0_stack, tsf0_mean, tsf_tile, burned_np, pyro_np

# -----------------------------------------------------------------------
# Assembly: read all checkpoints and write the final NetCDF
# -----------------------------------------------------------------------
print("\n🧩 Assembling tiles into final NetCDF...")

tsf_cube = np.full((n_time, nlat, nlon), np.nan, dtype=np.float32)

for i in lat_starts:
    for j in lon_starts:
        i2 = min(i + tile_size, nlat)
        j2 = min(j + tile_size, nlon)
        cp = checkpoint_path(i, j)
        if not os.path.exists(cp):
            print(f"  WARNING: missing checkpoint for tile ({i},{j}), leaving NaN.")
            continue
        data = np.load(cp, allow_pickle=True)
        if data.shape == () or (data.ndim == 1 and data[0] == "SKIP"):
            continue  # ocean / empty tile
        tsf_cube[:, i:i2, j:j2] = data

print("💾 Saving final TSF NetCDF...")
ds_out = xr.Dataset(
    {"tsf": (("time", "lat", "lon"), tsf_cube)},
    coords={"time": burned_da.band.values, "lat": lat, "lon": lon},
)
ds_out.to_netcdf(
    output_path,
    encoding={"tsf": {"zlib": True, "complevel": 4}},
)
print(f"✅ TSF NetCDF saved: {output_path}")
