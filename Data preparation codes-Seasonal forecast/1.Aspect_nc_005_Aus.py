import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load the elevation data from netCDF
elevation_nc_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/1.Elevation/Elevation_nc_005_Aus.nc'
ds = xr.open_dataset(elevation_nc_path)
elevation_data = ds['Elevation']

# Calculate gradients
grad_y, grad_x = np.gradient(elevation_data)

# Calculate aspect in degrees, normalised to [0, 360)
aspect = np.arctan2(grad_y, -grad_x)
aspect_degrees = np.degrees(aspect)
aspect_degrees = (aspect_degrees + 360) % 360

# Convert to radians and compute circular components
aspect_radians = np.radians(aspect_degrees)
aspect_sin = np.sin(aspect_radians)
aspect_cos = np.cos(aspect_radians)

# Round to 3 decimal places
aspect_degrees = np.round(aspect_degrees, 3)
aspect_sin    = np.round(aspect_sin, 3)
aspect_cos    = np.round(aspect_cos, 3)

# --- Plot (degrees) ---
plt.figure(figsize=(10, 6))
plt.imshow(aspect_degrees, cmap='twilight',
           extent=[ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()],
           origin='lower')
plt.colorbar(label='Aspect (degrees)')
plt.title('Aspect Map Australia')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().invert_yaxis()
# plt.savefig('/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/3.Aspect/1_Aspect_nc_005_Aus.png', dpi=300)

output_dir = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/3.Aspect/'

# Shared coordinate dict
coords = {"lat": ds.lat, "lon": ds.lon}

# 1) Degrees version
aspect_deg_ds = xr.Dataset(
    {"aspect_degrees": (["lat", "lon"], aspect_degrees)},
    coords=coords
)
aspect_deg_ds.attrs['description'] = 'Terrain aspect calculated from elevation data'
aspect_deg_ds.attrs['units'] = 'degrees, range [0, 360)'
aspect_deg_ds.to_netcdf(output_dir + 'Aspect_degrees_nc_005_Aus.nc')
print("Saved: Aspect_degrees_nc_005_Aus.nc")

# 2) Sin version
aspect_sin_ds = xr.Dataset(
    {"aspect_sin": (["lat", "lon"], aspect_sin)},
    coords=coords
)
aspect_sin_ds.attrs['description'] = 'Sine of terrain aspect (circular encoding), derived from elevation data'
aspect_sin_ds.attrs['units'] = 'dimensionless, range [-1, 1]'
aspect_sin_ds.to_netcdf(output_dir + 'Aspect_sin_nc_005_Aus.nc')
print("Saved: Aspect_sin_nc_005_Aus.nc")

# 3) Cos version
aspect_cos_ds = xr.Dataset(
    {"aspect_cos": (["lat", "lon"], aspect_cos)},
    coords=coords
)
aspect_cos_ds.attrs['description'] = 'Cosine of terrain aspect (circular encoding), derived from elevation data'
aspect_cos_ds.attrs['units'] = 'dimensionless, range [-1, 1]'
aspect_cos_ds.to_netcdf(output_dir + 'Aspect_cos_nc_005_Aus.nc')
print("Saved: Aspect_cos_nc_005_Aus.nc")

print("\nAll three aspect NetCDF files written to:", output_dir)