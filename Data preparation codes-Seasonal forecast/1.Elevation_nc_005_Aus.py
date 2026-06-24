import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
import xarray as xr
from datetime import datetime
from scipy.spatial import cKDTree

# Path to the original ESRI grid file
adf_file_path = r'/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/1.Elevation/0.Raw_data/w001001.adf'

# Open the original raster file
with rasterio.open(adf_file_path) as src:
    # Define the bounds and resolution for the new grid
    lat_start, lat_end = -44.0, -10.0  # Southern latitude should be first if it's smaller
    lon_start, lon_end = 110.0, 155.0
    new_resolution = 0.005

    # Calculate the transform and dimensions for the new grid
    transform, width, height = calculate_default_transform(
        src.crs, src.crs,
        width=src.width, height=src.height,
        left=lon_start, bottom=min(lat_start, lat_end), right=lon_end, top=max(lat_start, lat_end),
        dst_width=int((lon_end - lon_start) / new_resolution),
        dst_height=int(abs(lat_end - lat_start) / new_resolution)
    )

    # Create a new array to hold the resampled data
    data_resampled = np.zeros((height, width), dtype=src.dtypes[0])

    # Reproject and resample the data
    reproject(
        source=rasterio.band(src, 1),
        destination=data_resampled,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=src.crs,
        resampling=Resampling.average
    )

# Load the locations dataset
location_ds = xr.open_dataset('/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc')

# Generate 2D arrays for latitude and longitude coordinates for elevation
lat_elev, lon_elev = np.meshgrid(np.linspace(lat_start, lat_end, height), np.linspace(lon_start, lon_end, width), indexing='ij')
elevation_flat = data_resampled.flatten()

# Convert elevation data to float to allow NaN assignment
elevation_flat = elevation_flat.astype(float)

# Replace -32768 values with NaN to represent missing data
elevation_flat[elevation_flat == -32768] = np.nan

# Creating coordinate pairs from meshed latitude and longitude arrays
coords_elevation = np.column_stack((lat_elev.ravel(), lon_elev.ravel()))
tree = cKDTree(coords_elevation)

# Assuming location_ds also requires meshing of lat and lon
lat_loc, lon_loc = np.meshgrid(location_ds.lat.values, location_ds.lon.values, indexing='ij')
coords_loc = np.column_stack((lat_loc.ravel(), lon_loc.ravel()))

# Find nearest neighbors
_, indices = tree.query(coords_loc, k=1)  # k=1 for the nearest neighbor

# Get elevation values corresponding to the nearest indices
nearest_elevations = elevation_flat[indices]

# Reshape nearest_elevations to match the lat/lon grid dimensions of location_ds
nearest_elevations_reshaped = nearest_elevations.reshape(lat_loc.shape)

# Create a new dataset with the matched elevation values
new_ds = xr.Dataset(
    {
        'Elevation': (('lat', 'lon'), nearest_elevations_reshaped)
    },
    coords={
        'lat': location_ds.lat,
        'lon': location_ds.lon
    }
)

# Flip the elevation data along the latitude axis if it's reversed
elevation_vals_corrected = np.flip(new_ds['Elevation'].values, axis=0)  # Flipping along the latitude axis

# Update the 'Elevation' variable in the dataset with the corrected values
new_ds['Elevation'] = xr.DataArray(elevation_vals_corrected, coords=[new_ds['lat'], new_ds['lon']], dims=["lat", "lon"])

# Define global attributes for the new dataset
new_ds.attrs = {
    'title': 'Combined Elevation Data',
    'institution': 'WSU',
    'source': 'Synthetic generation from location and elevation data',
    'history': f'Created using Python with xarray and scipy.spatial.cKDTree on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    'comment': 'Elevation matched to nearest location points, corrected for latitude flip',
    'CRS': 'EPSG:4326 - WGS84'
}

# Save the corrected netCDF file directly to the final location
final_nc_file_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/1.Elevation/Elevation_nc_005_Aus.nc'
new_ds.to_netcdf(final_nc_file_path)

# Confirm save path
print(f"Corrected netCDF file saved to: {final_nc_file_path}")
