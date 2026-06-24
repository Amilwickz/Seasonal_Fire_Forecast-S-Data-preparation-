import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import xarray as xr
from scipy.spatial import cKDTree
import time  # Import the time module

# --- Step 1: Resample the TIF file ---
input_file_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/8.Vegetation/0.Raw_data/1_reprojected_MVG500m.tif'
output_file_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/8.Vegetation/1_Resampling_raw_vege_data_TIF.TIF'

with rasterio.open(input_file_path) as src:
    # Set the bounds and resolution for the new grid
    dst_crs = src.crs  # Same as source CRS, which is EPSG:4326
    new_width = int((155.0 - 110.0) / 0.005)  # Calculate new width
    new_height = int(((-10) - (-44.0)) / 0.005)  # Calculate new height
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, new_width, new_height,
        left=110.0, bottom=-44.0, right=155.0, top=-10.0
    )

    # Prepare new metadata for the output file
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Perform the resampling
    with rasterio.open(output_file_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):  # Assuming there's only one band
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

print("Resampling complete. Saved to:", output_file_path)

# --- Step 2: Read the Resampled TIF directly into memory (no intermediate NetCDF) ---
with rasterio.open(output_file_path) as src:
    data = src.read(1)  # Read the raster data
    transform = src.transform  # Affine transform to calculate lat/lons

    # Define the longitude and latitude arrays correctly
    lon = np.arange(src.bounds.left, src.bounds.right, abs(transform[0]))
    lat = np.arange(src.bounds.top, src.bounds.bottom, -abs(transform[4]))

    # Replace -2147483648 with 99 for 'No Data' value
    vegetation_data = np.copy(data)
    vegetation_data[vegetation_data == -2147483648] = 99

print("TIF data read into memory. Proceeding directly to nearest-neighbour matching.")

# --- Step 3: Match Nearest Vegetation to Location Points ---
location_ds = xr.open_dataset('/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc')

# Generate 2D arrays for latitude and longitude coordinates from in-memory arrays
lat_veg, lon_veg = np.meshgrid(lat, lon, indexing='ij')
vegetation_flat = vegetation_data.flatten()

# Creating coordinate pairs from meshed latitude and longitude arrays
coords_vegetation = np.column_stack((lat_veg.ravel(), lon_veg.ravel()))
tree = cKDTree(coords_vegetation)

# Assuming location_ds also requires meshing of lat and lon
lat_loc, lon_loc = np.meshgrid(location_ds.lat.values, location_ds.lon.values, indexing='ij')
coords_loc = np.column_stack((lat_loc.ravel(), lon_loc.ravel()))

# Find nearest neighbors
_, indices = tree.query(coords_loc, k=1)  # k=1 for the nearest neighbor

# Get vegetation type values corresponding to the nearest indices
nearest_vegetation_types = vegetation_flat[indices]

# Reshape nearest_vegetation_types to match the lat/lon grid dimensions of location_ds
nearest_vegetation_types_reshaped = nearest_vegetation_types.reshape(lat_loc.shape)

# Create a new dataset
new_ds = xr.Dataset(
    {
        'Vegetation_Type': (('lat', 'lon'), nearest_vegetation_types_reshaped)
    },
    coords={
        'lat': location_ds.lat,
        'lon': location_ds.lon
    }
)

# Define global attributes
new_ds.attrs = {
    'title': 'Combined Vegetation Type Data',
    'institution': 'WSU',
    'source': 'Synthetic generation from location and vegetation type data',
    'history': 'Created using Python with xarray and scipy.spatial.cKDTree',
    'comment': 'Vegetation type matched to nearest location points',
    'CRS': 'EPSG:4326 - WGS84'
}

# Save to a NetCDF file
new_ds.to_netcdf('/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/8.Vegetation/Vegetation_nc_Aus_005.nc')

print("Combined vegetation types saved to NetCDF.")
