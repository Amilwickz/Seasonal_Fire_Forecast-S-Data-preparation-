import xarray as xr
import numpy as np
import rasterio
from pyproj import Transformer

# Paths to the input files
nc_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc'
tif_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/4.Population/0.Raw_data/apg23r_1_0_0.tif'

# Open the netCDF file
nc_data = xr.open_dataset(nc_path)

# Open the GeoTIFF file
with rasterio.open(tif_path) as src:
    # Initialize the transformer for coordinate conversion (EPSG:4326 to the coordinate reference system of the raster)
    transformer = Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
    
    # Read the raster data
    raster_data = src.read(1)

    # Initialize an array for population data (default to NaN)
    population_data = np.full((nc_data.sizes['lat'], nc_data.sizes['lon']), np.nan, dtype=float)

    # Transform coordinates and fetch population data in batches (adjust batch size as needed)
    for i in range(0, nc_data.sizes['lat'], 100):
        end_i = min(i + 100, nc_data.sizes['lat'])
        for j in range(0, nc_data.sizes['lon'], 100):
            end_j = min(j + 100, nc_data.sizes['lon'])
            
            # Select slices of latitude and longitude
            lat_slice = nc_data.lat.isel(lat=slice(i, end_i))
            lon_slice = nc_data.lon.isel(lon=slice(j, end_j))
            
            # Create a meshgrid for latitude and longitude to transform into coordinates
            lon, lat = np.meshgrid(lon_slice, lat_slice)
            x, y = transformer.transform(lon.ravel(), lat.ravel())
            
            # Convert transformed coordinates to raster indices
            rows, cols = zip(*[src.index(x_val, y_val) for x_val, y_val in zip(x, y)])
            
            # Extract population data using raster indices (safe bounds check)
            for idx, (row, col) in enumerate(zip(rows, cols)):
                if (0 <= row < src.height) and (0 <= col < src.width):
                    idx_lat = idx // len(lon_slice)
                    idx_lon = idx % len(lon_slice)
                    population_data[i + idx_lat, j + idx_lon] = raster_data[row, col]

# Replace -1.0 with 0 in the population data
population_data = np.where(population_data != -1.0, population_data, 0)

# Create a new dataset with the population data
nc_data['population'] = (('lat', 'lon'), population_data)

# Save the corrected dataset
final_nc_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/4.Population/Population_nc_Aus_005.nc'
nc_data.to_netcdf(final_nc_path)

# Calculate and print the new max and min values
max_value = population_data.max()
min_value = population_data.min()
print(f"Corrected maximum population count: {max_value}")
print(f"Corrected minimum population count: {min_value}")

# Close the dataset
nc_data.close()

print(f"Population extraction and correction completed. Data saved to: {final_nc_path}")
