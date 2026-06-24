import xarray as xr
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Load the location data from the NetCDF file
nc_file = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc'
locations_ds = xr.open_dataset(nc_file)

# Extract latitudes and longitudes
lats = locations_ds['lat'].values
lons = locations_ds['lon'].values

# Load the pyroregions shapefile using GeoPandas
pyroregions_shp = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/17.pyroregions/0.Raw_data/Australias_pyroregions.shp'
pyroregions_gdf = gpd.read_file(pyroregions_shp)

# Add the pyroregion_id column for identification
pyroregions_gdf['pyroregion_id'] = np.arange(len(pyroregions_gdf))

# Create a spatial index for the pyroregions to speed up spatial queries
spatial_index = pyroregions_gdf.sindex  # Spatial index

# Create a meshgrid of latitudes and longitudes for all locations
lon_grid, lat_grid = np.meshgrid(lons, lats)
locations = np.vstack((lat_grid.flatten(), lon_grid.flatten())).T

# Create a GeoSeries of all locations to use spatial join
points = [Point(lon, lat) for lat, lon in locations]
locations_gdf = gpd.GeoSeries(points, crs='EPSG:4326')

# Initialize an empty list for the pyroregion assignments
assigned_regions = []

# Loop over each point, using spatial index for efficient region assignment
for point in locations_gdf:
    possible_matches_index = list(spatial_index.intersection(point.bounds))  # Fast filtering using bounds
    matched_region = np.nan  # Default value is NaN
    
    # Check if the point is inside any of the pyroregions
    for index in possible_matches_index:
        region = pyroregions_gdf.iloc[index]  # Access the original GeoDataFrame row
        if region['geometry'].contains(point):
            matched_region = region['pyroregion_id']  # Assign the region id
            break
    
    assigned_regions.append(matched_region)

# Reshape the assignments to match the grid structure
pyroregion_array = np.array(assigned_regions).reshape(lat_grid.shape)

# Create a new xarray Dataset for the pyroregions
pyroregion_ds = xr.Dataset(
    {
        'pyroregion': (['lat', 'lon'], pyroregion_array),
        'lat': (['lat'], lats),
        'lon': (['lon'], lons),
    },
    attrs={
        'title': 'pyroregion Data',
        'institution': 'WSU',
        'source': 'Synthetic generation from location and pyroregion',
        'history': 'Created using Python',
        'CRS': 'EPSG:4326 - WGS84',
    }
)

# Save the resulting dataset to a new NetCDF file
output_file = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/17.pyroregions/pyroregions.nc'
pyroregion_ds.to_netcdf(output_file)

# Optionally, print a few random records from the result
print(np.random.choice(len(pyroregion_array.flatten()), size=2000))
