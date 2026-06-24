import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from haversine import haversine, Unit

# Load the road lines from the shapefile
road_lines_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/7.Roads/0.Raw_data/MajorRoads2024_12.shp'
road_lines = gpd.read_file(road_lines_path).to_crs(epsg=3577)  # Convert to Australian Albers for consistency in measurements

# Load the Australia shapefile
aus_shapefile_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/AUS_2021_AUST_SHP_GDA2020/AUS_2021_AUST_GDA2020.shp'
australia_map = gpd.read_file(aus_shapefile_path).to_crs(epsg=3577)  # Ensure same CRS as road lines for plotting

# Function to interpolate points every 250 meters along a LineString
def interpolate_points(line, interval=250):
    num_segments = int(np.floor(line.length / interval))
    points = [line.interpolate(distance) for distance in np.linspace(0, line.length, num=num_segments)]
    return points

# Apply the function to each line in the GeoDataFrame and flatten the list of lists
point_list = [point for geom in road_lines.geometry for point in interpolate_points(geom)]

# Create a new GeoDataFrame from the list of points
points_geo = gpd.GeoDataFrame(geometry=gpd.GeoSeries(point_list), crs='EPSG:3577')

# Convert the CRS to EPSG:4326 to get latitude and longitude
points_geo = points_geo.to_crs(epsg=4326)

# Extracting and rounding latitude and longitude
points_geo['latitude'] = points_geo.geometry.y.round(3)
points_geo['longitude'] = points_geo.geometry.x.round(3)

# Code 1: Plotting the points on the map of Australia
fig, ax = plt.subplots(figsize=(10, 10))
australia_map.plot(ax=ax, color='lightgrey')  # Plot the Australia map as a background
points_geo.plot(ax=ax, marker='o', color='red', markersize=5)  # Plot the points

# Setting titles and labels
ax.set_title('Road Network Points on Australia Map', fontsize=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Save the map as a PNG file
#plt.savefig('/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/7.Roads/1_Roadline_points_Map.png', dpi=300)
#plt.show()

# Code 2: Calculate distances to the nearest road points
# Load the location layer NetCDF
nc_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc'
locations_layer = xr.open_dataset(nc_path)

# Prepare coordinates for KDTree
coordinates = np.deg2rad(points_geo[['latitude', 'longitude']].values)  # Convert to radians for KDTree
tree = cKDTree(coordinates)

# Define function to calculate distances
def calculate_nearest(lat, lon):
    # Find the nearest location in the tree (using haversine distance)
    distance, location_idx = tree.query(np.deg2rad([lat, lon]), k=1)
    # Calculate actual haversine distance
    nearest_point = points_geo.iloc[location_idx]
    haversine_dist = haversine((lat, lon), (nearest_point['latitude'], nearest_point['longitude']), unit=Unit.KILOMETERS)
    return haversine_dist

# Apply function to each coordinate in the dataset
distance_array = np.zeros((locations_layer.sizes['lat'], locations_layer.sizes['lon']))
for i, lat in enumerate(locations_layer.lat.values):
    for j, lon in enumerate(locations_layer.lon.values):
        distance_array[i, j] = calculate_nearest(lat, lon)

# Create a new xarray DataArray with the computed distances
distance_da = xr.DataArray(distance_array, coords=[locations_layer.lat, locations_layer.lon], dims=['lat', 'lon'])
new_dataset = xr.Dataset({'distance': distance_da})
new_dataset.attrs = locations_layer.attrs  # Copy metadata if necessary

# Set CRS attribute
new_dataset.distance.attrs['crs'] = 'EPSG:4326'

# Save to a new NetCDF file
output_nc_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/7.Roads/Distance_to_major_roads_nc_Aus_005.nc'
new_dataset.to_netcdf(output_nc_path)

print("NetCDF with distance data saved successfully.")

# Close the dataset
locations_layer.close()
