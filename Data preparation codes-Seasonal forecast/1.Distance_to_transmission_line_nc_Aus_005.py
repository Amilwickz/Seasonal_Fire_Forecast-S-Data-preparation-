import geopandas as gpd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from haversine import haversine, Unit
import matplotlib.pyplot as plt

# Step 1: Load the power lines shapefile
powerline_path = r'/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/5.PowerTransmission/0.Raw_data/Electricity_Transmission_Lines.shp'
power_lines = gpd.read_file(powerline_path)

# Set CRS to EPSG:3577 if not set
if power_lines.crs is None:
    power_lines.set_crs(epsg=7844, inplace=True)
power_lines = power_lines.to_crs(epsg=3577)

# Function to interpolate points every 250 meters along a LineString
def interpolate_points(line, interval=250):
    num_segments = int(np.floor(line.length / interval))
    points = [line.interpolate(distance) for distance in np.linspace(0, line.length, num=num_segments + 1)]
    return points

# Apply the function to each line in the GeoDataFrame and flatten the list of lists
point_list = [point for geom in power_lines.geometry for point in interpolate_points(geom)]

# Create a new GeoDataFrame from the list of points
points_geo = gpd.GeoDataFrame(geometry=gpd.GeoSeries(point_list), crs='EPSG:3577')

# Convert the CRS to EPSG:4326 to get latitude and longitude
points_geo = points_geo.to_crs(epsg=4326)

# Extracting and rounding latitude and longitude to three decimal places
points_geo['latitude'] = points_geo.geometry.y.round(3)
points_geo['longitude'] = points_geo.geometry.x.round(3)

# Step 2: Load the location layer NetCDF
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

# Step 3: Apply function to each coordinate in the dataset and create distance array
distance_array = np.zeros((6800, 9000))
for i, lat in enumerate(locations_layer.lat.values):
    for j, lon in enumerate(locations_layer.lon.values):
        distance_array[i, j] = calculate_nearest(lat, lon)

# Create a new xarray DataArray with the computed distances
distance_da = xr.DataArray(distance_array, coords=[locations_layer.lat, locations_layer.lon], dims=['lat', 'lon'])
new_dataset = xr.Dataset({'distance': distance_da})
new_dataset.attrs = locations_layer.attrs  # Copy metadata if necessary

# Set CRS attribute for distance data
new_dataset.distance.attrs['crs'] = 'EPSG:4326'

# Save the calculated distance data to a NetCDF file
output_nc_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/5.PowerTransmission/Distance_to_transmission_line_nc_Aus_005.nc'
new_dataset.to_netcdf(output_nc_path)

print("NetCDF with distance data saved successfully.")

# Step 4: Plotting the map with transmission lines and Australia outline
aus_shapefile_path = r'/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/AUS_2021_AUST_SHP_GDA2020/AUS_2021_AUST_GDA2020.shp'
aus_gdf = gpd.read_file(aus_shapefile_path)

# Calculate the total length of the transmission lines in kilometers
total_length_km = power_lines['length_m'].sum() / 1000

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
aus_gdf.plot(ax=ax, color='lightgray')  # Australia map in light gray as background
power_lines.plot(ax=ax, linewidth=1, color='blue')  # Transmission lines in blue

# Add text for total length
ax.text(0.05, 0.95, f'Total Length: {total_length_km:.2f} km', transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left',
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

ax.set_title('Map of Electricity Transmission Lines in Australia')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Removing the x and y axis ticks for a cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Save the figure
output_path = r'/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/5.PowerTransmission/1_Distance_to_transmission_line_nc_Aus_005.png'
#plt.savefig(output_path, format='png', bbox_inches='tight')



print("Map saved successfully.")
