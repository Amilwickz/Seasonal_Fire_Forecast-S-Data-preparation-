import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin

# Load the elevation data from netCDF
elevation_nc_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/1.Elevation/Elevation_nc_005_Aus.nc'
ds = xr.open_dataset(elevation_nc_path)
elevation_data = ds['Elevation'].values  # Ensure to extract the numpy array

# Calculate the gradients
grad_y, grad_x = np.gradient(elevation_data)

# Calculate slope in degrees
slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * (180 / np.pi)

# Round the slope data to 2 decimal places
slope = np.round(slope, 2)

# Plotting the slope data
plt.figure(figsize=(10, 6))
plt.title('Slope Map Australia')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.imshow(slope, cmap='terrain', extent=[ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()])
plt.colorbar(label='Slope (degrees)')
#plt.savefig('/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/2.Slope/1.Slope_nc_Aus_005_plot.png', format='png', dpi=300)


# Save the slope as a new netCDF file
slope_nc_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/2.Slope/Slope_nc_Aus_005.nc'
slope_ds = xr.Dataset(
    {
        "slope": (["lat", "lon"], slope),
    },
    coords={
        "lat": ds.lat,
        "lon": ds.lon
    }
)
slope_ds.attrs['description'] = 'Slope calculated from elevation data'
slope_ds.attrs['units'] = 'degrees'
slope_ds.to_netcdf(slope_nc_path)

print(f"Slope calculation completed and saved as netCDF to: {slope_nc_path}")
