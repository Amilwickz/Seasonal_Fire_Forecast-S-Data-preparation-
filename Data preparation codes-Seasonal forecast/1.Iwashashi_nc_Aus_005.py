import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xarray as xr
import time
from netCDF4 import Dataset  # For creating the final NetCDF file
from scipy.spatial import cKDTree

# --- Step 1: Resample the TIF file and convert to NetCDF ---
def resample_geotiff_to_netcdf(src_file_path, new_crs, new_transform, new_width, new_height, latitude_start, latitude_end, longitude_start, longitude_end):
    with rasterio.open(src_file_path) as src:
        # Set the target metadata based on new transformed values
        data = np.empty((src.count, new_height, new_width), dtype=src.dtypes[0])

        # Reproject and write each band using nearest neighbor resampling
        for i in range(src.count):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=data[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=new_crs,
                resampling=Resampling.nearest  # Changed to nearest neighbor interpolation
            )

        # Create an xarray Dataset
        coords = {'band': np.arange(1, src.count + 1),
                  'lat': np.linspace(latitude_end, latitude_start, new_height),
                  'lon': np.linspace(longitude_start, longitude_end, new_width)}
        dims = ('band', 'lat', 'lon')
        ds = xr.Dataset(
            {'raster': (dims, data)},
            coords=coords
        )

        # Manually add CRS as an attribute to the dataset
        ds.attrs['crs'] = new_crs

        # Metadata to include
        ds.attrs['description'] = 'Resampled raster data from GeoTIFF to NetCDF'
        ds.attrs['source'] = 'Original 500m resolution terrain units GeoTIFF'
        ds.attrs['institution'] = 'Western Sydney University'
        ds.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        ds.attrs['resolution'] = '0.005 degree grid'
        ds.attrs['geographic_extent'] = f"Longitude {longitude_start} to {longitude_end}, Latitude {latitude_start} to {latitude_end}"

        return ds

# --- Step 2: Combine the Landform Data and Match to Location Points ---
def match_landform_to_location(location_ds, landform_ds):
    # Generate 2D arrays for latitude and longitude coordinates
    lat_landform, lon_landform = np.meshgrid(landform_ds.lat.values, landform_ds.lon.values, indexing='ij')
    landform_flat = landform_ds.raster.values.flatten()

    # Creating coordinate pairs from meshed latitude and longitude arrays
    coords_landform = np.column_stack((lat_landform.ravel(), lon_landform.ravel()))
    tree = cKDTree(coords_landform)

    # Assuming location_ds also requires meshing of lat and lon
    lat_loc, lon_loc = np.meshgrid(location_ds.lat.values, location_ds.lon.values, indexing='ij')
    coords_loc = np.column_stack((lat_loc.ravel(), lon_loc.ravel()))

    # Find nearest neighbors
    _, indices = tree.query(coords_loc, k=1)  # k=1 for the nearest neighbor

    # Get landform values corresponding to the nearest indices
    nearest_landforms = landform_flat[indices]

    # Reshape nearest_landforms to match the lat/lon grid dimensions of location_ds
    nearest_landforms_reshaped = nearest_landforms.reshape(lat_loc.shape)

    # Create a new dataset
    new_ds = xr.Dataset(
        {
            'Landform': (('lat', 'lon'), nearest_landforms_reshaped)
        },
        coords={
            'lat': location_ds.lat,
            'lon': location_ds.lon
        }
    )

    # Define global attributes
    new_ds.attrs = {
        'title': 'Combined Landform Data',
        'institution': 'WSU',
        'source': 'Synthetic generation from location and landform data',
        'history': 'Created using Python with xarray and scipy.spatial.cKDTree',
        'comment': 'Iwashi Landform matched to nearest location points',
        'CRS': 'EPSG:4326 - WGS84'
    }

    return new_ds

def main():
    # --- Step 1: Resample the GeoTIFF to NetCDF ---
    src_file_path = r'/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/9.LandForm/0.Raw_data/iwashashi2_AU_500m_with_legend.tif'
    new_crs = 'EPSG:4326'  # WGS 84
    longitude_start, longitude_end = 110.0, 155.0
    latitude_start, latitude_end = -44.0, -10.0
    resolution = 0.005

    # Calculate new transform and dimensions
    new_transform, new_width, new_height = calculate_default_transform(
        src_crs=new_crs, dst_crs=new_crs,
        width=int((longitude_end - longitude_start) / resolution),
        height=int((latitude_end - latitude_start) / resolution),
        left=longitude_start, right=longitude_end,
        top=latitude_end, bottom=latitude_start
    )

    # Resample and get xarray dataset
    landform_ds = resample_geotiff_to_netcdf(src_file_path, new_crs, new_transform, new_width, new_height, latitude_start, latitude_end, longitude_start, longitude_end)

    # --- Step 2: Match the Landform Data with Location Data ---
    location_ds = xr.open_dataset('/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/0.Lat_Lon_layer/locations_layer.nc')

    # Match landform data to location points
    final_ds = match_landform_to_location(location_ds, landform_ds)

    # --- Step 3: Save the Final Dataset ---
    final_output_path = '/scratch/ey42/aw1142/Seasonal_forecast_pipeline_data/9.LandForm/Iwashashi_nc_Aus_005.nc'
    final_ds.to_netcdf(final_output_path)

    print(f"Combined Landform data saved to: {final_output_path}")

if __name__ == '__main__':
    main()
