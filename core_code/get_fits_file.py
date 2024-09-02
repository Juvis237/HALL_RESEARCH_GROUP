

import psutil
import os
import time
from tqdm import tqdm
from astropy.io import fits
import numpy as np
import h5py

def save_hdf5_incrementally(file_name, name, data, maxshape):
    """Save data incrementally to an HDF5 file."""
    with h5py.File(file_name, 'a') as h5f:
        if name not in h5f:
            dataset = h5f.create_dataset(name, data.shape, maxshape=maxshape, chunks=True)
        else:
            dataset = h5f[name]
            dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
            dataset[-data.shape[0]:] = data

def check_memory_usage(threshold=80):
    """Check if the memory usage exceeds a threshold percentage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Convert bytes to MB
    total_memory = psutil.virtual_memory().total / (1024 ** 2)  # Convert bytes to MB
    memory_percent = (memory_usage / total_memory) * 100
    
    if memory_percent > threshold:
        print(f"Warning: Memory usage is {memory_percent:.2f}% of total memory ({memory_usage:.2f} MB).")

def extract_fits_images(even_numbers, dir_name, prefix):
    file_name = f"{prefix}_images.h5"
    for i in tqdm(even_numbers, desc="Processing FITS files"):
        fits_image_filename = f"{dir_name}/job_{i}/data_1300/RT.fits.gz"
        if os.path.exists(fits_image_filename):
            try:
                image_data = fits.getdata(fits_image_filename, ext=0, memmap=True)
                top_view_image = image_data[0, 0, 0, :, :]

                # Save incrementally
                save_hdf5_incrementally(file_name, 'images', np.expand_dims(top_view_image, axis=0), maxshape=(None,) + top_view_image.shape)

                # Check memory usage after each file is processed
                check_memory_usage(threshold=80)

            except Exception as e:
                print(f"Error reading FITS file {fits_image_filename}: {e}")
                continue  # Continue to the next iteration

        else:
            print(f"The directory {fits_image_filename} does not exist.")

# Example usage
# even_numbers = [f'{num:05}' for num in range(0, 1001, 2)]
# prefix = "planet23"
# extract_fits_images(even_numbers, "your_directory_path", prefix)
