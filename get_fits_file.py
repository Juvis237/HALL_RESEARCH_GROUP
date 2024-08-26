from tqdm import tqdm
import os
from astropy.io import fits
import numpy as np


prefix = "planet23"

def save_hdf5(file_name, name, data):
    """!@brief Saves a hdf5 file.
    
    Very simple, just creates a hdf5 file.
    
    @param file_name the name of the file which will be created, should be a string.
    
    @param name the name given to the data in the file, the "key".
    
    @param data the actual data, probably a numpy array.
       
    @returns 0
    """
    import h5py
    h5f = h5py.File(file_name, 'w')
    h5f.create_dataset(name, data=data)
    h5f.close() 
    return(0)

even_numbers = [f'{num:05}' for num in range(0, 1001, 2)] # I did this cause the valid snapshots which produced images were all even numbered i.e they were of the format snapshot101_X where X is the 5 digit format of even numbers
valid_image_data = []

for i in tqdm(even_numbers, desc="Getting Array ready"):
    fits_image_filename = f"job_{i}/RT.fits.gz"
    if os.path.exists(fits_image_filename):

        try:
            image_data = fits.getdata(fits_image_filename, ext=0)
            valid_image_data.append(image_data[0, 0, 0, :, :]) #getting just the top view of the image

        except Exception as e:
            print(f"Error reading FITS file {fits_image_filename}: {e}")
        # Handle the error as needed
        continue  # Continue to the next iteration
        
        
    else:
        print(f"The directory {fits_image_filename} does not exist.")

save_hdf5(f"{prefix}_images.h5", 'images', np.stack(valid_image_data, axis=0))