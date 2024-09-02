
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
    
    
def open_hdf5(file_name, mode='full', i=0):
    """!@brief Opens a hdf5 file.
    
    Very simple, just opens a hdf5 file.
    
    @param file_name the name of the file which will be opened, should be a string.
    
    @param mode optional keyword, default set to 'full', the possible options are:
        'full' which loads the full array
        'one' which loads one frame, the ith frame
        'range' which loads a range of frames from i[0] to i[1].
    
    @param i optional keyword, default set to 0, see mode above.
       
    @returns 0
    """ 
    import h5py
    file    = h5py.File(file_name, 'r') 
    name    = list(file.keys())[0]
    d       = file[str(name)] 
    if mode == 'full': result = d[:] 
    if mode == 'one':  result = d[i] 
    if mode == 'range':result = d[i[0]:i[1]]
    file.close()  
    return(result)



def plot_log_image(image, title=None, logScale=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    """
    Plots an image with logarithmic color scale using Astropy and Matplotlib.

    Args:
        image: The 2D image data as a NumPy array.
        title: The title to display for the plot (optional).
    """

    # Create the plot figure
    plt.figure()

    # Use imshow with LogNorm for logarithmic color scale
    if logScale:
        norm = LogNorm(vmin=1E-22, vmax=2E-19)
        plt.imshow(image, norm=norm)
    else:
        plt.imshow(image,vmin=1E-22 , vmax= 2E-19)


    # Add title if provided
    if title:
        plt.title(title)

    # Set colorbar and label
    plt.colorbar(label='Pixel Value (Log Scale)')

    # Show the plot
    plt.show()
def save_fits(fits_filename, data_cube):
    #data_cube should be a np array
    from astropy.io import fits
    

    # Create a FITS HDU (Header/Data Unit) object with the data
    hdu = fits.PrimaryHDU(data_cube)

    # Create an HDU list and write it to a FITS file
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(fits_filename, overwrite=True)

    print(f"Data has been saved to {fits_filename}")