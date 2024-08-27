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