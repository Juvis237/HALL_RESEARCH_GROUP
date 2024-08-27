#!/usr/bin/env python3
import numpy as np

## @package my_func
# my_func contains the main functions which are used by the rest of the modules.
# Specifically, it contains functions to create VLBI observables for the Event 
# Horizon Telescope from GRMHD simulations.


## @mainpage Documentation for GRMHD_VLBI_utils
#
## @section intro_sec Introduction
#
# These packages contain the main fuctions that are used to calculate VLBI 
# observables from GRMHD simulations and to plot these observables. These are 
# the functions which are used by the other modules. These are the functions 
# which were used to create the figures in Medeiros et al. 2016a, b.
#
## @section my_func
#
# my_func contains the main functions which are used by the rest of the modules.
# Specifically, it contains functions used to create VLBI observables for the Event 
# Horizon Telescope from GRMHD simulations.
#
#
## @section my_plot
#
# my_plot contains the main plotting functions used to plot VLBI observables 
# from GRMHD simulations. The plotting functions were made to be used together, 
# so we can create single figures and figures with multiple subplots. 
#
#
## @section my_pca
#
# my_pca contains the main functions needed to find the PCA decomposition of a 
# set of images.
#
## @section circle_stats
#
# circle_stats contains some functions which are useful for calculating statists 
# on a set of angles or other looping quantities.
#
## @section scatter_utils
#
# scatter_utils contains some useful functions which are wrappers for the 
# scattering functions in ehtim.
#
#
## @section PS_utils
#
# PS_utils contains functions that are useful for calculating power spectra in 
# time, and performing fits to power spectra. 
#



def myfft(image, m, centered = True, pad=8):    
    """!@brief Takes the fft of an image. 
    
    Takes the 2D fourier transform of an image using np.fft.fft2(), 
    additionally it corrects for offset luminosity peaks, 
    pads the array by pad for increased resolution in the resulting fft, 
    and uses fftshift() appropriately.     
    
    @param image 2D numpy array of the image.
    
    @param m an integer between 0 and 4 corresponding to the model in use, 
    this is used to find the average luminosity peak which has already been 
    calculated for the 5 models I use.
    
    @param centered optional keyword, default set to True. If true will recenter 
    to the average luminosity peak for this model before taking fft, if False, 
    will not recenter.
    
    @param pad optional keyword, default set to 8. This is the factor by which 
    to pad the array before taking the fft.
    
    @returns 2D numpy array of complex numbers, these are the complex Fourier 
    components of the image.
    """
    
    x0,y0=0,0
    if centered == True:
        x0_arr = np.array([-24,-20,  1,-19, -9, 0])
        y0_arr = np.array([  8, 11, 18, 12, 19, 0])
        x0     = x0_arr[m]
        y0     = y0_arr[m]
    
    s = np.shape(image)[0]
    if pad != False: image = padding2D(image, x0=x0, y0=y0, pad=pad)
    image   = np.fft.fftshift(image)
    ft      = np.fft.fftshift(np.fft.fft2(image))
    if pad != False: ft = ft[int(s*(pad-1)/2): int(s*(pad+1)/2), int(s*(pad-1)/2): int(s*(pad+1)/2)]
    return(ft)
    
    
def myfft_centered(image, pad=8): 
    """!@brief Like my_fft, but uses lum_peak() to center each image before 
    taking fft.
    
    Takes the 2D fourier transform of an image using np.fft.fft2(), 
    additionally it corrects for offset luminosity peaks, 
    pads the array by pad for incresed resolution in the resulting fft, 
    and uses fftshift() before and after shifting. 
    
    This is different from myfft because it calculates the 
    luminosity peaks for each snapshot and recenters it before taking the fft.

    @param image 2D numpy array of the image.
    
    @param pad optional keyword, default set to 8. This is the factor by which 
    to pad the array before taking the fft.
    
    @returns 2D numpy array of complex numbers, these are the complex Fourier 
    components of the image.
    """
    x0, y0  = lum_peak(image, unit_pix=True)
    s       = np.shape(image)[0]
    if pad != False: image = padding2D(image, x0=x0, y0=y0, pad=pad)
    image   = np.fft.fftshift(image)
    ft      = np.fft.fftshift(np.fft.fft2(image))
    if pad != False: ft = ft[int(s*(pad-1)/2): int(s*(pad+1)/2), int(s*(pad-1)/2): int(s*(pad+1)/2)]
    return(ft)
    
    
def myifft(VA,VP,m,centered=True, pad=8):
    """!@brief Takes the inverse fft of an image. 
    
    Takes the inverse of the 2D fourier transform of an image using np.fft.ifft2(), 
    additionally it pads the array by pad for increased resolution in the resulting fft, 
    and uses ifftshift() appropriately. This should be the opposite of myfft().
    
    As an example:
    
    @code
    FT        = myfft2(sim[number,:,:],m, pad=False)
    inverted  = myifft2(np.abs(FT), np.angle(FT),m, pad=False)
    @endcode    
    will return the original array. Note that you need to set pad=False for both 
    of these otherwise you will loose part of your Fourier components when you 
    zoom into the center of the padded array. Not setting pad=False will result 
    in a blurry image since you have effectively filtered small scqle structure 
    by removing higher frequencies.              
    
    @param VA 2D numpy array of the visibility amplitude of the image.
    
    @param VP 2D numpy array of the visibility phase of the image.
    
    @param m an integer between 0 and 4 corresponding to the model in use, 
    this is used to find the average luminosity peak which has already been 
    calculated for the 5 models I use.
    
    @param centered optional keyword, defaults to True, if true will recenter 
    to the average luminosity peak for this model before taking fft, if False, 
    will not recenter.

    @param pad optional keyword, default set to 8. This is the factor by which 
    to pad the array before taking the fft.
    
    @returns 2D numpy array of real numbers, that correspond to the inverse 
    fft of the image.
    """

    s = np.shape(VA)[0]
    x0,y0=0,0
    if centered == True:
        x0_arr = np.array([-24,-20,  1,-19, -9,0])
        y0_arr = np.array([  8, 11, 18, 12, 19,0])
        x0     = x0_arr[m]
        y0     = y0_arr[m]
    image        = VA*np.exp(1j*VP)
    if pad != False: image = padding2D(image, x0=x0, y0=y0, pad=pad, type_complex=True)
    image      = np.fft.ifftshift(image)
    ft           = np.fft.fftshift(np.fft.ifft2(image))
    if pad != False:ft = ft[int(s*(pad-1)/2-y0): int(s*(pad+1)/2-y0), int(s*(pad-1)/2-x0): int(s*(pad+1)/2-x0)]
    return(ft)
    
        
def lum_peak(image, unit_pix= False, x=512, M=64): 
    """!@brief Finds the luminosity peak of an image and returns the 
    location of the peak in pixels.
    
    This calculation is similar to the center of mass calculation, 
    
    \f[
    x_{peak}=\frac{\sum _{i=1}^{n}\bar{x}_i d}{\sum _{i=1}^{n}\bar{x}_i}, 
    \quad \mathrm{and}\quad  
    y_{peak}=\frac{\sum _{j=1}^{n}\bar{x}_j d}{\sum _{j=1}^{n}\bar{x}_j}
    \f]
    
    where 
    
    \f[
    \bar{x}_i = \frac{\sum _{j=1}^{n}x_{ij}}{n}.
    \f]

    @param image 2D numpy array containing the image.
    
    @param unit_pix optional keyword, default set to False. If False the 
    returned values will be in units of \f$GM/c^2 \f$. If True the returned 
    values will be in units of pixels.
    
    @param x int, optional keyword, default set to 512, size of the x axis of 
    the array.

    @param M int, optional keyword, default set to 64, size the array in units 
    of \f$ GM/c^2 \f$.
   
    
    @returns 2 integers, (x,y) that correspond to the distance of the 
    luminosity peak from the center of the image in units of \f$GM/c^2 \f$, or 
    in units of pixels according to the keyword unit_pix.
    """
    pixel=M/x
    s = np.shape(image)[0]
    xcollapse = np.mean(image, axis=0)# an horizontal row of the average value for each x value
    ycollapse = np.mean(image, axis=1)# a vertical column of the average value for each y value
    distance  = np.arange(s)-(s-1)/2# the distance from the center point in either the x or y direction
    if unit_pix == False:
        x0        = np.sum(xcollapse*distance)/(np.sum(xcollapse))*pixel
        y0        = np.sum(ycollapse*distance)/(np.sum(ycollapse))*pixel
    else:
        x0        = np.sum(xcollapse*distance)/(np.sum(xcollapse))
        y0        = np.sum(ycollapse*distance)/(np.sum(ycollapse))
    return(x0,y0)
    
    
def gauss_array(x, y, x0, y0, sigx, sigy,theta=0):
    """!@brief Creates an array with a 2D gaussian. 
    
    This creates a gaussian, given an array size, the center 
    of the gaussian, and the sigmas. This is useful when I want to 
    test things on an image of a gaussian, or add a gaussian to an image.
    
    It uses the following equation:
        
    \f[
    G = \exp{-\left(\frac{x-x_0}{\sigma_x}+\frac{y-y_0}{\sigma_y}\right)}.
    \f]    
        
    @param x size of x-axis of the return array.
    
    @param y size of y-axis of the return array.
    
    @param x0 x location of the center of the gaussian, for a centered gaussian, 
    x0=x/2.
    
    @param y0 y location of the center of the gaussian, for a centered gaussian, 
    y0=y/2.
    
    @param sigx sigma (width of gaussian) in x direction.
    
    @param sigy sigma (width of gaussian) in y direction.
    
    @param theta optional parameter, default set to 0, angle to rotate the 
    gaussian, given in degrees.
    
    
    @returns a 2D numpy array of floats, this is an image of a gaussian.
    """
    
    Gauss    = np.zeros((x,y))
    theta = theta*(2.0*np.pi)/360.0
    for i in range(0, x):
        for j in range(0, y):
            xx,yy        = np.float(j),np.float(i)
            
            a =  ((np.cos(theta)*np.cos(theta))/(2.0*sigx*sigx))+((np.sin(theta)*np.sin(theta))/(2.0*sigy*sigy))
            b = -((np.sin(theta)*np.sin(theta))/(4.0*sigx*sigx))+((np.sin(theta)*np.sin(theta))/(4.0*sigy*sigy))
            c =  ((np.sin(theta)*np.sin(theta))/(2.0*sigx*sigx))+((np.cos(theta)*np.cos(theta))/(2.0*sigy*sigy))
            Gauss[i,j] = np.exp(-1*(a*np.float((xx-x0)*(xx-x0))+2.0*b*np.float((xx-x0)*(yy-y0))+c*np.float((yy-y0)*(yy-y0))))
    return(Gauss)
    
    
def padding1D(array, pad=8):
    """!@brief Pads a 1D array by a factor pad.
    
    Just makes a new array which is pad times original size long, and 
    puts the original array at the beggining of the new one, 
    used by power_spec().
    
    @param array 1D numpy array .
    
    @param pad optional keyword, default set to 8. This is the factor by which 
    to pad the array before taking the fft.
    
    @returns a 1D numpy array of floats, new array is pad times bigger 
    than original, and has original at the biginning.
    """
    ss = np.size(array)
    array2 = np.zeros(int(ss*pad))
    array2[0:ss] = array # pad the array by pad
    return(array2)
    
def padding2D(array, x0=0, y0=0, pad=8, type_complex=False):
    """!@brief Pads a 2D array by a factor pad.
    
    Makes a new array where each axis is pad times original size, 
    and then puts the original array in the center of this new big one. 
    This function is used by myfft() and myfft_centered().
    
    @param array 2D numpy array.
     
    @param x0 optional keyword, default set to zero, if not set to zero 
    will offset the original array such that it is centered x0 away from 
    the center of the big array.
    
    @param y0 optional keyword, default set to zero, if not set to zero 
    will offset the original array such that it is centered y0 away from 
    the center of the big array.
    
    
    @param pad optional keyword, default set to 8. This is the factor by which 
    to pad the array before taking the fft.
    
    @param type_complex optional keyword, default set to False. If True, will 
    assume the input array is complex and return a complex array.
    
    @returns a 2D numpy array of floats, new array is pad times bigger 
    than original, and has original centered x0,y,0 away from the center.
    """
    ss = np.shape(array)
    if len(ss)==2:
        s = ss[0]
        padded = np.zeros((s*pad, s*pad))
        if type_complex==True: padded = np.array(padded,dtype=complex)
        padded[int(s*(pad-1)/2-y0): int(s*(pad+1)/2-y0), int(s*(pad-1)/2-x0): int(s*(pad+1)/2-x0)] = array
    elif len(ss)==3:
        s = ss[1]
        padded = np.zeros(((ss[0], s*pad, s*pad)))
        if type_complex==True: padded = np.array(padded,dtype=complex)
        padded[:, int(s*(pad-1)/2-y0): int(s*(pad+1)/2-y0), int(s*(pad-1)/2-x0): int(s*(pad+1)/2-x0)] = array
    return(padded)
        
    
def unpad2D(array, pad=8, x0=0, y0=0):
    """!@brief Unpads an array.
    
    Can take into account a shift in the centroid of the image due to an 
    off-center luminosity peak. works for both 2D and 3D arrays. 
    
    @param array this is the input array that you would like to unpad.
    
    @param pad int, optional keyword, dafault set to 8. This is the factor by 
    which you would like to unapd your array.
    
    @param x0 optional keyword, default set to 0, this is the amount by which to 
    shift the center to image in the x direction due to for example an off 
    center luminosity peak.
    
    @param y0 optional keyword, default set to 0, this is the amount by which to 
    shift the center to image in the y direction due to for example an off 
    center luminosity peak. 
    
    @returns 2 a 2D or 3D array that contains the unpaded original array.
    """
    
    ss = np.shape(array)
    if len(ss)==2:
        s = ss[0]
        x   = int(s/pad)
        return(array[int(x*(pad-1)/2-y0): int(x*(pad+1)/2-y0), int(x*(pad-1)/2-x0): int(x*(pad+1)/2-x0)])
    elif len(ss)==3:
        s = ss[1]
        x   = int(s/pad)
        return(array[:,int(x*(pad-1)/2-y0): int(x*(pad+1)/2-y0), int(x*(pad-1)/2-x0): int(x*(pad+1)/2-x0)])

    
def rebin(array, freq, box):
    """!@brief Rebins 1D array of power spectra and its frequencies.
    
    Rebinning is kind of like a smoothing function, but new array is
    smaller than original, takes the average of the values within box
    and populates new, smaller array with these averages.
    
    @param array this is the original 1D numpy array,
    this probably has a power spectrum in it.
    
    @param freq this is a 1D numpy array with the frequencies 
    which correspond to the power spectra in array.
    
    @param box this is an integer, it is the factor by which 
    we will bin, the output arrays will be smaller than 
    the original buy this factor.
    
    @returns 2 1D numpy arrays, the first is the binned version of array, 
    the second is the binned version of freq.
    """
    
    new   = np.zeros(int(np.size(array)/box/2))
    F_new = np.zeros(int(np.size(array)/box/2))
    for k in range(0,int(np.size(array)/box/2)): # loop over the new bins                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \                                                                                                                                                                                                                                                                                                                                                                                      
        new[k]  = np.mean(array[k*box:(k+1)*box])
        F_new[k]= np.mean(freq[k*box:(k+1)*box])
    new         = new/(np.sum(new)*(F_new[2]-F_new[1]))#normalize it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \                                                                                                                                                                                                                                                                                                                                                                                      
    return(new,F_new)
    
    
def gauss_distribution(mean1,sig1,mean2,sig2,N):
    """!@brief Creates 2 Gaussian distributions.
    
    I originally wrote this in Dimitrios' stats class but have used it multiple 
    times since. It will create a set of random numbers taken from a Gaussian 
    distribution.
    
    @param mean1 float, mean of the gaussian.
    
    @param sig1 float, standard deviation of the gaussian.
    
    @param mean2 float, mean of the gaussian.
    
    @param sig2 float, standard deviation of the gaussian.
    
    @param N integer, number of points in the distribution.
    
    @returns 2 1D numpy arrays of length N, each has a gaussian distribution.
    """
    import random
    y1_array = np.zeros(N)
    y2_array = np.zeros(N)
    for i in range(0,N):
        x1 = random.random()
        x2 = random.random()
        if x1 !=0:
            y1_array[i] = np.sqrt(-2.0*np.log(x1))*np.cos(2.0*np.pi*x2)
            y2_array[i] = np.sqrt(-2.0*np.log(x1))*np.sin(2.0*np.pi*x2)
    G1 = mean1+sig1*y1_array
    G2 = mean2+sig2*y2_array
    return(G1,G2)


    
def parallel(func, j=True, array=False, zip_=False, num=64, cores_per_job=1,num_cores=None):
    """!@brief Runs a function in parallel over the available cores.
    
    This uses multiprocessing, specifically pool and map. It will use all of the 
    available cores. Written to run on el gato.
    
    @param func the function that will be run in parallel, this function will 
    only receive one argument, this argument is an integer, in analogy with an 
    index in a for loop.
    
    @param j optional keyword, default set to True, if True this function will 
    use sys.argv to read in j from the argument which was given when you called 
    the file, if not True should be equal to a number, and this function will 
    run the equivalent to a for loop from index
    \f$ j \times num\f$ to \f$ (j+1)\times num \f$.
    
    @param array optional keyword, default set to False, if the function you are
    running in parallel only returns one 2D array for each "loop" and you 
    want to get a 3D array from the list of 2D arrays you should set array=True.
    The result will then be a 3D array where the first dimension corresponds to 
    the different indices in the loop, so if you are looping through time, the 
    first index will be time. 
    
    @param zip_ optional keyword, default set to False, if the function you are 
    running in parallel returns multiple 2D arrays for each "loop" then you want
    to set zip_=True, this will return a 4D array which can easily be unpacked 
    into the different 3D arrays for each thing which the function returns.
    
    @param num optional keyword, default set to 64, this is how many iterations 
    you want to run in parallel.
    
    @param cores_per_job optional keyword, default set to 1. If one each core 
    will get assigned a job and they will all be working. However, if you need 
    more memory you can assign 2 (or more) to this variable such that half 
    (or more) of the cores are idle. The cores that are working will each be 
    able to use more memory.
       
    @returns output of the function which was run in parallel, if array=True 
    this will be an array, if zip_=True will return a list of arrays which can 
    be unpacked. 
    """
    import multiprocessing
    import sys
    if j == True: j= sys.argv[1] 
    j         = int(j)
    inputs    = range(int(j*num), int((j+1)*num))
    if num_cores==None:
        num_cores = int((multiprocessing.cpu_count())/(cores_per_job))
    print('multi-processing found ',num_cores,' cores')
    pool      = multiprocessing.Pool(num_cores)
    result    = pool.map(func, inputs)
    if array == True: result = np.asarray(result)
    if zip_  == True: result = zip(*result)
    return(result)
    
    
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

    
    
def rms_cube(array):
    """!@brief Takes the root mean square of a cube.
    
    Takes the square of the array, then the mean of that along the first axis, 
    so you now have the square of the average image. The it takes the square 
    root of that.
    
    @param array 3D numpy array, you will take the rms of the first dimension of 
    this array. 
    
    @returns 2D numpy array with the same dimension as the last two of the input 
    cube, contains the rms of the cube.
    """ 

    return(np.sqrt(np.mean(np.square(array), axis=0)))
    
    
def rotate_E(array):
    """!@brief Rotates an array so that the black hole now points East.
    
    Assumes that the black hole was pointing North originally. Works for both 2D 
    and 3D arrays.
    
    
    @param array 3D or 2D numpy array that you want to rotate.
    
    @returns numpy array of the same size and dimension as the original array.
    """ 
    
    ss = np.shape(array)
    s = ss[-1]
    if len(ss)==2:    
        new = np.zeros((s,s))
        for i in range(0,s):
            for j in range(0,s):
                new[j,(s-1)-i] = array[i,j]
    else:
        new = np.zeros(((ss[0],s,s)))
        print(np.shape(new))
        for i in range(0,s):
            for j in range(0,s):
                new[:,j,(s-1)-i] = array[:,i,j]
    return(new)
    


def sinc_kernel(axis, dx, sigma):
    result = np.exp(-1*(axis-dx)*(axis-dx)/(2.*sigma))*(-1)**(axis)/(axis-dx)
    return(result/np.sum(result))

def interp_sinc_image(image, pix_u, pix_v, sigma=10., number=11):
    '''
    @param image  a 2d array containing the image we want to interpolate over, image must be square.
    @param pix_u u location in pixels.
    @param pix_v v location in pixels.
    
    These are essentially the u-v coordinates of the point
    divided by the size of a pixel. pix_u and pix_u are 
    measured away from the center of the image.
    
    Note that python makes it such that to go from array to plot it's 
    image[v,u] not image[u,v].'''
    
    round_u = round(pix_u)
    round_v = round(pix_v)
    
    du = pix_u-round_u
    dv = pix_v-round_v
    # round_u and round_v will 
    # have the index of the closest pixel 
    # to the point, but going away from the center.
    # The center of the VA and VP arrays will 
    # always be at [n/2, n/2] for an n by n array.
    
    # the pixel that corresponds to round_u and round_v is 
    # image[int(n/2.+round_v),-1*int(n/2.+round_u)]
    n = np.shape(image)[0]
    subset = image[(int(n/2.+round_v-np.floor(number/2))):(int(n/2.+round_v+np.floor(number/2)+1)), (int(n/2.+round_u-np.floor(number/2))):(int(n/2.+round_u+np.floor(number/2)+1))]

    
    
    coeffs = np.zeros((number,number))
    axis   = np.arange(number)-np.floor(number/2.)
    sincedu = sinc_kernel(axis, du, sigma)
    sincedv = sinc_kernel(axis, dv, sigma)
#     print(axis)
    for u in range(0,number):
        for v in range(0,number):
            coeffs[v,u] = sincedu[u]*sincedv[v]


    result = np.sum(subset*coeffs)

    return(result)

def observe_uv(image,u, v,uvpix=0.21100961538461538):
    sim_data = np.zeros(np.shape(u)[0])
    for i in range(0,np.shape(u)[0]):
        pix_u = u[i]/uvpix
        pix_v = v[i]/uvpix
        sim_data[i] = interp_sinc_image(image, pix_u, pix_v, sigma=10., number=11)
    return(sim_data)

def butterworth_filt(x,y,radius,n):
    return(1./(np.sqrt(1.+((np.sqrt(x*x+y*y)/radius)**(2.*n)))))

def mask_butterworth_filt(radius = 100, n=2,x=512):
    mask_filt = np.zeros((x,x))
    for i in range(0,x):
        for j in range(0,x):
            ii = i-x/2
            jj = j-x/2
            mask_filt[i,j]=butterworth_filt(ii,jj,radius,n)
    return(mask_filt)

def filter_image_padded(image, mask, pad=4):
    image = padding2D(image, pad=pad)
#     mask  = padding2D(mask,  pad=pad)
    FT    = myfft(image,-1,pad=False)*mask
    image_filt = myifft(np.abs(FT),np.angle(FT),-1,pad=False)
    image_filt = unpad2D(np.abs(image_filt), pad=pad)
    return(image_filt)


def rotate_scale_uv(u, v, angle):
    '''angle is in degrees'''
    angle = angle*2.*np.pi/360.
    ss = np.shape(u)[0]
    u_new = np.zeros(ss)
    v_new = np.zeros(ss)
    u_new = u*np.cos(angle)-v*np.sin(angle)
    v_new = v*np.cos(angle)+u*np.sin(angle)
    return(u_new,v_new)

def image_asym_cross_sec_gt1(image, spin=0.0, M=64., x=512., inc=17):
    offset = spin*np.sin(inc/360.*2.*np.pi)

    # now we estimate to the closest pixel
    offset_pix       = np.round(offset/M*x)

    horizontal_cross = image[int(x/2.),:]
    left_half        = horizontal_cross[0:int(int(x/2.)+offset_pix)]
    right_half       = horizontal_cross[int(int(x/2.)+offset_pix):]
    vertical_cross   = image[:,int(int(x/2.)+offset_pix)]
    bottom_half      = vertical_cross[0:int(x/2.)]
    top_half         = vertical_cross[int(x/2.):]
    
    if np.sum(np.sum(left_half)) > np.sum(right_half):
        horizontal_ratio = np.sum(left_half)/np.sum(right_half)
    else: 
        horizontal_ratio = np.sum(right_half)/np.sum(left_half)
        
    if np.sum(bottom_half) > np.sum(top_half):
        vertical_ratio = np.sum(bottom_half)/np.sum(top_half)
    else: 
        vertical_ratio = np.sum(top_half)/np.sum(bottom_half)
        
    return(horizontal_ratio, vertical_ratio)

def image_asym_half_gt1(image, spin=0.0, M=64., x=512., inc=17):
    offset = spin*np.sin(inc/360.*2.*np.pi)

    # now we estimate to the closest pixel
    offset_pix       = np.round(offset/M*x)

    left_half        = image[:int(int(x/2.)+offset_pix), :]
    right_half       = image[int(int(x/2.)+offset_pix):, :]
    bottom_half      = image[:, :int(x/2.)]
    top_half         = image[:, int(x/2.):]
    
    if np.sum(np.sum(left_half)) > np.sum(right_half):
        horizontal_ratio = np.sum(left_half)/np.sum(right_half)
    else: 
        horizontal_ratio = np.sum(right_half)/np.sum(left_half)
        
    if np.sum(bottom_half) > np.sum(top_half):
        vertical_ratio = np.sum(bottom_half)/np.sum(top_half)
    else: 
        vertical_ratio = np.sum(top_half)/np.sum(bottom_half)
    return(horizontal_ratio, vertical_ratio)

def image_asym_cross_sec_rotate_gt1(image, angle, spin=0.0, M=64., x=512, inc=17):
    import scipy
    offset = spin*np.sin(inc/360.*2.*np.pi)

    # now we estimate to the closest pixel
    offset_pix       = np.round(offset/M*x)
    
    new_image = np.zeros((x,x))
    new_image[:, :int(x-offset_pix)] = image[:,int(offset_pix):]
    
    imager = scipy.ndimage.interpolation.rotate(image, angle, reshape=False)

    horizontal_cross = imager[int(x/2.),:]
    left_half        = horizontal_cross[0:int(int(x/2.)+offset_pix)]
    right_half       = horizontal_cross[int(int(x/2.)+offset_pix):]

    if np.sum(np.sum(left_half)) > np.sum(right_half):
        horizontal_ratio = np.sum(left_half)/np.sum(right_half)
    else: 
        horizontal_ratio = np.sum(right_half)/np.sum(left_half)
    return(horizontal_ratio)


def image_asym_cross_sec_all(image, M=64.,  spin=0.0, inc=17, x_center=None, y_center=None, zoom=False, buffer=40, plot=False, start_angle=-90, end_angle=90):
    import scipy
    asyms = np.zeros(int(end_angle - start_angle))

    x = np.shape(image)[0]

    if zoom==True:
        if x_center==None: x_center=int(s/2)
        if y_center==None: y_center=int(s/2)

        new_image = image[y_center-buffer:y_center+buffer, x_center-buffer:x_center+buffer]
        x=int(buffer*2)
    else:
        offset     = spin*np.sin(inc/360.*2.*np.pi)
        # now we estimate to the closest pixel
        offset_pix = np.round(offset/M*x)
        new_image  = np.zeros((x,x))
        new_image[:, :int(x-offset_pix)] = image[:,int(offset_pix):]

    
    if plot==True:
        plt.imshow(new_image, cmap='inferno', origin='lower')
        plt.axhline(y=(x-1)/2.)
        plt.axvline(x=(x-1)/2.)
        
    j=0
    for i in range(start_angle, end_angle):
        imager = scipy.ndimage.interpolation.rotate(new_image, i, reshape=False)

        horizontal_cross = imager[int(x/2.),:]
        left_half        = horizontal_cross[0:int(int(x/2.))]
        right_half       = horizontal_cross[int(int(x/2.)):]
        if np.sum(np.sum(left_half)) > np.sum(right_half):
            asyms[j] = np.sum(left_half)/np.sum(right_half)
        else: 
            asyms[j] = np.sum(right_half)/np.sum(left_half)
        j+=1
    return(asyms)


def image_asym_cross_sec_max(image, spin=0.0, M=64., inc=42, return_all=False, start_angle=-90, end_angle=90):
    import scipy
    offset = spin*np.sin(inc/360.*2.*np.pi)
    x = np.shape(image)[0]

    # now we estimate to the closest pixel
    offset_pix       = np.round(offset/M*x)
    
    new_image = np.zeros((x,x))
    new_image[:, :int(x-offset_pix)] = image[:,int(offset_pix):]
    
    asyms = np.zeros((2, int(end_angle-start_angle)))
    
    for i in range(start_angle, end_angle):
        imager = scipy.ndimage.interpolation.rotate(new_image, i, reshape=False)

        horizontal_cross = imager[int(x/2.),:]
        left_half        = horizontal_cross[0:int(int(x/2.)+offset_pix)]
        right_half       = horizontal_cross[int(int(x/2.)+offset_pix):]
        vertical_cross   = imager[:,int(int(x/2.)+offset_pix)]
        bottom_half      = vertical_cross[0:int(x/2.)]
        top_half         = vertical_cross[int(x/2.):]

        if np.sum(np.sum(left_half)) > np.sum(right_half):
            asyms[0,i] = np.sum(left_half)/np.sum(right_half)
        else: 
            asyms[0,i] = np.sum(right_half)/np.sum(left_half)

        if np.sum(bottom_half) > np.sum(top_half):
            asyms[1,i] = np.sum(bottom_half)/np.sum(top_half)
        else: 
            asyms[1,i] = np.sum(top_half)/np.sum(bottom_half)
    if return_all==False: return(np.max(asyms))
    else: return(asyms)

def gauss_curve(mean,sigma,num=1000.,minimum=-0.2,maximum=0.2):
    axis = np.arange(minimum,maximum, ((maximum-minimum)/num), dtype=float)
#     print(axis)
    gauss = 1./(2*np.sqrt(2.*np.pi*sigma*sigma))*np.exp(-(axis-mean)*(axis-mean)/(2.*sigma*sigma))
    return(axis,gauss)

def Gauss_filt(sigma, image, pad=4):
    # Currently sigma is in units of pixels in the Fourier domain
    ss         = np.shape(image)
    mask_gauss = gauss_array(ss[0]*pad, ss[0]*pad, (ss[0]*pad-1)/2., (ss[0]*pad-1)/2., sigma, sigma, theta=0)
    mask_gauss = mask_gauss/np.max(mask_gauss)
    image      = padding2D(image, pad=pad)
    FT         = myfft(image,-1,pad=False)*mask_gauss
    image_filt = myifft(np.abs(FT),np.angle(FT),-1,pad=False)
    image_filt = unpad2D(np.abs(image_filt), pad=pad)
    return(image_filt)


def observe_w_errors(image, errorEHT, u, v, name, data, data_string, uvpix, BW=True, flux=None, return_vis=False, random_seed=None, pad=8):
    # on March 31 I changed this so that it creates fake data with negative 
    # phase to match the EHT data per D's suggestion
    np.random.seed(seed=random_seed)
    
    if flux != None:
        image = image/np.sum(image)*flux
    if BW == True: image = filter_image_padded(image, mask_butter, pad=padBW)
    FT = (myfft(image, -1, pad=pad))
    
    VA = np.abs(FT)
    VP = -1.*np.angle(FT) # this is to make it consistent with the VLBI convention
    
    FT = VA*np.exp(VP*1j)
    
    RE = FT.real
    IM = FT.imag
    
    # Observe the simulated image with the u-v coords
    sim_RE = np.zeros(np.shape(u)[0])
    sim_IM = np.zeros(np.shape(u)[0])
    for i in range(0,np.shape(u)[0]):
        pix_u     = u[i]/uvpix
        pix_v     = v[i]/uvpix
        sim_RE[i] = interp_sinc_image(RE, pix_u, pix_v, sigma=10., number=11)
        sim_IM[i] = interp_sinc_image(IM, pix_u, pix_v, sigma=10., number=11)
        
    # create noise with properties similar to errors given
    sim_err = np.zeros((2, np.shape(u)[0]))
    for i in range(0,np.shape(u)[0]):
        # grab a couple random numbers from a Gaussian distribution centered at 0 with sigma 1
        noise1,noise2 = gauss_distribution(0.,1.,0.0,1., 1)
        # We want to use numbers from a Gaussian distribution with sigma equal to $\sqrt{2}$ times 
        # times the error we got from the EHT

        sim_err[0,i] = noise1*errorEHT[i]/np.sqrt(2)
        sim_err[1,i] = noise2*errorEHT[i]/np.sqrt(2)
        
        
    # here we normalize the simulated data so that the size of the error for the EHT data actually 
    # makes sense with the simulated data
    # VA = np.abs(myfft(image, -1, pad=pad))
    sim_data_orig = observe_uv(VA, u, v,uvpix=uvpix)
    
    if flux == None:
        max_VA = np.max(sim_data_orig)

        sim_RE = sim_RE/max_VA*.5
        sim_IM = sim_IM/max_VA*.5
        
    new_complex = (sim_RE+sim_err[0,:]) + 1j*(sim_IM+sim_err[1,:]) 
    
    with open(name,'w') as f:
        print('ctime,st1,st2,u,v,vis,error', file=f)
        for i in range(0,np.shape(u)[0]):
            print(str(round(data[i,0],7))+','+ data_string[i,1]+','+data_string[i,2]+',{0:.6f}'.format(data[i,3])+',{0:.6f}'.format(data[i,4])+','+str(round(new_complex[i],6))+','+str(round(data[i,6], 6)), file=f)
            

    if return_vis == True:
        return(new_complex)
   
  