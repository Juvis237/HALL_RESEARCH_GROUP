import sys
sys.path.append('/home/u31/lia00/research/lia/eht-imaging')
sys.path.append('/home/u31/lia00/research/lia/master_code')
from my_func import *
import numpy as np

## @package my_pca
# my_pca contains functions that are used to calculate principal component 
# analysis for a set of images. 
# These are the main functions that were used in the paper "PCA APPROACH TO 
# CHARACTERIZING VARIABILITY IN BLACK HOLE IMAGES".

def cov(X):
    
    """!@brief Calculates the covariance matrix.
    
    Takes the dot product \f$ (X^T \cdot X)/n \f$ to find the normalized 
    covariance matrix of X. In the notation of the eigenfaces paper if this takes 
    in \f$A \f$ with dimension \f$N^2 \times m\f$ then it will normalize by 
    \f$N^2\f$ and will return \f$A^TA\f$.
    
    Note: numpy's `cov` uses n-1 as normalization, here we use n.

    @param X should probably be A from the paper and have dimension 
    \f$N^2 \times m\f$ where \f$N\f$ is the number of pixels 
    in each side of the image, and \f$ m\f$ is the number of images in the set. 
    X should be a 2D array where each column is an image that has been flattened 
    into a 1D vector of length \f$ N^2\f$. 
       
    @returns a 2D \f$m \times m\f$ numpy array of the normalized covariance 
    matrix of X.

    """
    return np.dot(X.T, X) / X.shape[0]
    
    
def diy_PCA(cube, save_amps=False, sub_mean_flux=False, sub_mean_image=False):
    """!@brief Calculates the PCA decomposition of a cube.
    
    Assumes that the first dimension 
    of the cube is time and the second and third dimensions are for the x and y 
    dimensions of the image. Will return normalized eigenvalues and eigenvectors.
    
    Specifically, this function takes in a cube which contains a set of images, 
    flattens images into vectors to make \f$A\f$, uses the cov()
    function which is also a part of this module to find \f$A^TA\f$. Takes the 
    eigenvectors and eigenvalues of \f$A^TA\f$ using the built in numpy 
    function np.linalg.eigh(). Sorts the eigenvalues and vectors in decreasing 
    order. Normalizes the eigenvalues so that they add up to unity. Calculates 
    \f$Av_i \f$, see the paper for more info. Reshapes the result so that the 
    images are 2D and not vectors.
    
    @param cube 3D numpy array where the first dimension corresponds to time and 
    the other two dimensions are the x and y dimensions of the images. 
    
    @param save_amps optional keyword, default set to False. If True will also 
    return a third array which contains the amplitudes, this array will be 
    $m \times m$. The resulting array is structured as follows, amps[:,0] gives 
    all of the amplitudes for the 0-th PCA component, and amps[0,:] gives all 
    of the amplitudes for all PCA comps needed to reconstruct the 0th image 
    
    @param sub_mean_flux optional keyword, default set to False. If set to True 
    the mean flux of each image will be subtracted from it prior  to 
    finding the PCA decomposition of the set of images. 
    
    @param sub_mean_image optional keyword, default set to False. If set to True 
    the mean image will be subtracted from all images prior to 
    finding the PCA decomposition of the set of images. 
    
    @returns 2 numpy arrays, the first array contains the \f$ m \f$ normalized 
    eigen values, the second array contains the \f$ m\f$ eigenvectors in a cube 
    the same shape as the original data cube. 
    """
    
    #reshape cube so that the images are now vectors
    ss        = np.shape(cube)
    cube_vect = cube.reshape((ss[0], ss[1]*ss[2])).T 
    # this should be [N^2,m], cube_vect is A in the Eigen faces paper
    
    # if we want to subtract the mean flux of each image first
    if sub_mean_flux == True: cube_vect -= np.mean(cube_vect, axis= 0)
        
    # if we want to subtract the mean image from each image
    if sub_mean_image == True:
        cube_vect -=np.mean(cube_vect, axis=1).repeat(ss[1]).reshape(ss[0],ss[1])
    
    # covariance matrix, this should be [m,m], normalized by 1/N^2
    cov_mat = cov(cube_vect)
    
    # eigen values and vectors of the covariance matrix
    # use numpy built in func, assumes cov_mat is symmetric
    E_val, E_vect = np.linalg.eigh(cov_mat)
    # eigen values and eigen vectors of the 
    # A.T A matrix in the eigen face paper, E_vect dim[m,m]
    
    # sort the eigenvalues/vectors in decreasing order
    key = np.argsort(E_val)[::-1]
    E_val, E_vect = E_val[key], E_vect[:, key]
    
    # normalize the E_val so they add up to 1
    E_val_norm = E_val/np.sum(E_val)
    
    # get Av_i from eigenfaces, this E_vect has dim [m, N^2]
    E_vect_N2 = np.dot(cube_vect, E_vect).T
    
    #reshape so that the "vectors" are images again, dim[m,N,N]
    E_vect_N2 = E_vect_N2.reshape(ss[0], ss[1],ss[2])
    
    if save_amps == True: return(E_val_norm, E_vect_N2, E_vect)
    else: return(E_val_norm, E_vect_N2)

    
def get_amps(comps, image):
    """!@brief Calculates amplitudes needed to recreate a given image using PCA 
    components.
    
    Takes in the PCA components and the image that you would like to reconstruct 
    with these components. Will return an array containing the amplitudes of 
    each component so that you can recreate an approximation of the image 
    as a linear combination of the principal components which were provided. 
    
    Specifically, this function projects the image onto each of the components, 
    by taking the dot product of the transpose of each component with the image, 
    and then normalizing by the dot product of the transposed image with itself. 
    
    @param comps 3D numpy array containing the principal components. First 
    dimension has to be the different components, and the other two dimensions 
    the x and y of each of the components.
    
    @param image a 2D numpy array containing the image you would like to project 
    onto the components. Note that the image must have the same dimensions as the 
    components.
     
    @returns a 1D numpy array of length equal to the number of components given.
    """
    
    # get the shape of the set of pricipal components
    s = np.shape(comps) 
    a_array = np.zeros(s[0])
    for i in range(0,s[0]):
        # get the ith component
        comp = comps[i,:,:]
        
        # flatten the ith component
        comp = comp.reshape(np.shape(comp)[0]*np.shape(comp)[1]) 
        
        # flatten the image
        image_flat = image.reshape(np.shape(image)[0]*np.shape(image)[1])
        
        # project the image onto the eigen image
        a_array[i] = np.dot(comp.T, image_flat)/(np.dot(comp.T, comp))
    return(a_array)
    
    
def projection(image, comps, num):
    x = np.shape(image)[0]
    amps       = get_amps(comps[0:num,:,:],image)
    amplitudes = amps.repeat(x*x).reshape(((num,x,x)))
    recon      = np.sum(comps[:num,:,:]*amplitudes, axis=0)
    return(recon)
    
    
    
    
    
    
    