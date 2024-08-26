#!/usr/bin/env python3
import numpy as np

## @package circle_stats
# circle_stats contains functions used to calculate statistics for angles 
# taking into account the fact that angles are circular quantities.


def circlemean(Phase, unit='rad'):
    """!@brief Calculates the mean of a distribution of angles. 
    
    This can handle either a single image or 
    an entire cube, as long as the cube is structured
    such that the first axis is time. If the cube is 3D 
    it will take the mean of the 0th  axis and return a 2D array, if it is 
    not 3D it will take the mean of all axes and return one number.
    
    It uses the following equation to find the mean:
        
    \f[
    \bar{\theta}= \tan^{-1}\left( \bar{S}/\bar{C}\right),
    \quad \mathrm{if } \quad \bar{C}\geq 0
    \quad \mathrm{and} \quad
    \tan^{-1}\left( \bar{S}/\bar{C}\right)+\pi,              
    \quad\mathrm{if }\quad \bar{C}< 0,
    \f]   
    
    where 
    
    \f[
    \bar{S} = \frac{1}{n}\sum^n_{j=1} \sin (\theta_j)
    \quad \mathrm{and} \quad
    \bar{C} = \frac{1}{n}\sum^n_{j=1} \cos (\theta_j).
    \f]  
    

    @param Phase either 1D or 3D numpy array of either a phase distribution 
    or movie of phases as long as a movie has time for the first axis.
    
    @param unit keyword set to 'rad' by default, if set to 'deg' 
    will convert to rad before proceeding.
      
    @returns either a float or a 2D numpy array of floats, corresponding 
    to the mean of the angles given by the above equation.
    """
    if unit == 'deg': Phase = Phase/360.*2.*np.pi
    ss = np.shape(Phase)
    if np.shape(ss)[0] != 3:
        # print('circlemean() took the mean of all axes, input had ', ss, 'shape')
        C_mean = np.mean(np.cos(Phase))
        S_mean = np.mean(np.sin(Phase))
        mean_phase = np.arctan(S_mean/C_mean)
        if C_mean < 0: mean_phase=mean_phase+np.pi
    else:
        C_mean = np.mean(np.cos(Phase), axis = 0)
        S_mean = np.mean(np.sin(Phase), axis = 0)
        mean_phase = np.arctan(S_mean/C_mean)
        mean_phase[np.where(C_mean < 0.0)] = mean_phase[np.where(C_mean < 0.0)]+np.pi
    return(mean_phase)
      


def circlerms(Phase,unit='rad'):
    """!@brief Calculates dispersion of a distribution of angles. 
    
    This can handle either a single image or 
    an entire cube, as long as the cube is structured
    such that the first axis is time. If the cube is 3D 
    it will take the mean of the 0th  axis and return a 2D array, if it is 
    not 3D it will take the mean of all axes and return one number.
    
    It uses the following equation to find the dispersion:
        
    \f[
    D = \frac{1}{n}\sum^n_{j=1} \{  1-\cos(\theta_j-\bar{\theta})\}
    \f]    
    
    where \f$\bar{\theta}\f$ is as defined in circlemean().    

    @param Phase either 1D or 3D numpy array of either a phase distribution or 
    movie of phases as long as a movie has time for the first axis.
    
    @param unit keyword set to 'rad' by default, if set to 'deg' 
    will convert to rad before proceeding.
      
    @returns either a float or a 2D numpy array of floats corresponding 
    to \f$D\f$ in the above equation.
    """
    
    if unit == 'deg': Phase = Phase/360.*2.*np.pi
    if np.size(Phase.shape) == 1:
        C_mean = np.mean(np.cos(Phase))
        S_mean = np.mean(np.sin(Phase))
        ave_phase = np.arctan(S_mean/C_mean)
        if C_mean < 0: ave_phase=ave_phase+np.pi
        rms_phase = np.mean((1-np.cos(Phase-ave_phase)))# I tested this and python can in fact broadcast these things together correctly
    else:
        C_mean = np.mean(np.cos(Phase), axis = 0)
        S_mean = np.mean(np.sin(Phase), axis = 0)
        ave_phase = np.arctan(S_mean/C_mean)
        ave_phase[np.where(C_mean < 0.0)] = ave_phase[np.where(C_mean < 0.0)]+np.pi
        rms_phase = np.mean((1-np.cos(Phase-ave_phase)), axis=0)
    return(rms_phase) 