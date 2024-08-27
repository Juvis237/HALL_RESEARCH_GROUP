import matplotlib
# matplotlib.use('Agg')
import sys
import os
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
sys.path.append('/home/u31/lia00/research/lia/master_code')
from my_func import *
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm, colors, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as PathEffects
try: 
    import ehtplot as eht
except ImportError:
    pass
import matplotlib.pyplot as plt

try:
    from os.path import expanduser
    home = expanduser("~")
    sys.path.append(home+'/Dropbox/research/ehtplot')
    from ehtplot import *
except ImportError:
    pass

## @package my_plot
# my_plot contains the main plotting functions which are used by the rest of the modules.
# Specifically, it contains functions plot VLBI observables fromm GRMHD simulations.

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def r_photon(a):
    """!@brief used by make_circle_kerr()
    """
    #calculates equation 5 and in units of M
    r_temp1 = 2.0*(1.0+np.cos((2.0/3.0)*np.arccos(np.abs(a))))
    r_temp2 = 2.0*(1.0+np.cos((2.0/3.0)*np.arccos(-np.abs(a))))
    return(r_temp1,r_temp2)

def alpha(a,r,i=60):
    """!@brief used by make_circle_kerr()
    """
    #calculates eq 6a in units of M 
    i = i/360.0*2.0*np.pi
    return(-(a*a*(r+1.0)+(r-3.0)*r*r)/(np.sin(i)*a*(r-1.0)))

def beta(a,r,i=60):
    """!@brief used by make_circle_kerr()
    """
    # calculate eq 6b in units of M 
    i = i/360.0*2.0*np.pi
    coef_in_front = 1.0/(a*(r-1.0))
    first_part    = a*a*a*a*(r-1.0)*(r-1.0)*np.cos(i)**2
    second_part   = -(a*a*(r+1.0)+(r-3.0)*r*r)**2/(np.tan(i)**2)
    third_part    = -r*r*r*((r-3.0)*(r-3.0)*r-4.0*a*a)
    return(coef_in_front*np.sqrt(first_part+second_part+third_part))           

def make_circle_kerr(ax1,a,i=60,circle_width=1,color='r',label=None,center=False,return_stuff=False,alpha_circle=1, angle=None, ls='-', flip_x=False):
    """!@brief Adds analytically calculated Kerr black hole shadow to plot.
    
    @param ax1 the name of the subplot where you want to plot your shadow.
    
    @param a black hole spin, should be float, between -1 and 1.
    
    @param i optional keyword, default set to 60.0 degrees. This is the 
    inclination of the observer relative to the black hole in degrees.
    
    @param circle_width optional keyword, default set to 1. This controls the 
    line-width of the circle that corresponds to the black hole shadow.
    
    @param color optional parameter, default set to 'r', the color of the circle. 
    """
    if a == 0.0:
        circ1 = plt.Circle((0,0), radius=np.sqrt(27), color=color, lw = circle_width, fill=False,label=label,alpha=alpha_circle)
        ax1.add_patch(circ1)
    
    else:
        rr = r_photon(a)
        N=2001
        axis = np.arange(rr[1],rr[0],((rr[0]-rr[1])/N))
        a_array = np.zeros(N+1)
        b_array = np.zeros(N+1)
        k=0
        for r in axis:
            a_array[k] = alpha(a,r,i=i)
            b_array[k] = beta(a,r,i=i)
            k+=1
        a_array = np.delete(a_array, np.where(np.isnan(b_array)==True)[0])
        b_array = np.delete(b_array, np.where(np.isnan(b_array)==True)[0])
        a_array = a_array[:-1]
        b_array = b_array[:-1]
        a_array = np.append(-a_array,-a_array[::-1])
        b_array = np.append(b_array,-b_array[::-1])
        a_array = np.append(a_array, a_array[0])
        b_array = np.append(b_array, b_array[0])
        if center == True: a_array=a_array - 2.0*a*np.sin(i/360.0*2.0*np.pi)
        if angle != None: 
            angle_r = angle/180.*np.pi
            a_new   = a_array*np.cos(angle_r)+b_array*np.sin(angle_r)
            b_new   = b_array*np.cos(angle_r)-a_array*np.sin(angle_r)
            a_array = a_new
            b_array = b_new
        if flip_x==False:
            ax1.plot(a_array, b_array, c=color, lw = circle_width,label=label,alpha=alpha_circle, ls=ls)
        else:
            ax1.plot(-a_array, b_array, c=color, lw = circle_width,label=label,alpha=alpha_circle, ls=ls)
        if return_stuff==True: return(a_array, b_array)


def plot_image(array, fig=None, ax1=None, spin=None, inc=60.0, angle=0, output=None, name=None, norm=True, scale='lin', 
font=10.56, colorbar=True, norm_num=1, lim_lin=np.array([0,1]), lim_log=False, flip_x=False, 
horz=False, M=64, x_label=True, y_label=True, colorbar_ticks='set', circle_width=1, 
zoom=True, tick_color='w', cb_tick_color='k',cmap='gnuplot2', interpolation='bilinear', mass=4.31*10**6,distance=8.3, uas=False):
    """!@brief Makes a plot of an image.
    
    This can be used for a single image or for multiple subplots, 
    below is an example of how this can be used for a single image:
    @code    
    plot_image(image_array, spin=.9, name='Model B', output= 'output_file.pdf')
    @endcode
    
    This is an example for multiple subplots:
        
    @code
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    fig.set_size_inches(12,12)
    
    plot_image(image_array1, fig=fig, ax1=ax1, spin=.9, colorbar=False)
    plot_image(image_array2, fig=fig, ax1=ax2, spin=.9, colorbar=False)
    plot_image(image_array3, fig=fig, ax1=ax3, spin=.9, colorbar=False)
    plot_image(image_array4, fig=fig, ax1=ax4, spin=.9, colorbar=False)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    Note that for multiple subplots you might want to omit color bars, 
    and only include the model name in one of the subplots.

    @param array 2D numpy array of the image to be plotted.
    
    @param fig optional keyword, default set to None. If None will create a 
    figure of size columnwidth by columnwidth with one subplot. If not None this 
    should be the name of the figure where you want to plot your image, 
    see the example code above.
    
    @param ax1 optional keyword, default set to None this keyword is closely 
    tied with fig, either they both must be None or neither. If None will create a 
    figure of size columnwidth by columnwidth with one subplot. If not None this 
    should be the name of the subplot (where applicable) where you want to plot 
    your image, see the example code above.
    
    @param spin optional keyword, default set to None. If None, will not draw a 
    circle over the image, if not None will draw the black hole shadow calculated 
    analytically for a kerr black hole with black hole spin given by this 
    parameter and the inclination of the observer given by the parameter inc.
    
    @param inc optional keyword, default set to 60.0 degrees. This is the 
    inclination of the observer relative to the black hole in degrees, used to 
    analytically calculate the shape and position of the black hole shadow see 
    also spin. 
    
    @param output optional keyword, default set to None, if not None should be 
    a string and  will save the figure to a file with file name equal to output. 
    
    @param name optional keyword, default set to None. If not None must be a 
    string and will add a text label to the plot equal to this string.
    
    @param norm optional keyword, default set to True, if True will normalize 
    image so that the maximum is 1, if false does not normalize image.
    
    @param scale optional keyword, default set to 'lin', 'log' is also 
    supported, this sets the scale of the color map.   
    
    @param font this is an optional keyword, default is set to 10.56 because this 
    is the size of the font in the emulate ApJ 2-column latex format, this sets 
    the font size for the axis labels, and numbers as well as the numbers for 
    the color bar.
    
    @param colorbar optional keyword, default set to True, if True will plot the 
    color bar, if False will do nothing, and if set to 'top' will plot colorbar 
    on top.
    
    @param norm_num optional keyword, default set to 1, this is in case 
    normalizing to 1 doesn't give the desired image, you can control a more 
    specific normalization with norm_num, for example, if norm_num=1.2, the 
    maximum value in the image will be 1.2, but the color bar will go from 0 to 
    1 (assuming scale='lin').
    
    @param lim_lin optional keyword, default set to np.array([0,1]), this is the 
    limits for the color bar if scale='lin'.
    
    @param lim_log optional keyword, default set to False, this is the limits 
    for the color bar if scale='log'.
    
    @param flip_x optional keyword, default set to False. if set to True will 
    flip the array in the left-right direction.
    
    @param horz optional keyword, default set to False, sometimes I want to plot 
    the images transposed so that we can compare different orientations, this 
    will move the red circle accordingly.

    @param M int, optional keyword, default set to 64, size the array in units 
    of \f$ GM/c^2 \f$.
    
    @param x_label optional keyword, default set to True. If True will add a 
    label to the x-axis, if False, will not add this label.
    
    @param y_label optional keyword, default set to True. If True will add a 
    label to the y-axis, if False, will not add this label.
    
    @param colorbar_ticks optional keyword, default set to 'set'. If set to 
    'set' the colorbar ticks will be at [0,0.2,0.4,0.6,0.8,1], if set to 'auto' 
    will let matplotlib set the colorbar ticks automatically.  
    
    @param circle_width optional keyword, default set to 1. This controls the 
    line-width of the circle that corresponds to the black hole shadow, see also 
    spin.
    
    @param zoom optional keyword, default set to True. If set to True will zoom 
    in to about 20 \f$ GM/c^2 \f$ on each side, if not set to True, will leave 
    the full array visible.
    
    @param tick_color optional keyword, default set to 'w' will set the color of 
    the ticks in the plot.
    
    @param cb_tick_color optional keyword, default set to 'k' will set the color 
    of the ticks for the colobar, as long as colorbar not set to 'top'.

    @param interpolation optional keyword, default set to 'bilinear'. This will 
    control the type of interpolation that is used in the plot, the options are 
    the same as those for matplotlib.

    @param mass optional keyword, default set to \f $4.31\times10^6 \f$, this is 
    only used if the parameter uas is set to True.  

    @param distance optional keyword, default set to 8.3kpc, his is 
    only used if the parameter uas is set to True.

    @param uas optional keyword, default set to False. If True will plot the figure
    in units of uas, and will use the keywords mass and distance to calculate this. 
    
    @returns ax1 if ax1 not given, or the image object if ax1 is given.
    """

    make_fig = False
    if (fig == None) and (ax1 == None):
        make_fig =True
        columnwidth = 3.39441
        fig,(ax1) = plt.subplots(1,1)
        fig.set_size_inches(columnwidth,columnwidth)
    
    x       = np.shape(array)[0]
    r0      = x*np.sqrt(27)/M # this is the radius of the black hole shadow
    r0M     = r0*M/x # this gives the BH shadow in units of GM/c**2
    if norm == True: array = array/(np.max(array))*norm_num

    ext = M/2.0
    if uas == True:
        pix_uas = np.arctan(M*mass/(x*distance)*(4.79*10**(-17)))# this is the size of a pixel in uas on the sky, here in rad 
# the factor in parenthesis above deals with the units of the equation, (G*M_solar)/(c^2* kpc)
        pix_uas = pix_uas *2.063*10**(11)# here convert to uas
        ext = (x-1)/2.0*pix_uas
    
    # plt.set_cmap(cmap)
    if scale == 'lin':
        if flip_x == True: 
            # array = np.fliplr(array)
            im1   = ax1.imshow(array, cmap=cmap,extent=[ext,-ext,-ext,ext],vmin=lim_lin[0], vmax=lim_lin[1], origin='lower', interpolation=interpolation)
        else:
            im1   = ax1.imshow(array, cmap=cmap, extent=[-ext,ext,-ext,ext],vmin=lim_lin[0], vmax=lim_lin[1], origin='lower', interpolation=interpolation)
        if colorbar==True:
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("right", size="7%", pad=0.05)
            if colorbar_ticks == 'auto': cbar1    = plt.colorbar(im1, cax=cax1)
            else: cbar1    = plt.colorbar(im1, cax=cax1, ticks=[0,0.2,0.4,0.6,0.8,1])
            cbar1.ax.tick_params(labelsize=font, color=cb_tick_color,width=1, direction='in')
        elif colorbar== 'top':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("top", size="7%", pad=0.05)
            if colorbar_ticks == 'auto': cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            else: cbar1    = plt.colorbar(im1, cax=cax1,orientation="horizontal", ticks=[0,0.2,0.4,0.6,0.8])
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('top')        
    elif scale == 'log':
        if type(lim_log) ==bool: 
            if flip_x == True: 
                # array = np.fliplr(array)
                im1=ax1.imshow(array, cmap=cmap, extent=[ext,-ext,-ext,ext], norm=LogNorm()                                , origin='lower', interpolation=interpolation)
            else:
                im1=ax1.imshow(array, cmap=cmap, extent=[-ext,ext,-ext,ext], norm=LogNorm()                                , origin='lower', interpolation=interpolation)
        else: 
            if flip_x == True: 
                # array = np.fliplr(array)                   
                im1=ax1.imshow(array, cmap=cmap, extent=[ext,-ext,-ext,ext], norm=LogNorm(vmin=lim_log[0], vmax=lim_log[1]), origin='lower', interpolation=interpolation)
            else:
                im1=ax1.imshow(array, cmap=cmap, extent=[-ext,ext,-ext,ext], norm=LogNorm(vmin=lim_log[0], vmax=lim_log[1]), origin='lower', interpolation=interpolation)
        if colorbar==True:
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("right", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1, cax=cax1)
            cbar1.ax.xaxis.set_ticks_position('top')
            cbar1.ax.tick_params(labelsize=font, color=cb_tick_color, direction='in')
        elif colorbar== 'top':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("top", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('top')
    ax1.tick_params(axis='both', which='major', labelsize=font, color=tick_color,width=1.5, direction='in')
    
    if uas == False:
        if flip_x == False:
            if zoom == True: # flip_x = False, zoom=True
                ax1.set_xlim([-r0M*2, r0M*2])
                ax1.set_ylim([-r0M*2, r0M*2])
                ax1.set_xticks([-10,-5,0,5,10])
                ax1.set_yticks([-10,-5,0,5,10])
                if name != None: 
                    ax1.text(-9,-9, name, fontsize=font, color='w') #makes the text label
            else:# flip_x = False, zoom=False
                ax1.set_yticks(ax1.get_xticks())
                ax1.set_ylim(ax1.get_xlim())
                if name !=None: 
                    ax1.text(-0.47*M,-0.47*M, name, fontsize=font, color='w') #makes the text label
        elif zoom == True: # flip_x = True, zoom=True
            ax1.set_xlim([r0M*2, -r0M*2])
            ax1.set_ylim([-r0M*2, r0M*2])
            ax1.set_xticks([10,5,0,-5,-10])
            ax1.set_yticks([-10,-5,0,5,10])
            if name != None: 
                ax1.text(9,-9, name, fontsize=font, color='w') #makes the text label
        elif zoom == False:# flip_x = True, zoom=False
            ax1.set_yticks(-1*ax1.get_xticks())
            temp = ax1.get_xlim()
            ax1.set_ylim(-1*temp[0], -1*temp[1])
            if name !=None: 
                    ax1.text(0.47*M,-0.47*M, name, fontsize=font, color='w') #makes the text label
        if x_label==True: ax1.set_xlabel('X ($GMc^{-2}$)',fontsize=font)
        if y_label==True: ax1.set_ylabel('Y ($GMc^{-2}$)',fontsize=font)
    else: 
        if flip_x == False:
            if zoom == True: # flip_x = False, zoom=True
                ax1.set_xlim([-ext*.32, ext*.32])
                ax1.set_ylim([-ext*.32, ext*.32])
                ax1.set_xticks([-30,-15,0,15,30])
                ax1.set_yticks([-30,-15,0,15,30])
                if name != None: 
                    ax1.text(-32,-32, name, fontsize=font, color='w') #makes the text label
            else:# flip_x = False, zoom=False
                ax1.set_yticks(ax1.get_xticks())
                ax1.set_ylim(ax1.get_xlim())
                if name !=None: 
                    ax1.text(-0.47*M,-0.47*M, name, fontsize=font, color='w') #makes the text label
        elif zoom == True: # flip_x = True, zoom=True
            ax1.set_xlim([ext*.32, -ext*.32])
            ax1.set_ylim([-ext*.32, ext*.32])
            ax1.set_xticks([30,15,0,-15,-30])
            ax1.set_yticks([-30,-15,0,15,30])
            if name != None: 
                ax1.text(32,-32, name, fontsize=font, color='w') #makes the text label
        elif zoom == False:# flip_x = True, zoom=False
            ax1.set_yticks(-1*ax1.get_xticks())
            temp = ax1.get_xlim()
            ax1.set_ylim(-1*temp[0], -1*temp[1])
            if name !=None: 
                    ax1.text(0.47*M,-0.47*M, name, fontsize=font, color='w') #makes the text label
        if x_label==True: ax1.set_xlabel(r'X ($\mu \mathrm{as}$)',fontsize=font)
        if y_label==True: ax1.set_ylabel('Y ($\mu \mathrm{as}$)',fontsize=font)


    if spin != None: 
        make_circle_kerr(ax1,spin,i=inc,circle_width=circle_width, flip_x=flip_x, angle=angle)
    if output != None: 
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    if make_fig == True:return(ax1)
    else: return(im1)
    
    
def plot_VA(array, fig=None, ax1=None, output=None, name=None, norm=True, scale='lin', 
btracks=False, font=10.56, colorbar=True, lim_lin=np.array([0,1]), lim_log=False, 
norm_num=1, pad=8, M=64, bounds='default', x_label=True, y_label=True, zoom=True, 
colorbar_ticks='set', tick_color='w', cb_tick_color='k', mass=4.31*10**6, distance=8.3,flip_x=True,cmap='gnuplot2'):  

    """!@brief Makes a plot of a visibility amplitude map.
    
    This can be used for a single image or for multiple subplots, 
    below is an example of how this can be used for a single image:
    @code     
    plot_VA(VA_array, name='Visibility Amplitude', output='output_file.pdf')
    @endcode
    
    This is an example for multiple subplots:
        
    @code
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    fig.set_size_inches(12,12)
    
    plot_VA(VA_array1, fig=fig,ax1=ax1, colorbar=False)
    plot_VA(VA_array2, fig=fig,ax1=ax2, colorbar=False)
    plot_VA(VA_array3, fig=fig,ax1=ax3, colorbar=False)
    plot_VA(VA_array4, fig=fig,ax1=ax4, colorbar=False)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    Note that for multiple subplots you might want to omit colorbars, 
    and only include the model name in one of the subplots.

    @param array 2D numpy array of the image to be plotted.
    
    @param fig the name of the figure where you want to plot your image, 
    see the example code above.
    
    @param ax1 the name of the subplot (where applicable) where you want to plot 
    your image, see the example code above.
    
    @param output optional keyword, default set to None, if not None should be 
    a string and  will save the figure to a file with file name equal to output. 
    
    @param name optional keyword, default set to None. If not None must be a 
    string and will add a text label to the plot equal to this string.
    
    @param norm optional keyword, default set to True, if 
    True will normalize image so that the maximum is 1, 
    if false does not normalize image.
    
    @param scale optional keyword, default set to 'lin', 
    'log' is also supported, this sets the scale of the color map.
    
    @param btracks optional keyword, default set to True, if True will plot the 
    baseline tracks for the EHT over the VA map.
    
    @param font this is an optional keyword, default is set to 20, this sets 
    the font size for the axis labels, and numbers as well as the numbers for 
    the color bar.
    
    @param colorbar optional keyword, default set to True, if True will plot the 
    color bar, if False will do nothing, and if set to 'top' will plot colorbar 
    on top.
    
    @param lim_lin optional keyword, default set to np.array([0,1]), this is the 
    limits for the color bar if scale='lin'.
    
    @param lim_log optional keyword, default set to False, this is the limits 
    for the color bar if scale='log'.
    
    @param norm_num optional keyword, default set to 1, this is in case 
    normalizing to 1 doesn't give the desired image, you can control a more 
    specific normalization with norm_num, for example, if norm_num=1.2, the 
    maximum value in the image will be 1.2, but the color bar will go from 0 to 
    1 (assuming scale='lin').
    
    @param pad int, optional keyword, default set to 8, factor by which I want 
    to pad my arrays before taking the fft.

    @param M int, optional keyword, default set to 64, size the the array in 
    units of \f$ GM/c^2 \f$.
    
    @param bounds optional keyword, default set to 'default'. If this is set to 
    a string will use the default bounds for the x and y axes. Otherwise, this 
    variable can be set to a number and the bounds will then be [-bounds,bounds].
    
    @param x_label optional keyword, default set to True. If True will add a 
    label to the x-axis, if False, will not add this label.
    
    @param y_label optional keyword, default set to True. If True will add a 
    label to the y-axis, if False, will not add this label.
    
    @param zoom optional keyword, default set to True. If set to True will zoom 
    in to about 20 \f$ G \lambda \f$ on each side, if not set to True, will leave 
    the full array visible, unless bounds is set.
    
    @param colorbar_ticks optional keyword, default set to 'set'. If set to 
    'set' the colorbar ticks will be at [0,0.2,0.4,0.6,0.8,1], if set to 'auto' 
    will let matplotlib set the colorbar ticks automatically. 
 
    @param tick_color optional keyword, default set to 'w' will set the color of 
    the ticks in the plot.
    
    @param cb_tick_color optional keyword, default set to 'k' will set the color 
    of the ticks for the colobar, as long as colorbar not set to 'top'.
    
    @returns ax1 if ax1 not given, or the image object if ax1 is given.
    """
    make_fig = False
    if (fig == None) and (ax1 == None):
        make_fig =True
        columnwidth = 3.39441
        fig,(ax1) = plt.subplots(1,1)
        fig.set_size_inches(columnwidth,columnwidth)
        
    x       = np.shape(array)[0]
    r0      = x*np.sqrt(27)/M # this is the radius of the black hole shadow
    
    uvpix   = distance/(M*mass*pad)*2.09*10**7 # 2.09*10**7 is kpc c**2/(solar mass G)*10**-9, equation: D c**2/(M mass G pad)
    widthL  = (1/(np.pi*r0))*x*uvpix*pad
    ext     = widthL*4
          
    if norm == True: array = array/(np.max(array))*norm_num
    
    plt.set_cmap('gnuplot2')
    
    if scale == 'lin':
        if flip_x==True:
            im1=ax1.imshow(array,extent=[x/2*uvpix,-x/2*uvpix,-x/2*uvpix,x/2*uvpix],vmin=lim_lin[0], vmax=lim_lin[1], origin='lower', interpolation='bilinear', cmap=cmap)
        else:
            im1=ax1.imshow(array,extent=[-x/2*uvpix,x/2*uvpix,-x/2*uvpix,x/2*uvpix],vmin=lim_lin[0], vmax=lim_lin[1], origin='lower', interpolation='bilinear', cmap=cmap)
        if colorbar==True:
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("right", size="7%", pad=0.05)
            if colorbar_ticks == 'auto': cbar1 = plt.colorbar(im1, cax=cax1)
            else: cbar1 = plt.colorbar(im1, cax=cax1, ticks=[0,0.2,0.4,0.6,0.8,1])
            cbar1.ax.tick_params(labelsize=font, which='both',color=cb_tick_color,width=1.5, direction='in')
        elif colorbar== 'top':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("top", size="7%", pad=0.05)
            if colorbar_ticks == 'auto': cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            else: cbar1    = plt.colorbar(im1, cax=cax1,orientation="horizontal", ticks=[0,0.2,0.4,0.6,0.8])
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('top')    
    if scale == 'log': 
        if type(lim_log) ==bool: 
            if flip_x==True:
                im1=ax1.imshow(array,extent=[x/2*uvpix,-x/2*uvpix,-x/2*uvpix,x/2*uvpix],norm=LogNorm(),origin='lower', interpolation='bilinear', cmap=cmap)
            else: 
                im1=ax1.imshow(array,extent=[-x/2*uvpix,x/2*uvpix,-x/2*uvpix,x/2*uvpix],norm=LogNorm(),origin='lower', interpolation='bilinear', cmap=cmap)
        else: 
            if flip_x==True:
                im1=ax1.imshow(array,extent=[x/2*uvpix,-x/2*uvpix,-x/2*uvpix,x/2*uvpix],norm=LogNorm(vmin=lim_log[0], vmax=lim_log[1]),origin='lower', interpolation='bilinear', cmap=cmap)
            else:
                im1=ax1.imshow(array,extent=[-x/2*uvpix,x/2*uvpix,-x/2*uvpix,x/2*uvpix],norm=LogNorm(vmin=lim_log[0], vmax=lim_log[1]),origin='lower', interpolation='bilinear', cmap=cmap)
        if colorbar==True:
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("right", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1, cax=cax1)
            cbar1.ax.tick_params(labelsize=font, which='both', color=cb_tick_color, direction='in')
        elif colorbar== 'top':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("top", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            cbar1.ax.tick_params(labelsize=font)#, color=
            cbar1.ax.xaxis.set_ticks_position('top')
    if x_label==True: ax1.set_xlabel('$u$ (G $\lambda$)',fontsize=font)
    if y_label==True: ax1.set_ylabel('$v$ (G $\lambda$)',fontsize=font)
    if btracks == True:
        path    = __file__.replace('my_plot.py', '')
        U, V    = np.load(path+'U.npy'), np.load(path+'V.npy')        
        U,V     = U*10**(-6),V*10**(-6)
        ax1.scatter( U, V, c='w', s=2, marker='o',edgecolors='none')
        ax1.scatter(-U,-V, c='w', s=2, marker='o',edgecolors='none')
    ax1.tick_params(axis='both', which='major', labelsize=font, color=tick_color, width=1.5, direction='in')
    if zoom == True:
        if type(bounds) == str:
            ax1.set_xlim([-ext,ext])
            ax1.set_ylim([-ext,ext])
            ax1.set_xticks([-5,0,5])
            ax1.set_yticks([-5,0,5])
            if name != None: ax1.text(9,-9, name, fontsize=font,color='w') #makes the text label
        else:
            ax1.set_xlim([-bounds,bounds])
            ax1.set_ylim([-bounds,bounds])
            if name != None: ax1.text(.9*bounds,-.9*bounds, name, fontsize=font,color='w') #makes the text label
    else:
        ax1.set_yticks(ax1.get_xticks())
        temp = ax1.get_xlim()
        ax1.set_ylim(temp[0], temp[1])
        if name !=None: 
            ax1.text(.9*temp[0],-.9*temp[0], name, fontsize=font, color='w') #makes the text label
    # if flip_x == True:
    #     temp = ax1.get_xlim()
    #     ax1.set_xlim(-1*temp[0], -1*temp[1])
    if output != None: 
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    if make_fig == True:return(ax1)
    else: return(im1)

    
    
def plot_VP(array, fig=None, ax1=None, output=None, name=None, btracks=False, 
font=10.56, colorbar=True, pad=8, M=64,  x_label=True, y_label=True, zoom=True,
white_width=5, interpolation='bilinear',tick_color='k', cb_tick_color='k',
mass=4.31*10**6, distance=8.3,flip_x=True):
    """!@brief Makes a plot of a visibility phase map.
    
    This can be used for a single image or for multiple subplots, 
    below is an example of how this can be used for a single image:
   @code        
    plot_VP(VP_array,name='Visibility Phase', output='output_file.pdf')
    @endcode
    
    This is an example for multiple subplots:
        
    @code
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    fig.set_size_inches(12,12)
    
    plot_VP(VP_array1, fig=fig,ax1=ax1, colorbar=False)
    plot_VP(VP_array2, fig=fig,ax1=ax2, colorbar=False)
    plot_VP(VP_array3, fig=fig,ax1=ax3, colorbar=False)
    plot_VP(VP_array4, fig=fig,ax1=ax4, colorbar=False)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    
    Note that for multiple subplots you might want to omit color bars, 
    and only include the model name in one of the subplots.

    @param array 2D numpy array of the image to be plotted.
    
    @param fig the name of the figure where you want to plot your image, 
    see the example code above.
    
    @param ax1 the name of the subplot (where applicable) where you want to plot 
    your image, see the example code above.
    
    @param output optional keyword, default set to None, if not None should be 
    a string and  will save the figure to a file with file name equal to output. 
    @param name optional keyword, default set to None. If not None must be a 
    string and will add a text label to the plot equal to this string.
    
    @param name optional keyword, default set to None. If not None must be a 
    string and will add a text label to the plot equal to this string.
    
    @param btracks optional keyword, default set to True, if True will plot the 
    baseline tracks for the EHT over the VA map.
    
    @param font optional keyword, default set to 20, this sets 
    the font size for the axis labels, and numbers as well as the numbers for 
    the color bar.
    
    @param colorbar optional keyword, default set to True, if True will plot the 
    color bar, if False will do nothing, and if set to 'top' will plot colorbar 
    on top.
    
    @param pad int, optional keyword, default set to 8, factor by which I want 
    to pad my arrays before taking the fft.

    @param M int, optional keyword, default set to 64, size the the array in 
    units of \f$ GM/c^2 \f$.
    
    @param x_label optional keyword, default set to True. If True will add a 
    label to the x-axis, if False, will not add this label..
    
    @param y_label optional keyword, default set to True. If True will add a 
    label to the y-axis, if False, will not add this label.
    
    @param zoom optional keyword, default set to True. If set to True will zoom 
    in to about 20 \f$ G \lambda \f$ on each side, if not set to True, will leave 
    the full array visible, unless bounds is set.
    
    @param white_width optional keyword, default set to 5. This will control the 
    width of the white border around the black text on the plot.
    
    @param interpolation optional keyword, default set to 'bilinear'. This will 
    control the type of interpolation that is used in the plot, the options are 
    the same as those for matplotlib.
   
    @returns ax1 if ax1 not given, or the image object if ax1 is given.
    """
    make_fig = False
    if (fig == None) and (ax1 == None):
        make_fig =True
        columnwidth = 3.39441
        fig,(ax1) = plt.subplots(1,1)
        fig.set_size_inches(columnwidth,columnwidth)
    
    x       = np.shape(array)[0]
    r0      = x*np.sqrt(27)/M # this is the radius of the black hole shadow
    uvpix   = distance/(M*mass*pad)*2.09*10**7 # 2.09*10**7 is kpc c**2/(solar mass G)*10**-9, equation: D c**2/(M mass G pad)
    widthL  = (1/(np.pi*r0))*x*uvpix*pad
    ext     = widthL*4

    array[np.where(array >  np.pi)] = array[np.where(array >  np.pi)]-2.0*np.pi
    array[np.where(array < -np.pi)] = array[np.where(array < -np.pi)]+2.0*np.pi

    plt.set_cmap('hsv')
    if flip_x == True:
        im1=ax1.imshow(array,extent=[x/2*uvpix,-x/2*uvpix,-x/2*uvpix,x/2*uvpix], vmin=-np.pi, vmax=np.pi, origin='lower', interpolation=interpolation)
    else:
        im1=ax1.imshow(array,extent=[-x/2*uvpix,x/2*uvpix,-x/2*uvpix,x/2*uvpix], vmin=-np.pi, vmax=np.pi, origin='lower', interpolation=interpolation)
    if colorbar==True:
        divider1 = make_axes_locatable(ax1)
        cax1     = divider1.append_axes("right", size="7%", pad=0.05)
        cbar1    = plt.colorbar(im1, cax=cax1)
        cbar1.ax.tick_params(labelsize=font, color=cb_tick_color,width=1.5, direction='in')
    elif colorbar== 'top':
        divider1 = make_axes_locatable(ax1)
        cax1     = divider1.append_axes("top", size="7%", pad=0.05)
        cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
        cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
        cbar1.ax.xaxis.set_ticks_position('top') 
    if x_label==True: ax1.set_xlabel('$u$ (G $\lambda$)',fontsize=font)
    if y_label==True: ax1.set_ylabel('$v$ (G $\lambda$)',fontsize=font)
    if btracks == True:
        path    = __file__.replace('my_plot.pyc', '')
        U, V    = np.load(path+'U.npy'), np.load(path+'V.npy')
        U,V     = U*10**(-6),V*10**(-6)
        ax1.scatter( U, V, c='k', s=2, marker='o',edgecolors='none')
        ax1.scatter(-U,-V, c='k', s=2, marker='o',edgecolors='none')
    ax1.tick_params(axis='both', which='major', labelsize=font,color=tick_color,width=1.5, direction='in')
    if zoom == True:
        ax1.set_xlim([-ext,ext])
        ax1.set_ylim([-ext,ext])
        ax1.set_xticks([-5,0,5])
        ax1.set_yticks([-5,0,5])
        # if name != None: 
        #     txt = ax1.text(9,-9, name, fontsize=font,color='k') #makes the text label
        #     txt.set_path_effects([PathEffects.withStroke(linewidth=white_width, foreground='w')])
    else:         
        ax1.set_yticks(ax1.get_xticks())
        temp = ax1.get_xlim()
        ax1.set_ylim(temp[0], temp[1])
    if name !=None: 
        txt = ax1.text(.9*temp[0],-.9*temp[0], name, fontsize=font, color='k') #makes the text label
        txt.set_path_effects([PathEffects.withStroke(linewidth=white_width, foreground='w')])
    # if flip_x == True:
    #     temp = ax1.get_xlim()
    #     ax1.set_xlim(-1*temp[0], -1*temp[1])
    if output != None: 
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
    if make_fig == True:return(ax1)
    else: return(im1)    
    
 
def plot_base(array, fig, ax1, m, colors=np.array(['k','r']), line_style=np.array(['-','-']), dashes=None,
                labels=np.array(['$u=0$','$v=0$']),line_width=np.array([2,2]), data=True, x_scale='lin', 
                y_scale='log', model_text=True, font=20, lines=True, legend=True, norm=False, norm_num=1, subplot=False, 
                aspect=1, x_label='Baseline length (G $\lambda$)', y_label='Visibility Amplitude (Jy)', 
                x_lim='default', y_lim=np.array([0.008,4]), alpha=1, base_points=False,pad=8, M=64,mass=4.31*10**6, distance=8.3):      
                
    """!@brief Plots the vertical and horizontal cross sections of an array vs. 
    baseline.
    
    Takes a 2D array of floats assumed to be in u-v space, and plots the 
    vertical and horizontal cross sections of this array vs. baseline in 
    G-lambdas. 
    
    This function can be used for a single plot or for multiple subplots. 
    An example of how to use this function as a single plot is below:
        
    @code    
    fig,(ax1) = plt.subplots(1,1)
    fig.set_size_inches(6,6)
    
    plot_base(VA_array, fig, ax1, m)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    This is how you would create multiple subplots with this function:
    @code
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    fig.set_size_inches(12,12)
    
    plot_base(VA_array1, fig, ax1, m)
    plot_base(VA_array2, fig, ax2, m)
    plot_base(VA_array3, fig, ax3, m)
    plot_base(VA_array4, fig, ax4, m)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    for the keywords, it goes (vert, horz) 
   
    @param array 2D numpy array of image to be plotted.
    
    @param fig the name of the figure where you want to plot your image, 
    see the example code above.
    
    @param ax1 the name of the subplot (where applicable) where you want to plot 
    your image, see the example code above.
    
    @param m an integer between 0 and 4 which corresponds to the model, this 
    is used for labeling the model, see model_text, if model_text=False, then 
    m is irrelevant added support for m=-1, which is for when we aren't using 
    a model and want to normalize to 1. 
    
    @param colors optional keyword, default set to np.array(['k','b']), sets the 
    colors of the lines, the order of the array must be (vert, horz).
    
    @param line_style optional keyword, default set to np.array(['-','-']), can 
    change the linestyle of the lines, the order of the array must be 
    (vert, horz). 
    
    @param dashes optional keyword, default set to None. If not set to None can 
    be set to, for example [1,3,4,5] which would create a dash that is 1 point 
    on, 3 off, 4 on, 5 off.
    
    @param labels optional keyword, default set to np.array(['$u=0$','$v=0$']).
    
    @param line_width optional keyword, default set to np.array([2,2]). 
    
    @param data optional keyword, default set to True. 
    
    @param x_scale optional keyword, default set to 'lin'.
    
    @param y_scale optional keyword, default set to 'log'.
    
    @param model_text optional keyword, default set to True, when True 
    will add text to the image, example "Model A".
    
    @param font optional keyword, default set to 20, this sets 
    the font size for the axis labels, and numbers. 

    @param lines optional keyword, default set to True.
    
    @param legend optional keyword, default set to True.
    
    @param norm optional keyword, default set to False, if 
    True will normalize image so that the maximum is 1, 
    if false does not normalize image.
    
    @param norm_num optional keyword, default set to 1, this is in case 
    normalizing to 1 doesn't give the desired image, you can control a more 
    specific normalization with norm_num.
    
    @param subplot optional keyword, default set to False.
    
    @param aspect optional keyword, default set to 1.
    
    @param x_label optional keyword, default set to 'Baseline length 
    (G $\\lambda$)'.
    
    @param y_label optional keyword, default set to 'Visibility Amplitude (Jy)'.
    
    @param x_lim optional keyword, default set to 'default' which will set it 
    equal to np.array([0,ext]).
    
    @param y_lim optional keyword, default set to np.array([0.008,4]).
    
    @param alpha optional keyword, default set to 1. Set alpha to something 
    smaller if you would like the lines to not be completely opaque.
    
    @param base_points optional keyword, default set to False. If not equal to 
    False, will plot a green point on the curve for the vertical cross section 
    that corresponds to the length of the baseline formed by the South Pole and 
    Pico Veleta, and a magenta point on the curve for the horizontal cross 
    section that corresponds to the length of the baseline formed by the SMA and 
    the LMT.
    
    @param pad int, optional keyword, default set to 8, factor by which I want 
    to pad my arrays before taking the fft.
    
    @param x int, optional keyword, default set to 512, size of the x axis of 
    the array.

    @param M int, optional keyword, default set to 64, size the array in units 
    of \f$ GM/c^2 \f$.
    
    @returns 0
    """
    models  = np.array(['A','B','C','D','E','None'])
    x       = np.shape(array)[0]
    r0      = x*np.sqrt(27)/M # this is the radius of the black hole shadow
    uvpixel   = distance/(M*mass*pad)*2.09*10**7 # 2.09*10**7 is kpc c**2/(solar mass G)*10**-9, equation: D c**2/(M mass G pad)


    # uvpixel = 0.6288/pad 
    # r0      = x*np.sqrt(27)/M # this is the radius of the black hole shadow
    widthL  = (1./(np.pi*r0))*x*uvpixel*pad
    ext     = widthL*4.
    
    if type(x_lim)==str:x_lim=np.array([0,ext])
    
    array_horz = array[int(x/2.) ,int(x/2.):]
    array_vert = array[int(x/2.):,int(x/2.) ] 
    
    if norm == True:
        norm_model = np.array([3,3,2.6,2.6,2.5,1])   
        norm = 1/np.max(array_vert)*norm_model[m]
        array_horz = array_horz*norm*norm_num
        array_vert = array_vert*norm*norm_num
    
    if data == True:
        B07_d100, amp07_d100, err07_d100 = np.load('/home/u31/lia00/research/lia/EHT_obs/EHT_obs_2007_d100.npy')
        B07_d101, amp07_d101, err07_d101 = np.load('/home/u31/lia00/research/lia/EHT_obs/EHT_obs_2007_d101.npy')
        B09_d95,  amp09_d95,  err09_d95  = np.load('/home/u31/lia00/research/lia/EHT_obs/EHT_obs_2009_d95.npy')
        B09_d96,  amp09_d96,  err09_d96  = np.load('/home/u31/lia00/research/lia/EHT_obs/EHT_obs_2009_d96.npy')
        B09_d97,  amp09_d97,  err09_d97  = np.load('/home/u31/lia00/research/lia/EHT_obs/EHT_obs_2009_d97.npy')
        obsx = np.append(B07_d100,  B07_d101)
        obsx = np.append(obsx, B09_d95)
        obsx = np.append(obsx, B09_d96)
        obsx = np.append(obsx, B09_d97)
        obsy = np.append(amp07_d100,  amp07_d101)
        obsy = np.append(obsy, amp09_d95)
        obsy = np.append(obsy, amp09_d96)
        obsy = np.append(obsy, amp09_d97)
        obse = np.append(err07_d100,  err07_d101)
        obse = np.append(obse, err09_d95)
        obse = np.append(obse, err09_d96)
        obse = np.append(obse, err09_d97)
        obsx = obsx*10**(-3)
        ax1.errorbar(obsx,obsy, yerr=obse, color='k',fmt='.',markersize=5,markeredgecolor='none', ecolor='grey')
      
    if subplot == True:
        im1      = ax1.imshow(np.zeros((2,2)), extent=[-2,-1,-2,-1])
        divider1 = make_axes_locatable(ax1)
        cax1     = divider1.append_axes("right", size="7%", pad=0.05)
        cbar1    = plt.colorbar(im1, cax=cax1)
        cax1.set_visible(False)  
    
    base = (np.arange(int(x/2))+.5)*uvpixel
    ax1.set_xlabel(x_label,      fontsize=font)
    ax1.set_ylabel(y_label,      fontsize=font)
    if m ==-1 and y_label=='Visibility Amplitude (Jy)': ax1.set_ylabel('Visibility Amplitude (Normalized)',      fontsize=font)

    if np.shape(labels)[0]==2: 
        vert, = ax1.plot(base, array_vert, color=colors[0],lw=line_width[0], ls = line_style[0], label=labels[0], alpha=alpha)# vertical, parallel to spin
        horz, = ax1.plot(base, array_horz, color=colors[1],lw=line_width[1], ls = line_style[0], label=labels[1], alpha=alpha)# horizontal, perpendicular to spin 
    else: 
        vert, = ax1.plot(base, array_vert, color=colors[0],lw=line_width[0], ls = line_style[0], label=labels[0], alpha=alpha)# vertical, parallel to spin
        horz, = ax1.plot(base, array_horz, color=colors[1],lw=line_width[1], ls = line_style[0], alpha=alpha)# horizontal, perpendicular to spin     
    
    ax1.set_xlim([x_lim[0], x_lim[1]])
    ax1.set_ylim([y_lim[0], y_lim[1]])
    if legend == True: ax1.legend(loc=1,prop={'size':15})
    ax1.tick_params(axis='both', which='major', labelsize=font)
    if lines == True: 
        widthL = 3.04575
        widthG = 1.9453
        ax1.axvline(x=widthL, linewidth=1, color='r')
        ax1.axvline(x=widthG, linewidth=1, color='green')
        
    if base_points==True:
        sp_pv   = 8.88639381
        sma_lmt = 4.4596918

        # now we need to find the y-values for the 4 points above
        sp_pvy   = array_vert[base > sp_pv   ][0]
        sma_lmty = array_horz[base > sma_lmt ][0]
        
        ax1.scatter(sp_pv,sp_pvy,       c='g', s=100, edgecolors='k')
        ax1.scatter(sma_lmt, sma_lmty,  c='m', s=100, edgecolors='k')

        
        ax1.text(4.5, 1.2e-4,  'SPT-PICO', fontsize=font-6, color='g')
        ax1.text(.5, 1.2e-4, 'SMA-LMT',  fontsize=font-6, color='m')   
        
    if dashes != None: 
        vert.set_dashes(dashes)
        horz.set_dashes(dashes)
        
    if x_scale =='log': ax1.set_xscale('log',nonposy='clip')
    if y_scale =='log': ax1.set_yscale('log',nonposy='clip')
    if model_text == True: ax1.text(.8*ext,-3.8, "Model "+models[m], fontsize=font,color='k') #makes the text label
    if aspect ==1 :ax1.set_aspect('auto')
    return(0)
    
def rotate_tri(angle, base_U, base_V, deg=True): 
    """!@brief Rotates a closure triangle in u-v space.
    
    Given an angle, and the original closure triangle in u-v space, this 
    rotates the u-v coordinates by the given angle.
    
  

    @param angle float, defaults to degrees.
    
    @param base_U numpy array of three elements corresponding to the u 
    components of a triangle in u-v space.
    
    @param base_V numpy array of three elements corresponding to the v 
    components of a triangle in u-v space.
    
    @param deg optional keyword, default set to True, if angle is in radians, 
    then set def=False.
    
    @returns 2 numpy arrays with the rotated U and V, just like base_U and 
    base_V
    """

    if deg==True: angle = angle/360*2*np.pi
    U_prime = np.array([base_U[0]*np.cos(angle)-base_V[0]*np.sin(angle),base_U[1]*np.cos(angle)-base_V[1]*np.sin(angle),base_U[2]*np.cos(angle)-base_V[2]*np.sin(angle)])
    V_prime = np.array([base_V[0]*np.cos(angle)+base_U[0]*np.sin(angle),base_V[1]*np.cos(angle)+base_U[1]*np.sin(angle),base_V[2]*np.cos(angle)+base_U[2]*np.sin(angle)])
    return(U_prime,V_prime)
    
    
def uv_triangles(tri,angle=0):
    """!@brief Returns the u-v locations of the four 
    triangles we used for Medeiros et al. 2016b.
    
    The numbering for the triangles goes in increasing order as follows:
        
    -HI-AZ-CA
    
    -AZ-MX-CH
    
    -AZ-CH-SP
    
    -HI-AZ-SP
    
    Cautionary note: when I first started writting this code, I ordered it 
    1,4,3,2 so some of the hdf5 files are still numbered that way.
    

    @param tri integer, can be 1,2,3,or 4, this is how you specify which 
    triangle you want. 
    
    @param angle optional keyword, default set to 0, if set to something other 
    than zero, will rotate the u-v coords of your chosen triangle by the 
    specified angle, assumes angle is in degrees, uses rotate_tri(). 
    
    @returns U,V, where U and V are each numpy arrays with three elements which 
    correspond to the 3 corners of the triangle.
    """

    if tri==1:
        U   = np.array([ 3.38228814,-0.59857075,-2.78371740])
        V   = np.array([ 0.93785385, 0.20948923,-1.14734308])
    if tri==2: 
        U   = np.array([ 1.06205228, 1.70932966,-2.77138194])
        V   = np.array([-0.82489363,-3.78473120, 4.60962483])  
    if tri==3:
        U   = np.array([ 2.77138194,-3.63866754, 0.86728559])
        V   = np.array([-4.60962483,-3.84944387, 8.45906870])
    if tri==4:
        U   = np.array([ 3.38228814,-0.86728559,-2.51500255])
        V   = np.array([ 0.93785385,-8.45906870, 7.52121484])
    if angle !=0: U,V = rotate_tri(angle, U,V)
    return(U,V)
      
      
def plot_tri(tri, fig, ax1, color='red', angle=0, line_style='-',label='default', legend=True):
    """!@brief Adds one of the four triangles we use to a plot. 
    
    Plots one of the four triangles we used for Medeiros et al. 2016b.
    
    An example of how to use this with plot_VA():
    @code    
    fig,(ax1) = plt.subplots(1,1)
    fig.set_size_inches(6,6)
    
    plot_VA(VA_array, fig, ax1, m)
    plot_tri(1,fig,ax1)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    The numbering for the triangles goes in increasing order as follows:
        
    -HI-AZ-CA
    
    -AZ-MX-CH
    
    -AZ-CH-SP
    
    -HI-AZ-SP
    
    This code uses uv_triangles(). 
    
    Cautionary note: when I first started writting this code, I ordered it 
    1,4,3,2 so some of the hdf5 files are still numbered that way.
    

    @param tri integer, can be 1,2,3,or 4, this is how you specify which 
    triangle you want. 
   
    @param fig the name of the figure where you want to plot your image, 
    see the example code above.
    
    @param ax1 the name of the subplot (where applicable) where you want to plot 
    your image, see the example code above.
    
    @param color optional keyword, default set to 'red', use this to set the 
    color of the triangle, must be a string.
        
    @param angle optional keyword, default set to 0, if not zero will rotate 
    the triangles by the angle given must be in degrees.
    
    @param line_style optional keyword, default set to '-', can change the 
    linestyle to the lines that make the triangles.
    
    @param label optional keyword, default set to 'default', if default will 
    label the triangles based on their locations as specified above, otherwise 
    can set the label to whatever you want.
    
    @param legend optional keyword, default set to True, if True will plot a 
    legend based on label.
    
    @returns 0
    """
    
    triangles = np.array(['HI-AZ-CA','AZ-MX-CH','AZ-CH-SP','HI-AZ-SP'])
    
    if label == 'default':my_label = triangles[tri-1]
    else: my_label = label
    
    U,V = uv_triangles(tri,angle=angle) 
    ax1.plot(np.array([U[0],U[1]]), np.array([V[0],V[1]]), color=color,linewidth=2, ls=line_style, label = my_label)
    ax1.plot(np.array([U[1],U[2]]), np.array([V[1],V[2]]), color=color,linewidth=2, ls=line_style)
    ax1.plot(np.array([U[2],U[0]]), np.array([V[2],V[0]]), color=color,linewidth=2, ls=line_style)
    
    if legend ==True: ax1.legend(loc=3,prop={'size':14}) 
    ax1.set_frame_on(True)
    return(0)
    

def plot_triangles(fig, ax1, angle = 0, lengend = True, colors=np.array(['red', 'magenta','green','blue']), line_style='-'):
    """!@brief Adds the four triangles we use to a plot. 
    
    Plots the four triangles we used for Medeiros et al. 2016b.
    
    An example of how to use this with plot_VA():
    @code    
    fig,(ax1) = plt.subplots(1,1)
    fig.set_size_inches(6,6)
    
    plot_VA(VA_array, fig, ax1, m)
    plot_triangles(fig,ax1)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    The numbering for the triangles goes in increasing order as follows:
        
    -HI-AZ-CA
    
    -AZ-MX-CH
    
    -AZ-CH-SP
    
    -HI-AZ-SP
    
    This code uses plot_tri(). 
    
    Cautionary note: when I first started writting this code, I ordered it 
    1,4,3,2 so some of the hdf5 files are still numbered that way.

    @param fig the name of the figure where you want to plot your image, 
    see the example code above.
    
    @param ax1 the name of the subplot (where applicable) where you want to plot 
    your image, see the example code above.
    
    @param angle optional keyword, default set to 0, if not zero will rotate 
    the triangles by the angle given must be in degrees.
    
    @param legend optional keyword, default set to True, if True will plot a 
    legend labeling the triangles.
    
    @param colors optional keyword, default set to 
    np.array(['red', 'magenta','green','blue']), use this to set the colors of 
    the four triangles, must be a numpy array with 4 strings.
    
    @param line_style optional keyword, default set to '-', can change the 
    linestyle to the lines that make the triangles.
    
    @returns 0
    
    @callgraph
    """
    for i in range(1,5):
        plot_tri(i, fig, ax1, color=colors[i-1], angle=angle, line_style=line_style, legend=False)

    if lengend == True: ax1.legend(loc=3,prop={'size':9})  
    return(0)
    
    
def plot_CPvT(CP1, CP2, CP3, CP4, ax1,ax2,ax3,ax4, color='r', marker='o', font=12, scatter=True, line_style='-', label='', line=True, linewidth=1):
    """!@brief Plots closure phases as a function of time for the four CP 
    triangles we consider.
    
    Takes in four subplots and will plot each triangle on a separate subplot.
    
    The numbering for the triangles goes in increasing order as follows:
        
    -HI-AZ-CA
    
    -AZ-MX-CH
    
    -AZ-CH-SP
    
    -HI-AZ-SP
    
    
    Cautionary note: when I first started writting this code, I ordered it 
    1,4,3,2 so some of the hdf5 files are still numbered that way.

    @param CP1 1D numpy array containing the closure phase for triangle 1 in 
    units of degrees.
    
    @param CP2 1D numpy array containing the closure phase for triangle 2 in 
    units of degrees.
    
    @param CP3 1D numpy array containing the closure phase for triangle 3 in 
    units of degrees.
    
    @param CP4 1D numpy array containing the closure phase for triangle 4 in 
    units of degrees.
    
    @param ax1 the name of the subplot (where applicable) where you want to plot 
    triangle 1.
    
    @param ax2 the name of the subplot (where applicable) where you want to plot 
    triangle 2.
    
    @param ax3 the name of the subplot (where applicable) where you want to plot 
    triangle 3.
    
    @param ax4 the name of the subplot (where applicable) where you want to plot 
    triangle 4.
    
    @param color optional keyword, default set to 'r', use this to set the color
    of the markers or the lines.
    
    @param marker optional keyword, default set to 'o'. Use this to change the 
    markers in the scatter plot if scatter is set to True.
    
    @param optional keyword, default set to 12, this sets the size of the font 
    for axis labels and any other text.
    
    @param scatter optional keyword, default set to True, if True will make a 
    scatter plot.
    
    @param line_style optional keyword, default set to '-', can change the 
    linestyle of the curve if line is set to True.
    
    @param label optional keyword, default set to '', can use this if you want 
    to label the lines, for example if you want to put vertical and horizontal 
    on the same plots. 
    
    @param line optional keyword, default set to True, if True will plot a line.
    
    @param linewidth optional keyword, default set to 1, this controls with 
    thickness of the line if line is set to True.
    
    @returns 0
    """
    
    axis = np.arange(1024)*211.8/3600
    
    if scatter == True:
        ax1.scatter(axis, CP1, label=label, marker=marker,  color=color ,s=1)
        ax2.scatter(axis, CP2, label=label, marker=marker,  color=color ,s=1)
        ax3.scatter(axis, CP3, label=label, marker=marker,  color=color ,s=1)
        ax4.scatter(axis, CP4, label=label, marker=marker,  color=color ,s=1)
        
    if line == True:
        ax1.plot(   axis, CP1, label=label, ls=line_style,  color=color, linewidth = linewidth)
        ax2.plot(   axis, CP2, label=label, ls=line_style,  color=color, linewidth = linewidth)
        ax3.plot(   axis, CP3, label=label, ls=line_style,  color=color, linewidth = linewidth)
        ax4.plot(   axis, CP4, label=label, ls=line_style,  color=color, linewidth = linewidth)

    ax1.set_ylabel(r'HI-AZ-CA',   fontsize=font)
    ax2.set_ylabel(r'AZ-MX-CH',   fontsize=font)
    ax3.set_ylabel(r'AZ-CH-SP',   fontsize=font)
    ax4.set_ylabel(r'HI-AZ-SP',   fontsize=font)
    
    ax4.set_xlabel(r'Time (Hrs)', fontsize=font)
    
    ax3.set_xlabel(r'Time $t-t_{mathrm{begin}}$ (10$GMc^{-3}$)', fontsize=font)
    ax1.set_ylim([-180,180])
    ax2.set_ylim([-180,180])
    ax3.set_ylim([-180,180])
    ax4.set_ylim([-180,180])
    ax1.set_xlim([0,np.max(axis)])
    ax2.set_xlim([0,np.max(axis)])
    ax3.set_xlim([0,np.max(axis)])
    ax4.set_xlim([0,np.max(axis)])
    ax1.set_yticks([-90,0,90])
    ax2.set_yticks([-90,0,90])
    ax3.set_yticks([-90,0,90])
    ax4.set_yticks([-90,0,90])
    plt.subplots_adjust(hspace = 0.0)
    return(0)
      
        
            
    
        
    
def plot_power_spec(freq,FFT,params_both, tri=0, orient='vert', line=True, extra=False, plot_fit=True, marker='o', color_marker='blue', text=False):
    """!@brief Plots a power spectrum.
    
    @param freq in units of Hz, this will be the x-axis of the plot.
    
    @param FFT the power spectrum you wish to plot, normalized. The actual plot 
    will have FFT*freq, of \f$ f\times P(f)\f$.
    
    @param params_both this should contain the results of a fit to the power 
    spectrum that you wish to plot, see the functions in PS_utils, this will get 
    fed into power_law_aliased().
    
    @param tri optional keyword, default set to 0, if the power spectrum is for 
    one of the four EHT closure triangles used in plot_triangles(), for example, 
    this variable should be set to a number between 1 and 4 that corresponds to 
    the triangle in question.
    
    @param orient optional keyword, default set to 'vert'. If the power spectrum is of 
    a C.P., this should be set to 'vert' if the triangle is in the vertical 
    orientation (BH points N), or horz if in the horizontal orientation, 
    (BH points E).
    
    @param line optional keyword, default set to True. If True will plot a 
    vertical line at the break point, as determined by the fit.
    
    @param extra optional keyword, default set to False. If set to True, will 
    constrain the x and y bounds, will add x and y labels, will make both axes 
    log-scale, and gives the option to add some text to the plot if text if set 
    to True.
    
    @param plot_fit optional keyword, default set to True. If True will plot a 
    fit to the curve as well.
    
    @param marker optional keyword, default set to 'o'. This is the marker type 
    that will be used in the plot.
    
    @param color_marker optional keyword, default set to 'blue', this will be 
    the color of the markers used in the plot.
    
    @param text optional keyword, default set to False. if set to True, will add 
    text to the plot that says which closure phase triangle was used.
    
    @returns 0
    """
    
    ax1.scatter(freq*60, freq*FFT*60, color=color_marker, marker=marker) 
    if plot_fit==True:      
        fit  = power_law_aliased(freq, freq*FFT,  params_both)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        ax1.plot(freq*60,  freq*fit*60,   color='black', linewidth=1)
    
    if line==True: ax1.axvline(x=params_both[1]*60, color='black',linewidth=1)
    if extra==True:
        if text==True: 
            triangles = np.array(['HI-AZ-CA','AZ-MX-CH','AZ-CH-SP','HI-AZ-SP'])
            ax1.text(2*10**-3*60, .7*60, triangles[int(tri-1)]+', '+orient, fontsize=16)
        ax1.set_xscale("log", nonposy='clip')
        ax1.set_yscale("log", nonposy='clip')
        ax1.set_ylim([10**-3*60, 1*60])
        ax1.set_xlim([4*10**-5*60, 4*10**-2*60])
        ax1.set_xlabel('Frequency (min$^{-1}$)', fontsize=16)
        ax1.set_ylabel('Normalized f*P(f)', fontsize=16)
    return(0)
    
    
    #ax5.set_xlabel('Baseline length (G $\lambda$)',      fontsize=20)
    #ax5.set_ylabel('Visibility Amplitude (Jy)',          fontsize=20)
    #ax5.fill_between(base,average_horz[0,:]-average_horz[1,:], average_horz[0,:]+average_horz[1,:],facecolor='black' , alpha=0.3,interpolate=True)
    #ax5.plot(base,average_horz[0,:]                  , color='black',linewidth=2, label='$u=0$')
    #ax5.fill_between(base,average_vert[0,:]-average_vert[1,:], average_vert[0,:]+average_vert[1,:],facecolor='blue'  , alpha=0.3,interpolate=True)
    #ax5.plot(base,average_vert[0,:]                  , color='blue' ,linewidth=2, label='$v=0$')      
    #ax5.errorbar(obsx,obsy, yerr=obse, color='k',fmt='.',markersize=5,markeredgecolor='none', ecolor='grey')
    #ax5.axvline(x=widthL, linewidth=1, color='r')
    #ax5.set_ylim([0.008,4])
    #ax5.set_xlim([0,ext])
    ##ax5.legend(loc=1,prop={'size':15})
    #ax5.tick_params(axis='both', which='major', labelsize=20)
    #divider5 = make_axes_locatable(ax5)
    #cax5     = divider5.append_axes("right", size="7%", pad=0.05)
    #cbar5 = plt.colorbar(im3, cax=cax5)
    #cax5.set_visible(False)
    #ax5.set_yscale('log')
    #ax5.text(5,2, "mean of VA", fontsize=20,color='k')
    
    
def plot_frac_rms(array, fig, ax1, m, colors=np.array(['k','r']), line_style=np.array(['-','-']), 
                label='no scatter',line_width=np.array([2,2]), x_scale='lin', y_scale='lin', 
                font=20, legend=True, aspect=1, x_label='Baseline length (G $\lambda$)', 
                y_label='fractional RMS', x_lim='default', y_lim=np.array([0,1.4]),pad=8, 
                x=512, M=64, horz=False, base_points=True):      
                
    """!@brief Plots the vertical and horizontal cross sections of an array vs. baseline.
    
    Takes a 2D array of floats assumed to be in u-v space, and plots the vertical 
    and horizontal cross sections of this array vs. baseline in G-lambdas. 
    
    This function can be used for a single plot or for multiple subplots. 
    An example of how to use this function as a single plot is below:
        
    @code    
    fig,(ax1) = plt.subplots(1,1)
    fig.set_size_inches(6,6)
    
    plot_base(VA_array, fig, ax1, m)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    This is how you would create multiple subplots with this function:
    @code
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    fig.set_size_inches(12,12)
    
    plot_base(VA_array1, fig, ax1, m)
    plot_base(VA_array2, fig, ax2, m)
    plot_base(VA_array3, fig, ax3, m)
    plot_base(VA_array4, fig, ax4, m)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
   
    @param array 2D numpy array of image to be plotted
    
    @param fig the name of the figure where you want to plot your image, 
    see the example code above
    
    @param ax1 the name of the subplot (where applicable) where you want to plot 
    your image, see the example code above
    
    @param m an integer between 0 and 4 which corresponds to the model, this 
    is used for labeling the model, see model_text, if model_text=False, then 
    m is irrelevant added support for m=-1, which is for when we aren't using 
    a model and want to normalize to 1. 
    
    @param colors optional keyword, default set to np.array(['k','b']), sets the 
    colors of the lines, the order of the array must be (vert, horz)
    
    @param line_style optional keyword, default set to np.array(['-','-']), can 
    change the linestyle of the lines, the order of the array must be (vert, horz) 
    
    @param labels optional keyword, default set to np.array(['$u=0$','$v=0$'])
    
    @param line_width optional keyword, default set to np.array([2,2]) 
    
    @param x_scale optional keyword, default set to 'lin'
    
    @param y_scale optional keyword, default set to 'log'
    
    @param font optional keyword, default set to 20, this sets 
    the font size for the axis labels, and numbers 
    
    @param legend optional keyword, default set to True
    
    @param aspect optional keyword, default set to 1
    
    @param x_label optional keyword, default set to 'Baseline length (G $\\lambda$)'
    
    @param y_label optional keyword, default set to 'Visibility Amplitude (Jy)'
    
    @param x_lim optional keyword, default set to 'default' which will set it 
    equal to np.array([0,ext])
    
    @param y_lim optional keyword, default set to np.array([0.008,4]) 
    
    @param pad int, optional keyword, default set to 8, factor by which I want 
    to pad my arrays before taking the fft.
    
    @param x int, optional keyword, default set to 512, size of the x axis of 
    the array.

    @param M int, optional keyword, default set to 64, size the array in units 
    of \f$ GM/c^2 \f$.
    
    @param horz optional keyword, default set to False. If the black hole in the 
    simulation has a spin axis which points East, then this should be set to 
    True. This will control some labels on the plot.
    
    @param base_points optional keyword, default set to False. If not equal to 
    False, will plot a green point on the curve for the vertical cross section 
    that corresponds to the length of the baseline formed by the South Pole and 
    Pico Veleta, and a magenta point on the curve for the horizontal cross 
    section that corresponds to the length of the baseline formed by the SMA and 
    the LMT.
    
    @returns 0
    """
    uvpixel = 0.6288/pad 
    r0      = x*np.sqrt(27)/M # this is the radius of the black hole shadow
    widthL  = (1/(np.pi*r0))*x*0.6288
    ext     = widthL*4
    
    if type(x_lim)==str:x_lim=np.array([0,ext])
    
    array_horz = array[int(x/2) ,int(x/2):]
    array_vert = array[int(x/2):,int(x/2) ] 
    
    base = np.arange(int(x/2))*uvpixel
    ax1.set_xlabel(x_label,      fontsize=font)
    ax1.set_ylabel(y_label,      fontsize=font)

    ax1.plot(base, array_vert, color=colors[0],lw=line_width[0], ls = line_style[0], label=label)# vertical, parallel to spin
    ax1.plot(base, array_horz, color=colors[1],lw=line_width[1], ls = line_style[0])# horizontal, perpendicular to spin      
    
    ax1.set_xlim([x_lim[0], x_lim[1]])
    ax1.set_ylim([y_lim[0], y_lim[1]])
    ax1.set_yticks([0.0,.2,.4,.6,.8,1.0,1.2,1.4])
    if legend == True: ax1.legend(loc=2,prop={'size':16}, frameon=False)
    ax1.tick_params(axis='both', which='major', labelsize=font)
    
    if x_scale =='log': ax1.set_xscale('log',nonposy='clip')
    if y_scale =='log': ax1.set_yscale('log',nonposy='clip')
    if horz == True: #E_W 
        ax1.text(15.8,1.3, 'E-W spin',     fontsize=font  ,color='k')
        #ax1.text(2,.3, 'perpendicular',fontsize=font-3,color=colors[0])
        #ax1.text(3,  .97, 'parallel',     fontsize=font-3,color=colors[1])
    if horz == False: #N_S 
        ax1.text(15.8,1.3, 'N-S spin',     fontsize=font  ,color='k')
        #ax1.text(2,.3, 'perpendicular',fontsize=font-3,color=colors[1])
        #ax1.text(3,  .97, 'parallel',     fontsize=font-3,color=colors[0])
    if aspect ==1 :ax1.set_aspect('auto')
    
    if base_points == True:
        sp_pv   = 8.88639381
        sma_lmt = 4.4596918

        # now we need to find the y-values for the 2 points above
        sp_pvy   = array_vert[base > sp_pv   ][0]
        sma_lmty = array_horz[base > sma_lmt ][0]
        
        ax1.scatter(sp_pv,sp_pvy,       c='g', s=100, edgecolors='k')
        ax1.scatter(sma_lmt, sma_lmty,  c='m', s=100, edgecolors='k')
        
        ax1.text(13.2,1.1, 'SPT-PICO',     fontsize=font-6  ,color='g')
        ax1.text(16.2,1.1, 'SMA-LMT',      fontsize=font-6  ,color='m')
    
    return(0)
    
    
    
def plot_disp_base(array, fig, ax1, m, colors=np.array(['k','r']), line_style=np.array(['-','-']), 
                label='no scatter',line_width=np.array([2,2]), x_scale='lin', y_scale='lin', 
                font=20, legend=True, aspect=1, x_label='Baseline length (G $\lambda$)', 
                y_label='Directional Dispersion (D) of Phase', x_lim='default', y_lim=np.array([0,1.4]),pad=8, 
                x=512, M=64, horz=False, base_points=True):      
                
    """!@brief Plots the vertical and horizontal cross sections of an array vs. 
    baseline.
    
    Takes a 2D array of floats assumed to be in u-v space, and plots the 
    vertical and horizontal cross sections of this array vs. baseline in 
    G-lambdas. The difference between this and plot_base() is that this is for 
    the directional dispersion as a function of baseline and so the axis labels 
    and bounds are different.
    
    This function can be used for a single plot or for multiple subplots. 
    An example of how to use this function as a single plot is below:
        
    @code    
    fig,(ax1) = plt.subplots(1,1)
    fig.set_size_inches(6,6)
    
    plot_disp_base(VA_array, fig, ax1, m)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
    
    This is how you would create multiple subplots with this function:
    @code
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    fig.set_size_inches(12,12)
    
    plot_disp_base(VA_array1, fig, ax1, m)
    plot_disp_base(VA_array2, fig, ax2, m)
    plot_disp_base(VA_array3, fig, ax3, m)
    plot_disp_base(VA_array4, fig, ax4, m)
    
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    @endcode
   
    @param array 2D numpy array of image to be plotted
    
    @param fig the name of the figure where you want to plot your image, 
    see the example code above
    
    @param ax1 the name of the subplot (where applicable) where you want to plot 
    your image, see the example code above
    
    @param m an integer between 0 and 4 which corresponds to the model, this 
    is used for labeling the model, see model_text, if model_text=False, then 
    m is irrelevant added support for m=-1, which is for when we aren't using 
    a model and want to normalize to 1. 
    
    @param colors optional keyword, default set to np.array(['k','b']), sets the 
    colors of the lines, the order of the array must be (vert, horz)
    
    @param line_style optional keyword, default set to np.array(['-','-']), can 
    change the linestyle of the lines, the order of the array must be (vert, horz) 
    
    @param labels optional keyword, default set to np.array(['$u=0$','$v=0$'])
    
    @param line_width optional keyword, default set to np.array([2,2]) 
    
    @param x_scale optional keyword, default set to 'lin'
    
    @param y_scale optional keyword, default set to 'log'
    
    @param font optional keyword, default set to 20, this sets 
    the font size for the axis labels, and numbers 
    
    @param legend optional keyword, default set to True
    
    @param aspect optional keyword, default set to 1
    
    @param x_label optional keyword, default set to 'Baseline length (G $\\lambda$)'
    
    @param y_label optional keyword, default set to 'Visibility Amplitude (Jy)'
    
    @param x_lim optional keyword, default set to 'default' which will set it 
    equal to np.array([0,ext])
    
    @param y_lim optional keyword, default set to np.array([0.008,4]) 
    
    @param pad int, optional keyword, default set to 8, factor by which I want 
    to pad my arrays before taking the fft.
    
    @param x int, optional keyword, default set to 512, size of the x axis of 
    the array.

    @param M int, optional keyword, default set to 64, size the array in units 
    of \f$ GM/c^2 \f$.
        
    @param horz optional keyword, default set to False. If the black hole in the 
    simulation has a spin axis which points East, then this should be set to 
    True. This will control some labels on the plot.
    
    @param base_points optional keyword, default set to False. If not equal to 
    False, will plot a green point on the curve for the vertical cross section 
    that corresponds to the length of the baseline formed by the South Pole and 
    Pico Veleta, and a magenta point on the curve for the horizontal cross 
    section that corresponds to the length of the baseline formed by the SMA and 
    the LMT.
    
    @returns 0
    """
    uvpixel = 0.6288/pad 
    r0      = x*np.sqrt(27)/M # this is the radius of the black hole shadow
    widthL  = (1/(np.pi*r0))*x*0.6288
    ext     = widthL*4
    
    if type(x_lim)==str:x_lim=np.array([0,ext])
    
    array_horz = array[int(x/2) ,int(x/2):]
    array_vert = array[int(x/2):,int(x/2) ] 
    
    base = np.arange(int(x/2))*uvpixel
    ax1.set_xlabel(x_label,      fontsize=font)
    ax1.set_ylabel(y_label,      fontsize=font)

    ax1.plot(base, array_vert, color=colors[0],lw=line_width[0], ls = line_style[0], label=label)# vertical, parallel to spin
    ax1.plot(base, array_horz, color=colors[1],lw=line_width[1], ls = line_style[0])# horizontal, perpendicular to spin      
    
    ax1.set_xlim([x_lim[0], x_lim[1]])
    ax1.set_ylim([y_lim[0], y_lim[1]])
    ax1.set_yticks([0.0,.2,.4,.6,.8,1.0,1.2,1.4])
    if legend == True: ax1.legend(loc=2,prop={'size':16}, frameon=False)
    ax1.tick_params(axis='both', which='major', labelsize=font)
    
    if x_scale =='log': ax1.set_xscale('log',nonposy='clip')
    if y_scale =='log': ax1.set_yscale('log',nonposy='clip')
    if horz == True: #E_W 
        ax1.text(15.8,1.3, 'E-W spin',     fontsize=font  ,color='k')
        #ax1.text(9,.05, 'perpendicular',fontsize=font-3,color=colors[0])
        #ax1.text(10,1.05, 'parallel',     fontsize=font-3,color=colors[1])
    if horz == False: #N_S 
        ax1.text(15.8,1.3, 'N-S spin',     fontsize=font  ,color='k')
        #ax1.text(9,.05, 'perpendicular',fontsize=font-3,color=colors[1])
        #ax1.text(10,1.05, 'parallel',     fontsize=font-3,color=colors[0])
    if aspect ==1 :ax1.set_aspect('auto')
    
    if base_points == True:
        sp_pv   = 8.88639381
        sma_lmt = 4.4596918

        # now we need to find the y-values for the 2 points above
        sp_pvy   = array_vert[base > sp_pv   ][0]
        sma_lmty = array_horz[base > sma_lmt ][0]
        
        ax1.scatter(sp_pv,sp_pvy,       c='g', s=100, edgecolors='k')
        ax1.scatter(sma_lmt, sma_lmty,  c='m', s=100, edgecolors='k')
        
        ax1.text(13.2,1.1, 'SPT-PICO',     fontsize=font-6  ,color='g')
        ax1.text(16.2,1.1, 'SMA-LMT',      fontsize=font-6  ,color='m')
    return(0)
    
    
    
def plot_present(array, fig=None, ax1=None, spin=None, theta=60.0, output=None, name=None, norm=True, scale='lin', 
	font=10.56, colorbar='bottom', norm_num=1, lim_lin=np.array([0,1]), lim_log=False, flip_x=False, 
	horz=False, M=64, x_label=False, y_label=False, colorbar_ticks='set', circle_width=1, 
	zoom=True, tick_color='w', cb_tick_color='k', label_color='w',name_size=None, scale_bar=True,colormap='gist_heat', gamma=.2,no_border=False):
    if name_size==None: label_font_sz = font
    else: label_font_sz = name_size
    make_fig = False
    if (fig == None) and (ax1 == None):
        make_fig =True
        columnwidth = 3.39441
        fig,(ax1) = plt.subplots(1,1)
        fig.set_size_inches(columnwidth,columnwidth)
    
    x       = np.shape(array)[0]
    r0      = x*np.sqrt(27)/M # this is the radius of the black hole shadow
    r0M     = r0*M/x # this gives the BH shadow in units of GM/c**2
    if norm == True: array = array/(np.max(array))*norm_num
    
    plt.set_cmap(colormap)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if scale == 'lin':
        if flip_x == True: 
            array = np.fliplr(array)
            im1   = ax1.imshow(array, extent=[M/2.0,-M/2.0,-M/2.0,M/2.0],vmin=lim_lin[0], vmax=lim_lin[1], origin='lower', interpolation='bilinear')
        else:
            im1   = ax1.imshow(array, extent=[-M/2.0,M/2.0,-M/2.0,M/2.0],vmin=lim_lin[0], vmax=lim_lin[1], origin='lower', interpolation='bilinear')
        if colorbar==True:
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("right", size="7%", pad=0.05)
            if colorbar_ticks == 'auto': cbar1    = plt.colorbar(im1, cax=cax1)
            else: cbar1    = plt.colorbar(im1, cax=cax1, ticks=[0,0.2,0.4,0.6,0.8,1])
            cbar1.ax.tick_params(labelsize=font, color=cb_tick_color,width=1, direction='in')
        elif colorbar== 'top':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("top", size="7%", pad=0.05)
            if colorbar_ticks == 'auto': cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            else: cbar1    = plt.colorbar(im1, cax=cax1,orientation="horizontal", ticks=[0,0.2,0.4,0.6,0.8])
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('top')   
        elif colorbar== 'bottom':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("bottom", size="7%", pad=0.05)
            if colorbar_ticks == 'auto': cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            else: cbar1    = plt.colorbar(im1, cax=cax1,orientation="horizontal", ticks=[])
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('bottom')   
    elif scale == 'log':
        if type(lim_log) ==bool: 
            if flip_x == True: 
                array = np.fliplr(array)
                im1=ax1.imshow(array, extent=[M/2.0,-M/2.0,-M/2.0,M/2.0], norm=LogNorm()                                , origin='lower', interpolation='bilinear')
            else:
                im1=ax1.imshow(array, extent=[-M/2.0,M/2.0,-M/2.0,M/2.0], norm=LogNorm()                                , origin='lower', interpolation='bilinear')
        else: 
            if flip_x == True: 
                array = np.fliplr(array)                   
                im1=ax1.imshow(array, extent=[M/2.0,-M/2.0,-M/2.0,M/2.0], norm=LogNorm(vmin=lim_log[0], vmax=lim_log[1]), origin='lower', interpolation='bilinear')
            else:
                im1=ax1.imshow(array, extent=[-M/2.0,M/2.0,-M/2.0,M/2.0], norm=LogNorm(vmin=lim_log[0], vmax=lim_log[1]), origin='lower', interpolation='bilinear')
        if colorbar==True:
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("right", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1, cax=cax1)
            cbar1.ax.xaxis.set_ticks_position('top')
            cbar1.ax.tick_params(labelsize=font, color=cb_tick_color, direction='in')
        elif colorbar== 'top':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("top", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('top')
        elif colorbar== 'bottom':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("bottom", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('bottom')
    elif scale == 'semilog':
        array[np.where(array<=0.000001)] = 0.00001
        array=array**gamma
        if type(lim_log) ==bool: 
            if flip_x == True: 
                array = np.fliplr(array)
                im1=ax1.imshow(array, extent=[M/2.0,-M/2.0,-M/2.0,M/2.0], norm=LogNorm()                                , origin='lower', interpolation='bilinear')
            else:
                im1=ax1.imshow(array, extent=[-M/2.0,M/2.0,-M/2.0,M/2.0], norm=LogNorm()                                , origin='lower', interpolation='bilinear')
        else: 
            if flip_x == True: 
                array = np.fliplr(array)                   
                im1=ax1.imshow(array, extent=[M/2.0,-M/2.0,-M/2.0,M/2.0], norm=LogNorm(vmin=lim_log[0], vmax=lim_log[1]), origin='lower', interpolation='bilinear')
            else:
                im1=ax1.imshow(array, extent=[-M/2.0,M/2.0,-M/2.0,M/2.0], norm=LogNorm(vmin=lim_log[0], vmax=lim_log[1]), origin='lower', interpolation='bilinear')
        if colorbar==True:
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("right", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1, cax=cax1)
            cbar1.ax.xaxis.set_ticks_position('top')
            cbar1.ax.tick_params(labelsize=font, color=cb_tick_color, direction='in')
        elif colorbar== 'top':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("top", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('top')
        elif colorbar== 'bottom':
            divider1 = make_axes_locatable(ax1)
            cax1     = divider1.append_axes("bottom", size="7%", pad=0.05)
            cbar1    = plt.colorbar(im1,orientation="horizontal", cax=cax1)
            cbar1.ax.tick_params(labelsize=font)#, color='w',width=1.5, direction='in')
            cbar1.ax.xaxis.set_ticks_position('bottom')
            
    ax1.set_axis_off()
    if flip_x == False:
        if zoom == True: # flip_x = False, zoom=True
            ax1.set_xlim([-r0M*3, r0M*3])
            ax1.set_ylim([-r0M*3, r0M*3])
            ax1.set_xticks([-10,-5,0,5,10])
            ax1.set_yticks([-10,-5,0,5,10])
            if name != None: 
                temp = ax1.get_xlim()[0]
                ax1.text(0.9*temp,np.abs(0.8*temp), name, fontsize=label_font_sz, color=label_color) #makes the text label
        else:# flip_x = False, zoom=False
            ax1.set_yticks(ax1.get_xticks())
            ax1.set_ylim(ax1.get_xlim())
        if name !=None: 
            temp = ax1.get_xlim()[0]
            ax1.text(0.9*temp,np.abs(0.8*temp), name, fontsize=label_font_sz, color=label_color) #makes the text label
    elif zoom == True: # flip_x = True, zoom=True
        ax1.set_xlim([r0M*3, -r0M*3])
        ax1.set_ylim([-r0M*3, r0M*3])
        ax1.set_xticks([10,5,0,-5,-10])
        ax1.set_yticks([-10,-5,0,5,10])
        if name != None: 
            temp = ax1.get_xlim()[0]
            ax1.text(0.9*temp,np.abs(0.8*temp), name, fontsize=label_font_sz, color=label_color) #makes the text label
    elif zoom == False:# flip_x = True, zoom=False
        ax1.set_yticks(-1*ax1.get_xticks())
        temp = ax1.get_xlim()
        ax1.set_ylim(-1*temp[0], -1*temp[1])
        if name !=None: 
            temp = ax1.get_xlim()[0]
            ax1.text(0.9*temp,np.abs(0.8*temp), name, fontsize=label_font_sz, color=label_color) #makes the text label
    
    if x_label==True: ax1.set_xlabel('X ($GMc^{-2}$)',fontsize=font)
    if y_label==True: ax1.set_ylabel('Y ($GMc^{-2}$)',fontsize=font)
    if spin != None: 
        make_circle_kerr(ax1,spin,i=theta,circle_width=circle_width)
    if scale_bar == True:
        add_scale(ax1, padding=.07, font=10.56, end_factor=0.0)
    if no_border==False:
        if output != None: 
            fig.savefig(output, bbox_inches='tight')
            plt.close(fig)
    else:
        plt.axis('off')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        if output != None: 
            if (output.endswith(".png")):
                fig.savefig(output, bbox_inches='tight', pad_inches = -.05,dpi=400)
                plt.close(fig)
            else:
                fig.savefig(output, bbox_inches='tight', pad_inches = -.05)
                plt.close(fig)

    if make_fig == True:return(ax1)
    else: return(im1)
    
    
    
    
    
    
    