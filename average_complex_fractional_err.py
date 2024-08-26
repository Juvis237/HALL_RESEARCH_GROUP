import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams       # import to change plot parameters
from scipy import stats
import scipy
import sys
sys.path.append('/groups/dpsaltis/staging/lia/master_code')
from my_func import * # my module with some helpful functions
# from my_plot import * # my module with some plotting functions
from my_pca import *
from circle_stats import*


M_FoV       = 64.
pad         = 8
padBW       = 4
mass        = 6.5*10**9
distance    = 16800.
x           = 512
uvpix       = distance/(M_FoV*mass*pad)*2.09*10**7 



def complex_fractional_error(FT_orig, FT_new):
	# both input arrays should be complex
	return(np.sqrt(np.abs((FT_orig-FT_new)*(np.conj(FT_orig)-np.conj(FT_new))))/np.abs(FT_orig))

def complex_error(FT_orig, FT_new):
	# both input arrays should be complex
	return((np.sqrt(np.abs((FT_orig-FT_new)*(np.conj(FT_orig)-np.conj(FT_new)))))/(np.max(np.abs(FT_orig))))

def VA_fractional_error(FT_orig, FT_new):
	# both input arrays should be complex
	return((np.abs(np.abs(FT_orig) - np.abs(FT_new)))/(np.abs(FT_orig)))

def VA_error(FT_orig, FT_new):
	# both input arrays should be complex
	return((np.abs(np.abs(FT_orig) - np.abs(FT_new)))/(np.max(np.abs(FT_orig))))

def VP_error(FT_orig, FT_new):
	# both input arrays should be complex
	a = np.angle(FT_orig) - np.angle(FT_new)
	# the line below ensures that it takes into account the directionality of angles
	a = np.abs((a + np.pi) % (2.*np.pi) - np.pi)
	return(a)

def circle_average(array, start, end, interval, uvpix, directional=False):
	radii    = np.arange(start, end, interval)
	xx       = np.shape(array)[0]
	u_array  = ((np.arange(xx)-(xx/2))*uvpix).repeat(xx).reshape(xx,xx)
	v_array  = u_array.T

	distanceGL = np.sqrt(u_array*u_array+v_array*v_array)

	averages = np.zeros(np.shape(radii)[0]-1)
	for i in range(0, np.shape(radii)[0]-1):
		all_in_bound = np.where((distanceGL>radii[i]) & (distanceGL<radii[i+1]))
		if directional == True:
			averages[i] = circlemean(array[all_in_bound])
		else:
			averages[i] = np.average(array[all_in_bound])
	return(averages)


# comps       = open_hdf5('/xdisk/chanc/lia/gray/data/PCA_comps_huge_cropped_norm_filt15.h5')

# snaps_total = 1024
# start       = 0
# end         = 15
# interval    = .1
# axis        = np.arange(start, end, interval)
# axis        = axis[:-1]+(interval/2)
# mask_butter = mask_butterworth_filt(radius=71, n=2,x=x*padBW)

# for B_FIELD in ["MAD"]:
# 	for n in ["1e6"]:
# 		for M in ["6.5e9"]:
# 			for R in ["20","80","1"]:
# 				FILE = "/xdisk/chanc/lia/gray/data/a9"+B_FIELD+"_NE"+n+"_M"+M+"_R"+R

# 				cube          = open_hdf5(FILE+'_1.3mm.h5')
# 				# cube_filtered = open_hdf5(FILE+'filtered_norm.h5')

# 				results     = np.zeros((((snaps_total, np.shape(axis)[0], 8, 5))))
# 				# VP_err_cube = np.zeros(((snaps_total, x, x, 8)))

# 				for which in range(0,snaps_total):
# 					image    = cube[which]
# 					image    = image/np.sum(image)*6000
# 					# filtered = cube_filtered[which]
# 					filtered = filter_image_padded(image, mask_butter, pad=padBW)
# 					filtered    = filtered/np.sum(filtered)*6000

# 					imageFT    = myfft(image,    -1, pad=pad)
# 					filteredFT = myfft(filtered, -1, pad=pad)

# 					filt_complex_err  = complex_fractional_error(imageFT, filteredFT)
# 					fractional_VA_err = VA_fractional_error(     imageFT, filteredFT)
# 					VP_err            = VP_error(                imageFT, filteredFT)
# 					filt_comp_err_abs = complex_error(           imageFT, filteredFT)
# 					VA_err            = VA_error(                imageFT, filteredFT)

# 					# VP_err_cube[which, :, :, 0] = VP_err

# 					averagesCE        = circle_average( filt_complex_err, start, end, interval, uvpix)
# 					averagesVAE       = circle_average(fractional_VA_err, start, end, interval, uvpix)
# 					averagesVPE       = circle_average(           VP_err, start, end, interval, uvpix, directional=True)
# 					averagesCEabs     = circle_average( filt_comp_err_abs,start, end, interval, uvpix)
# 					averagesVAEabs    = circle_average(           VA_err, start, end, interval, uvpix)

# 					results[which, :, 0, 0] = averagesCE
# 					results[which, :, 0, 1] = averagesVAE
# 					results[which, :, 0, 2] = averagesVPE
# 					results[which, :, 0, 3] = averagesCEabs
# 					results[which, :, 0, 4] = averagesVAEabs


# 					k=1
# 					for num in [5,10,15, 20,25,30,35]:
# 						amps       = get_amps(comps[0:num,:,:], image)
# 						amplitudes = amps.repeat(x*x).reshape(((num,x,x)))
# 						recon      = np.sum(comps[:num,:,:]*amplitudes, axis=0)
# 						recon      = recon/np.sum(recon)*6000
# 						reconFT    = myfft(recon,    -1, pad=pad)

# 						test_complex_err  = complex_fractional_error(imageFT, reconFT)
# 						test_VA_err       = VA_fractional_error(     imageFT, reconFT)
# 						test_VP_err       = VP_error(                imageFT, reconFT)
# 						test_comp_err_abs = complex_error(           imageFT, reconFT)
# 						test_VA_err_abs   = VA_error(                imageFT, reconFT)

# 						# VP_err_cube[which, :,:, k] = test_VP_err

# 						averagesCE     = circle_average( test_complex_err, start, end, interval, uvpix)
# 						averagesVAE    = circle_average(      test_VA_err, start, end, interval, uvpix)
# 						averagesVPE    = circle_average(      test_VP_err, start, end, interval, uvpix, directional=True)
# 						averagesCEabs  = circle_average(test_comp_err_abs, start, end, interval, uvpix)
# 						averagesVAEabs = circle_average(  test_VA_err_abs, start, end, interval, uvpix)

# 						results[which, :, k, 0] = averagesCE
# 						results[which, :, k, 1] = averagesVAE
# 						results[which, :, k, 2] = averagesVPE
# 						results[which, :, k, 3] = averagesCEabs
# 						results[which, :, k, 4] = averagesVAEabs


# 						k+=1
# 				save_hdf5(FILE+'average_error_complex_VA_VP_15_5_norm.h5', 'complex error', results)

				# mean_VP_err_map = np.zeros(((x,x,8)))
				# rms_VP_err_map  = np.zeros(((x,x,8)))
				# for i in range(0,8):
				# 	mean_VP_err_map[:,:,i] = circlemean(VP_err_cube[:,:,:,i])
				# 	rms_VP_err_map[ :,:,i] = circlerms( VP_err_cube[:,:,:,i])
				# save_hdf5(FILE+'meanVP_err_map_15_norm.h5',  'mean directional err', mean_VP_err_map)
				# save_hdf5(FILE+'rmsVP_err_map_15_norm.h5', 'rms of directional err',  rms_VP_err_map)

is_it_a7SANE = '_a7SANE'

# comps       = open_hdf5('/xdisk/chanc/lia/gray/data/PCA_comps_huge_cropped_norm_filt15'+is_it_a7SANE+'.h5')

snaps_total = 1024
start       = 0
end         = 15
interval    = .1
axis        = np.arange(start, end, interval)
axis        = axis[:-1]+(interval/2)
# mask_butter = mask_butterworth_filt(radius=71, n=2,x=x*padBW)


# for B_FIELD in ["SANE"]:
# 	for n in ["7.5e5"]:
# 		for M in ["6.5e9"]:
# 			for R in ["20","80","1"]:
# 				FILE = "/xdisk/chanc/lia/gray/data/a7"+B_FIELD+"/a7"+B_FIELD+"_NE"+n+"_M"+M+"_R"+R

# 				cube          = open_hdf5(FILE+'_i17_1.3mm.h5')
# 				# cube_filtered = open_hdf5(FILE+'filtered_norm.h5')

# 				results     = np.zeros((((snaps_total, np.shape(axis)[0], 2, 5))))
# 				# VP_err_cube = np.zeros(((snaps_total, x, x, 8)))

# 				for which in range(0,snaps_total):
# 					image    = cube[which]
# 					image    = image/np.sum(image)*6000
# 					# filtered = cube_filtered[which]
# 					filtered = filter_image_padded(image, mask_butter, pad=padBW)
# 					filtered    = filtered/np.sum(filtered)*6000

# 					imageFT    = myfft(image,    -1, pad=pad)
# 					filteredFT = myfft(filtered, -1, pad=pad)

# 					filt_complex_err  = complex_fractional_error(imageFT, filteredFT)
# 					fractional_VA_err = VA_fractional_error(     imageFT, filteredFT)
# 					VP_err            = VP_error(                imageFT, filteredFT)
# 					filt_comp_err_abs = complex_error(           imageFT, filteredFT)
# 					VA_err            = VA_error(                imageFT, filteredFT)

# 					# VP_err_cube[which, :, :, 0] = VP_err

# 					averagesCE        = circle_average( filt_complex_err, start, end, interval, uvpix)
# 					averagesVAE       = circle_average(fractional_VA_err, start, end, interval, uvpix)
# 					averagesVPE       = circle_average(           VP_err, start, end, interval, uvpix, directional=True)
# 					averagesCEabs     = circle_average( filt_comp_err_abs,start, end, interval, uvpix)
# 					averagesVAEabs    = circle_average(           VA_err, start, end, interval, uvpix)

# 					results[which, :, 0, 0] = averagesCE
# 					results[which, :, 0, 1] = averagesVAE
# 					results[which, :, 0, 2] = averagesVPE
# 					results[which, :, 0, 3] = averagesCEabs
# 					results[which, :, 0, 4] = averagesVAEabs


# 					k=1
# 					for num in [20]:
# 						amps       = get_amps(comps[0:num,:,:], image)
# 						amplitudes = amps.repeat(x*x).reshape(((num,x,x)))
# 						recon      = np.sum(comps[:num,:,:]*amplitudes, axis=0)
# 						recon      = recon/np.sum(recon)*6000
# 						reconFT    = myfft(recon,    -1, pad=pad)

# 						test_complex_err  = complex_fractional_error(imageFT, reconFT)
# 						test_VA_err       = VA_fractional_error(     imageFT, reconFT)
# 						test_VP_err       = VP_error(                imageFT, reconFT)
# 						test_comp_err_abs = complex_error(           imageFT, reconFT)
# 						test_VA_err_abs   = VA_error(                imageFT, reconFT)

# 						# VP_err_cube[which, :,:, k] = test_VP_err

# 						averagesCE     = circle_average( test_complex_err, start, end, interval, uvpix)
# 						averagesVAE    = circle_average(      test_VA_err, start, end, interval, uvpix)
# 						averagesVPE    = circle_average(      test_VP_err, start, end, interval, uvpix, directional=True)
# 						averagesCEabs  = circle_average(test_comp_err_abs, start, end, interval, uvpix)
# 						averagesVAEabs = circle_average(  test_VA_err_abs, start, end, interval, uvpix)

# 						results[which, :, k, 0] = averagesCE
# 						results[which, :, k, 1] = averagesVAE
# 						results[which, :, k, 2] = averagesVPE
# 						results[which, :, k, 3] = averagesCEabs
# 						results[which, :, k, 4] = averagesVAEabs


# 						k+=1
# 				save_hdf5(FILE+'average_error_complex_VA_VP_15_5_norm'+is_it_a7SANE+'.h5', 'complex error', results)
# 				print('saved', FILE+'average_error_complex_VA_VP_15_5_norm'+is_it_a7SANE+'.h5')

results     = np.zeros((15, np.shape(axis)[0]))

k=0
for B_FIELD in ["SANE"]:
	for n in ["1e5", "2.5e5", "5e5", "7.5e5", "1e6"]:
		for M in ["6.5e9"]:
			for R in ["20","80","1"]:
				FILE = "/xdisk/chanc/lia/gray/data/a7"+B_FIELD+"/a7"+B_FIELD+"_NE"+n+"_M"+M+"_R"+R
				temp = open_hdf5(FILE+'average_error_complex_VA_VP_15_5_norm'+is_it_a7SANE+'.h5')
				results[k,:] = np.mean(temp[:,:,1,3], axis=0)
				k+=1

mean_all = np.mean(results, axis=0)
to_save = np.zeros((2,np.shape(axis)[0]))
to_save[0,:] = axis
to_save[1,:] = mean_all
np.savetxt("theory_error20_F_a7SANE.dat", to_save.T, delimiter=',', newline='\n',fmt='%f')


