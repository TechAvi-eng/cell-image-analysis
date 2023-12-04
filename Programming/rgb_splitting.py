import os
from PIL import Image
import pywt # PyWavelets package
from matplotlib import pyplot as plt
import numpy as np

# Set the path to the folder containing the image
folder_path = 'RPE_dataset/Images'

# Set the name of the image file
image_name = '2_00025.png'

# Obtaining the image path
image_1 = os.path.join(folder_path, image_name)
img = Image.open(image_1)

# Split the image into RGB channels
r, g, b = img.split()

import matplotlib.pyplot as plt

# Set the path to the folder containing the image
folder_path = 'RPE_dataset/Images'

# Set the name of the image file
image_name = '2_00025.png'

# Obtaining the image path
image_1 = os.path.join(folder_path, image_name)
img = Image.open(image_1)

# Split the image into RGB channels
r, g, b = img.split()

# Display the three separate color channels
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(r)
axs[0].set_title('Red Channel')
axs[1].imshow(g)
axs[1].set_title('Green Channel')
axs[2].imshow(b)
axs[2].set_title('Blue Channel')
plt.show()

# Convert each color channel to grayscale
r_gray = r.convert('L')
g_gray = g.convert('L')
b_gray = b.convert('L')

"""
# Wavlet decomposition (2 level)
n = 2 # specify the level of decomposition
w = 'db1' # specify the wavelet type - Daubechies 1 family (mother wavelet)


# Red channel
r_coeffs = pywt.wavedec2(r_gray, wavelet=w, level=n)

# normalize each coefficient array
r_coeffs[0] = r_coeffs[0]/np.abs(r_coeffs[0]).max() # normalize the lowpass component (approximation coefficients)
for detail_level in range(n): # loop through the detail levels
    r_coeffs[detail_level + 1] = [d/np.abs(d).max() for d in r_coeffs[detail_level + 1]] # normalize the detail coefficients

r_arr, r_coeff_slices = pywt.coeffs_to_array(r_coeffs) # convert the wavelet coefficients into an array 'arr', and provides information about coefficients arrangement 'coeff_slices', which is later used to reconstruct the image


# Green channel
g_coeffs = pywt.wavedec2(g_gray, wavelet=w, level=n)

# normalize each coefficient array
g_coeffs[0] = g_coeffs[0]/np.abs(g_coeffs[0]).max() # normalize the lowpass component (approximation coefficients)
for detail_level in range(n): # loop through the detail levels
    g_coeffs[detail_level + 1] = [d/np.abs(d).max() for d in g_coeffs[detail_level + 1]] # normalize the detail coefficients

g_arr, g_coeff_slices = pywt.coeffs_to_array(g_coeffs) # convert the wavelet coefficients into an array 'arr', and provides information about coefficients arrangement 'coeff_slices', which is later used to reconstruct the image


# Blue channel
b_coeffs = pywt.wavedec2(b_gray, wavelet=w, level=n)

# normalize each coefficient array
b_coeffs[0] = b_coeffs[0]/np.abs(b_coeffs[0]).max() # normalize the lowpass component (approximation coefficients)
for detail_level in range(n): # loop through the detail levels
    b_coeffs[detail_level + 1] = [d/np.abs(d).max() for d in b_coeffs[detail_level + 1]] # normalize the detail coefficients

b_arr, b_coeff_slices = pywt.coeffs_to_array(b_coeffs) # convert the wavelet coefficients into an array 'arr', and provides information about coefficients arrangement 'coeff_slices', which is later used to reconstruct the image


# Combine the RGB channels coefficients into a single array
arr = np.zeros((r_arr.shape[0], r_arr.shape[1], 3))
arr[:,:,0] = r_arr 
arr[:,:,1] = g_arr
arr[:,:,2] = b_arr

plt.rcParams["figure.figsize"] = (16,16) # set the default size of figures
plt.rcParams.update({'font.size': 18}) 

plt.imshow(arr, cmap='gray_r', vmin = -0.25, vmax = 0.75) # display the 2D array 'arr' as an image, with the colormap 'gray_r' (reverse gray), and the range of values to be displayed is between -0.25 and 0.75
plt.rcParams["figure.figsize"] = (8,8) # set the size of the figure
plt.show() # display the figure
"""