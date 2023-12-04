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

# Converting the image to grayscale
img_gray = img.convert('L')

# Display original image
img.show()

# Display grayscale image
img_gray.show()

# Wavlet decomposition (2 level)
n = 2 # specify the level of decomposition
w = 'db1' # specify the wavelet type - Daubechies 1 family (mother wavelet)

coeffs = pywt.wavedec2(img_gray, wavelet=w, level=n)

# normalize each coefficient array
coeffs[0] = coeffs[0]/np.abs(coeffs[0]).max() # normalize the lowpass component (approximation coefficients)
for detail_level in range(n): # loop through the detail levels
    coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level + 1]] # normalize the detail coefficients

arr, coeff_slices = pywt.coeffs_to_array(coeffs) # convert the wavelet coefficients into an array 'arr', and provides information about coefficients arrangement 'coeff_slices', which is later used to reconstruct the image

# plt.rcParams["figure.figsize"] = (16,16) # set the default size of figures
# plt.rcParams.update({'font.size': 18}) 

plt.imshow(arr, cmap='gray_r', vmin = -0.25, vmax = 0.75) # display the 2D array 'arr' as an image, with the colormap 'gray_r' (reverse gray), and the range of values to be displayed is between -0.25 and 0.75
plt.rcParams["figure.figsize"] = (8,8) # set the size of the figure
# fig = plt.figure(figsize=(18, 16)) # create a new figure with the specified size
plt.show() # display the figure