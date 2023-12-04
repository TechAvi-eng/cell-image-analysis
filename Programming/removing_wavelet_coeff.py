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

coeff_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

Csort = np.sort(np.abs(coeff_arr.reshape(-1))) # sort the wavelet coefficients in ascending order

for keep in (0.1, 0.05, 0.01, 0.005): # top 10%, 5%, 1%, 0.5% of wavelet coefficients are kept
    thresh = Csort[int(np.floor((1-keep)*len(Csort)))] # calculate the threshold value
    ind = np.abs(coeff_arr) > thresh # find the indices of the wavelet coefficients that are greater than the threshold
    Cfilt = coeff_arr * ind # filter out the wavelet coefficients that are less than the threshold

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format='wavedec2') # convert the filtered wavelet coefficients back to the format used by the pywt package

    # Plot reconstruction
    Arecon = pywt.waverec2(coeffs_filt, wavelet=w) # reconstruct the image from the filtered wavelet coefficients
    plt.figure()
    plt.imshow(Arecon.astype('uint8'), cmap='gray') # display the reconstructed image
    plt.axis('off')
    plt.rcParams["figure.figsize"] = (8,8) # set the size of the figure
    plt.title('Keep = ' + str(keep*100) + '% of coefficients')
