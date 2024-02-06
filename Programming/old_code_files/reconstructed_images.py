import numpy as np
import pywt
import pywt.data
import matplotlib.image as mping
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import (peak_signal_noise_ratio, structural_similarity)
import skimage.io


folder_path = 'Programming/images'
image_name = '1_00001'
image_path = os.path.join(folder_path, image_name) + '.png' # image path 
imArray = cv2.imread(image_path)
cv2.imshow('Image', imArray)
cv2.waitKey(0) 
cv2.destroyAllWindows()

imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)

n = 4
wavelet = 'db17'

coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

# SETTING DETAIL COEFFICIENTS TO ZERO
coeffs_A = list(coeffs)
for i in range(1, n+1):
    coeffs_A[i] = tuple(np.zeros_like(element) for element in coeffs[i])

reconstructed_image_A = pywt.waverec2(tuple(coeffs_A), wavelet)
reconstructed_image_A = np.uint8(reconstructed_image_A)

cv2.imshow('Approx Coefficients Reconstructed Image', reconstructed_image_A)
output_path = 'Programming/images/' + image_name + '_prepared.png'
cv2.imwrite(output_path, reconstructed_image_A)
cv2.waitKey(0)
cv2.destroyAllWindows()

# SETTING APPROXIMATION COEFFICIENTS TO ZERO
coeffs_D = list(coeffs)
for i in range(0, 1):
    coeffs_D[i] = tuple(np.zeros_like(element) for element in coeffs[i])

reconstructed_image_D = pywt.waverec2(tuple(coeffs_D), wavelet)
reconstructed_image_D = np.uint8(reconstructed_image_D)

cv2.imshow('Detail Coefficients Reconstructed Image', reconstructed_image_D.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use a for loop to calculate the PSNR and SSIM of the reconstrcuted_image_A for db1 through to db20, plotting the results on the same axis


# DENOISING PERFORMANCE METRICS
psnr_A = peak_signal_noise_ratio(imArrayG, reconstructed_image_A)
print('PSNR of Approx Coefficients: ', psnr_A)

ssim_A = structural_similarity(imArrayG, reconstructed_image_A)
print('SSIM of Approx Coefficients: ', ssim_A)
