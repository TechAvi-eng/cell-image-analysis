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
# cv2.imshow('Image', imArray)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)

levels = [1, 2]

wavelet_list = ['haar', 'db1', 'db2']
psnr_list = [[0]*len(wavelet_list)]*(len(levels))
print(psnr_list)
ssim_list = [[0]*len(wavelet_list)]*(len(levels))
print(ssim_list)

for count in range(len(levels)):
    n = levels[count]
    for i in range(len(wavelet_list)):
        wavelet = wavelet_list[i]
        coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

        coeffs_A = list(coeffs)

        for j in range(1, n+1):
            coeffs_A[j] = tuple(np.zeros_like(element) for element in coeffs[j])

        reconstructed_image_A = pywt.waverec2(tuple(coeffs_A), wavelet)
        reconstructed_image_A = np.uint8(reconstructed_image_A)

        psnr_A = peak_signal_noise_ratio(imArrayG, reconstructed_image_A)
        ssim_A = structural_similarity(imArrayG, reconstructed_image_A)

        psnr_list[count][i] = float(psnr_A)
        ssim_list[count][i] = float(ssim_A)

# Produce the graphs of the PSNR against the wavelet used
# Plot the graphs of the PSNR against the wavelet used, for each level on the same axis
for i in range(1, len(psnr_list)):
    plt.plot(wavelet_list, psnr_list[i])
plt.xlabel('Wavelet Type')
plt.ylabel('PSNR')
plt.title('PSNR of Approx Coefficients Images')
plt.show()

# Produce a graph of the SSIM against the wavelet used
for i in range(1, len(ssim_list)):
    plt.plot(wavelet_list, ssim_list[i])
plt.xlabel('Wavelet Type')
plt.ylabel('SSIM')
plt.title('SSIM of Approx Coefficients Images')
plt.show()


