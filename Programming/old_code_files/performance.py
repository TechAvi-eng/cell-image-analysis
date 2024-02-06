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

wavelet_list = ['haar', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20']
psnr_list = []
ssim_list = []

for i in wavelet_list:
    wavelet = i

    coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

    coeffs_A = list(coeffs)

    for i in range(1, n+1):
        coeffs_A[i] = tuple(np.zeros_like(element) for element in coeffs[i])

    reconstructed_image_A = pywt.waverec2(tuple(coeffs_A), wavelet)
    reconstructed_image_A = np.uint8(reconstructed_image_A)

    psnr_A = peak_signal_noise_ratio(imArrayG, reconstructed_image_A)
    ssim_A = structural_similarity(imArrayG, reconstructed_image_A)

    psnr_list.append(psnr_A)
    ssim_list.append(ssim_A)
    
# Produce a graph of the PSNR against the wavelet used
plt.figure(figsize=(10,7))
plt.plot(wavelet_list, psnr_list)
plt.xlabel('Wavelet Function')
plt.ylabel('Peak Signal-To-Noise Ratio (dB)')
plt.title('Peak Signal-To-Noise Ratio (dB) against Wavelet Function')
plt.show()

# Produce a graph of the SSIM against the wavelet used
plt.figure(figsize=(10,7))
plt.plot(wavelet_list, ssim_list)
plt.xlabel('Wavelet Function')
plt.ylabel('Structural Similarity (SSIM) Index')
plt.title('Structural Similarity (SSIM) Index against Wavelet Function')
plt.show()



