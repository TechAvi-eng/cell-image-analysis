import numpy as np
import pywt
import pywt.data
import matplotlib.image as mping
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import peak_signal_noise_ratio
import skimage.io


folder_path = 'Programming/images'
image_name = '1_00001'
image_path = os.path.join(folder_path, image_name) + '.png' # image path 
imArray = cv2.imread(image_path)
cv2.imshow('Image', imArray)
cv2.waitKey(0) 
cv2.destroyAllWindows()

imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)

sigma_est = estimate_sigma(imArrayG, average_sigmas=True)

wavelet_function = 'haar'
n = 5

img_bayes = denoise_wavelet(imArrayG, method = 'BayesShrink', mode = 'soft', wavelet_levels = n, wavelet = wavelet_function, rescale_sigma = True)
img_visushrink = denoise_wavelet(imArrayG, method = 'VisuShrink', mode = 'soft', wavelet_levels = n, wavelet = wavelet_function, sigma = sigma_est/3, rescale_sigma = True)

mean, std_dev = cv2.meanStdDev(imArrayG)
original_snr = 20 * np.log10(mean[0] / std_dev[0])

psnr_bayes = peak_signal_noise_ratio(imArrayG, img_bayes)
psnr_visushrink = peak_signal_noise_ratio(imArrayG, img_visushrink)

cv2.imshow('Original Grayscale Image', imArrayG)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Image After Bayes', img_bayes)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Image After Visushrink', img_visushrink)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('SNR of Original: ', original_snr)
print('PSNR of Bayes: ', psnr_bayes)
print('PSNR of Visushrink: ', psnr_visushrink)