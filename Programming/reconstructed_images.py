import numpy as np
import pywt
import pywt.data
import matplotlib.image as mping
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image


folder_path = 'Programming/images'
image_name = '1_00001'
image_path = os.path.join(folder_path, image_name) + '.png' # image path 
imArray = cv2.imread(image_path)
cv2.imshow('Image', imArray)
cv2.waitKey(0) 
cv2.destroyAllWindows()


imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)

n = 4
wavelet = 'haar'

coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

# SETTING DETAIL COEFFICIENTS TO ZERO
coeffs_A = list(coeffs)
for i in range(1, n+1):
    coeffs_A[i] = tuple(np.zeros_like(element) for element in coeffs[i])

reconstructed_image = pywt.waverec2(tuple(coeffs_A), 'haar')

cv2.imshow('Approx Coefficients Reconstructed Image', reconstructed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# SETTING APPROXIMATION COEFFICIENTS TO ZERO

coeffs_D = list(coeffs)
for i in range(0, 1):
    coeffs_D[i] = tuple(np.zeros_like(element) for element in coeffs[i])

reconstructed_image = pywt.waverec2(tuple(coeffs_D), 'haar')

cv2.imshow('Detail Coefficients Reconstructed Image', reconstructed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()