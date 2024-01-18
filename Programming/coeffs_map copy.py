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

n = 2
wavelet = 'haar'

coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

#coeffs[0] /= np.abs(coeffs[0]).max()
# for detail_level in range(n):
#     coeffs[detail_level + 1] = [d / np.abs(d).max() for d in coeffs[detail_level + 1]]

arr, coeff_slices = pywt.coeffs_to_array(coeffs)

cv2.imshow('Coefficients Plot', arr)
cv2.waitKey(0)
cv2.destroyAllWindows()


coeffs_modified = list(coeffs)
for i in range(1, n+1):
    coeffs_modified[i] = tuple(np.zeros_like(c) for c in coeffs[i])

reconstructed_image = pywt.waverec2(tuple(coeffs_modified), 'haar')

cv2.imshow('Reconstructed Image', np.clip(reconstructed_image, 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

