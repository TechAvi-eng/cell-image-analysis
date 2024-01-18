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

imArrayG = np.float32(imArrayG)
#imArrayG /= 255;

n = 1
wavelet = 'haar'

coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

arr, coeff_slices = pywt.coeffs_to_array(coeffs)

cv2.imshow('Image', arr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # coeffs_list = list(coeffs)
# # coeffs_D = coeffs_list[0]
# # coeffs_D[0] *= 0;

# coeffs_D = coeffs[0]

# # imArray_D = pywt.waverec2(coeffs, 'haar');
# # imArray_D *= 255;
# # imArray_D = np.uint8(imArray_D)

# cv2.imshow('Detail Coefficients Reconstructed', coeffs_D) # display the image
# cv2.waitKey(0)
# cv2.destroyAllWindows()