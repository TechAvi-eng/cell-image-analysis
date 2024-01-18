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
imArrayG = np.uint8(imArrayG)

n = 2
wavelet = 'haar'

coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

coeffs[0] = coeffs[0]/np.abs(coeffs[0]).max()

arr, coeff_slices = pywt.coeffs_to_array(coeffs)

Csort = np.sort(np.abs(arr.reshape(-1)))

for keep in (0.5, 0.1, 0.05, 0.01, 0.005):
    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(arr) > thresh
    Cfilt = arr * ind

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

    Arecon = pywt.waverec2(coeffs_filt, wavelet)
    # title of image is the keep value
    cv2.imshow('Threshold ' + str(keep), Arecon)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
