import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Prepared Image
folder_path = 'Programming/images'
image_name = '1_00001_prepared'
image_path = os.path.join(folder_path, image_name) + '.png' # image path 
img = cv2.imread(image_path)

# Convert to 8-bit grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.uint8(img)

# Grayscale Image
gray_image = '1_00001_gray'
gray_image_path = os.path.join(folder_path, gray_image) + '.png'
gray_image = cv2.imread(gray_image_path)


### NEW FUNCTION ###
# Simple Thresholding 
_, thresh = cv2.threshold(img, 103, 255, cv2.THRESH_BINARY_INV) # Pixel value > 103 set to 255, then inverted as cv2.findContours() requires white objects on black background

# Display thresholded image
cv2.imshow('Thresholded', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Morphological operations
kernel_open = np.ones((25, 25), np.uint8) # kernel with all ones
kernel_dilation = np.ones((16, 16), np.uint8)
kernel_close = np.ones((25, 25), np.uint8)

morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open) # Removes small white regions (noise in background)
morphed = cv2.dilate(morphed, kernel_dilation, iterations = 1) # Increases white regions (joins broken cells)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel_close) # Removes small black holes (noise in cells)


# Contour detection
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL retrieves only the extreme outer contours, cv2.CHAIN_APPROX_SIMPLE compresses the contour

# Draw contours on original grayscale image
result = gray_image.copy()

cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Contours on Grayscale Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Minimum contour area threshold - removes small contours
min_contour_area = 3000

# Filter contours based on area
filtered_contours = []
for contour in contours:
    if cv2.contourArea(contour) > min_contour_area:
        filtered_contours.append(contour)

# Draw contours on original image
result_filtered = gray_image.copy()

# Draw contours and number them
for i in range(len(filtered_contours)):
    cv2.drawContours(result_filtered, filtered_contours, i, (0, 255, 0), 2)
    cv2.putText(result_filtered, str(i+1), tuple(filtered_contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    
    # Output the contours area
    print('Contour ' + str(i+1) + ' area = ' + str(cv2.contourArea(filtered_contours[i])))

print('Number of Cells Found: ' + str(len(filtered_contours)))

# Display the results
cv2.imshow('Filtered Contours on Grayscale Image', result_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Potential further steps:
# Watershed segmentation to remove background
# Adaptive/Otsu thresholding to binarise image
# Obtain characteristics of each contour including area, perimeter, average intensity, etc.