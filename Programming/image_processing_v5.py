import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

folder_path = 'Programming/images'
image_name = '1_00001_prepared'
image_path = os.path.join(folder_path, image_name) + '.png' # image path 
img = cv2.imread(image_path)

# original_image_name = '1_00001'
# original_image_path = os.path.join(folder_path, original_image_name) + '.png'
# result = cv2.imread(original_image_path)
# Convert to 8-bit grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.uint8(img)


### NEW FUNCTION ###
# Thresholding 
_, thresh = cv2.threshold(img, 103, 255, cv2.THRESH_BINARY_INV) # Pixel value > 103 set to 255, then inverted as cv2.findContours() requires white objects on black background

# Display thresholded image
cv2.imshow('Thresholded', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Morphological operations
kernel = np.ones((15, 15), np.uint8) # 20x20 kernel with all ones, smaller the kernel size the more dilation (smooth curves)

morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) 
morphed = cv2.dilate(morphed,kernel,iterations = 1)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel) 

# Contour detection
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL retrieves only the extreme outer contours, cv2.CHAIN_APPROX_SIMPLE compresses the contour

# Draw contours on original image
result = img.copy()

cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Minimum contour area threshold
min_contour_area = 6000  # Adjust this value based on your requirements

# Filter contours based on area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Draw contours on original image
result = img.copy()

original_image_name = '1_00001_gray'
original_image_path = os.path.join(folder_path, original_image_name) + '.png'
result_2 = cv2.imread(original_image_path)

# Draw contours and number them
for i in range(len(filtered_contours)):
    cv2.drawContours(result_2, filtered_contours, i, (0, 255, 0), 2)
    cv2.putText(result_2, str(i), tuple(filtered_contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    # Output the contours area
    print('Contour ' + str(i) + ' area = ' + str(cv2.contourArea(filtered_contours[i])))

print('Number of Cells Found: ' + str(len(filtered_contours)))

# Display the results
cv2.imshow('Result', result_2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Now I want to obtain characteristics of each contour including area, perimeter, average intensity, etc.
#for i in range(len(filtered_contours)):


# result_2 = img.copy()
# for i in filtered_contours:
#     epsilon = 0.1*cv2.arcLength(i,True)
#     approx = cv2.approxPolyDP(i,epsilon,True)
#     cv2.drawContours(result_2, approx, i, (0, 255, 0), 2)

# cv2.imshow('Result', result_2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()