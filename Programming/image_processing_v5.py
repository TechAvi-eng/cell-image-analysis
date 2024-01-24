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
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)




# Draw contours on original image and number them
result = img.copy()

for i in range(len(contours)):
    cv2.drawContours(result, contours, i, (0, 255, 0), 2)
    cv2.putText(result, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Minimum contour area threshold
min_contour_area = 4725  # Adjust this value based on your requirements

# Filter contours based on area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Draw contours on original image
result = img.copy()

for i in range(len(filtered_contours)):
    cv2.drawContours(result, filtered_contours, i, (0, 255, 0), 2)
    cv2.putText(result, str(i), tuple(filtered_contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

print('Number of Cells Found: ' + str(len(filtered_contours)))

# Display the results
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


for i in filtered_contours:
    epsilon = 0.1*cv2.arcLength(i,True)
    approx = cv2.approxPolyDP(i,epsilon,True)
    cv2.drawContours(result, approx, i, (0, 255, 0), 2)

cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()