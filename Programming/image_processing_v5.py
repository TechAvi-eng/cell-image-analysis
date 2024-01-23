import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

folder_path = 'Programming/images'
image_name = '1_00001_prepared'

image_path = os.path.join(folder_path, image_name) + '.png' # image path 

img = cv2.imread(image_path)

# Convert to 8-bit grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.uint8(img)

# Thresholding
_, thresh = cv2.threshold(img,103, 255, cv2.THRESH_BINARY)

# Invert the image
thresh = cv2.bitwise_not(thresh)

# Display thresholded image
cv2.imshow('Thresholded', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Morphological operations
kernel = np.ones((5, 5), np.uint8)
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Contour detection
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Minimum contour area threshold
min_contour_area = 1000  # Adjust this value based on your requirements

# Filter contours based on area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Draw contours on original image
result = img.copy()
cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()