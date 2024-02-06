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
_, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# Invert the image
thresh = cv2.bitwise_not(thresh)

# Morphological operations
kernel = np.ones((5, 5), np.uint8)
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Contour detection
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Minimum contour area threshold
min_contour_area = 1000  # Adjust this value based on your requirements

# Filter contours based on area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Function to check if two contours are close
def are_contours_close(cnt1, cnt2, distance_threshold=200):
    moments1 = cv2.moments(cnt1)
    moments2 = cv2.moments(cnt2)

    cx1, cy1 = int(moments1['m10'] / moments1['m00']), int(moments1['m01'] / moments1['m00'])
    cx2, cy2 = int(moments2['m10'] / moments2['m00']), int(moments2['m01'] / moments2['m00'])

    distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return distance < distance_threshold

# Combine close contours
combined_contours = []

for i in range(len(filtered_contours)):
    merged = False
    for j in range(len(combined_contours)):
        if are_contours_close(filtered_contours[i], combined_contours[j]):
            combined_contours[j] = np.vstack([filtered_contours[i], combined_contours[j]])
            merged = True
            break

    if not merged:
        combined_contours.append(filtered_contours[i])

# Draw contours on original image
result = img.copy()
cv2.drawContours(result, combined_contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()