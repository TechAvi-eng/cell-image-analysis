import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


folder_path = 'Programming/images'
image_name = '1_00001_prepared'

image_path = os.path.join(folder_path, image_name) + '.png' # image path 

image = cv2.imread(image_path)

# Set our filtering parameters 
# Initialize parameter setting using cv2.SimpleBlobDetector 
params = cv2.SimpleBlobDetector_Params() 
  
# Set Area filtering parameters 
params.filterByArea = True
params.minArea = 1
  
# Set Circularity filtering parameters 
params.filterByCircularity = True 
params.minCircularity = 0.01
params.maxCircularity = 1
  
# Set Convexity filtering parameters 
params.filterByConvexity = True
params.minConvexity = 0.01
      
# Set inertia filtering parameters 
params.filterByInertia = True
params.minInertiaRatio >= 0
params.maxInertiaRatio <= 1
  
# Create a detector with the parameters 
detector = cv2.SimpleBlobDetector_create(params) 
      
# Detect blobs 
keypoints = detector.detect(image) 
  
# Draw blobs on our image as red circles 
blank = np.zeros((1, 1))  
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), 
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
  
number_of_blobs = len(keypoints) 
text = "Number of Circular Blobs: " + str(len(keypoints)) 
cv2.putText(blobs, text, (20, 550), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
  
# Show blobs 
cv2.imshow("Filtering Circular Blobs Only", blobs) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 