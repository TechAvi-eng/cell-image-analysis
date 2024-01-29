import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


folder_path = 'Programming/images'
image_name = '1_00001_prepared'

image_path = os.path.join(folder_path, image_name) + '.png' # image path 

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.uint8(img)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

kernel = np.ones((5, 5), np.uint8) 
dilation = cv2.dilate(img, kernel, iterations = 1) 
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 
  
# Adaptive thresholding on mean and gaussian filter 
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)

# Otsu's thresholding 
ret4, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
  
# Initialize the list 
Cell_count, x_count, y_count = [], [], [] 
  
# read original image, to display the circle and center detection   
#display = cv2.imread("D:/Projects / ImageProcessing / DA1 / sample1 / cellOrig.png") 

folder_path_org = 'Programming/images'
image_name_org = '1_00001'

image_path_org = os.path.join(folder_path_org, image_name_org) + '.png' # image path 
display = cv2.imread(image_path_org)

# convert to 8 bit image
image = display
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.uint8(image)

# display the original image
# cv2.imshow('original image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# hough transform with modified circular parameters 
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 20,  
                           param1 = 50, param2 = 28, minRadius = 1, maxRadius = 20) 
  
# circle detection and labeling using hough transformation  
if circles is not None: 
        # convert the (x, y) coordinates and radius of the circles to integers 
        circles = np.round(circles[0, :]).astype("int") 
  
        # loop over the (x, y) coordinates and radius of the circles 
        for (x, y, r) in circles: 
  
                cv2.circle(display, (x, y), r, (0, 255, 0), 2) 
                cv2.rectangle(display, (x - 2, y - 2),  
                              (x + 2, y + 2), (0, 128, 255), -1) 
                Cell_count.append(r) 
                x_count.append(x) 
                y_count.append(y) 
        # show the output image 
        cv2.imshow("gray", display) 
        cv2.waitKey(0) 
  
# display the count of white blood cells  
print(len(Cell_count)) 
# Total number of radius 
print(Cell_count)  
# X co-ordinate of circle 
print(x_count)      
# Y co-ordinate of circle 
print(y_count)     
