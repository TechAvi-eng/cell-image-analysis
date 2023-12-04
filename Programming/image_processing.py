import os
from PIL import Image, ImageDraw
import pywt # PyWavelets package
from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as np
from scipy import ndimage
import cv2
import math


def import_image(folder_path, image_name):
    # Obtaining the image path
    image_path = os.path.join(folder_path, image_name)

    # Open the image
    img = Image.open(image_path)
    # img.show()

    # Return the image path
    return img


def gray_conversion(img):
    # Converting the image to grayscale
    img_gray = img.convert('L')

    # Display grayscale image
    # img_gray.show()

    # Return the grayscale image
    return img_gray


def wavelet_coefficients(img_gray):
    
    # Wavlet decomposition (2 level)
    n = 2 # level of decomposition
    w = 'db1' # mother wavelet type

    coeffs = pywt.wavedec2(img_gray, wavelet=w, level=n) # perform wavelet decompositionq


    # normalize each coefficient array
    coeffs[0] = coeffs[0]/np.abs(coeffs[0]).max() # normalize the lowpass component (approximation coefficients)
    
    for detail_level in range(n): # loop through the detail levels
        coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level + 1]] # normalize the detail coefficients

    arr, coeff_slices = pywt.coeffs_to_array(coeffs) # convert the wavelet coefficients into an array 'arr', and provides information about coefficients arrangement 'coeff_slices', which is later used to reconstruct the image

    # Plot the wavelet coefficients
    plt.imshow(arr, cmap='gray_r', vmin = -0.25, vmax = 0.75) # display the 2D array 'arr' as an image, with the colormap 'gray_r' (reverse gray), and the range of values to be displayed is between -0.25 and 0.75
    plt.rcParams["figure.figsize"] = (8,8) # set the size of the figure
    plt.show() # display the figure
    

def wavelet_decomposition_removal(img_gray):

    # Wavlet decomposition (2 level)
    n = 2 # level of decomposition
    w = 'haar' # mother wavelet type 

    coeffs = pywt.wavedec2(img_gray, wavelet=w, level=n)

    coeff_arr, coeffs_slices = pywt.coeffs_to_array(coeffs) # convert the wavelet coefficients into an array 'coeff_arr', and provides information about coefficients arrangement 'coeff_slices', which is later used to reconstruct the image

    Csort = np.sort(np.abs(coeff_arr.reshape(-1))) # sort the wavelet coefficients in ascending order

    # for keep in (0.2, 0.1, 0.05, 0.01, 0.005): # top 10%, 5%, 1%, 0.5% of wavelet coefficients are kept
    #     thresh = Csort[int(np.floor((1-keep)*len(Csort)))] # calculate the threshold value
    #     ind = np.abs(coeff_arr) > thresh # find the indices of the wavelet coefficients that are greater than the threshold
    #     Cfilt = coeff_arr * ind # filter out the wavelet coefficients that are less than the threshold

    #     coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format='wavedec2') # convert the filtered wavelet coefficients back to the format used by the pywt package

    #     # Perform inverse wavelet transform to reconstruct the image
    #     Arecon = pywt.waverec2(coeffs_filt, wavelet=w)

    #     # Plot the reconstructed image
    #     plt.figure()
    #     plt.imshow(Arecon.astype('uint8'), cmap='gray') # display the reconstructed image
    #     plt.title('Keep = ' + str(keep*100) + '% of coefficients')
    #     plt.show()

    # After deciding on the percentage of coefficients to keep, store the new image
    keep = 0.05
    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format='wavedec2')

    Arecon = pywt.waverec2(coeffs_filt, wavelet=w)

    # Store the reconstructed image as img_post_wavelet
    img_post_wavelet = Image.fromarray(Arecon.astype('uint8'))

    # Display the reconstructed image
    #img_post_wavelet.show()

    return img_post_wavelet


def bw_conversion(img):
    # Display the image
    #img.show()

    # Convert the image to black and white binary image using a threshold value of 128
    img_bw = img.point(lambda x: 0 if x < 123 else 255, '1')

    # Display the black and white binary image
    #img_bw.show()

    return img_bw
    

def cell_boundary_reconstruction(img):
    dilation_radius = 4
    erosion_radius = 10

    bw_img_array = np.array(img)

    # Perform dilation to fill gaps and make boundaries smoother
    dilated_img = ndimage.binary_dilation(bw_img_array, structure=np.ones((dilation_radius, dilation_radius)))

    # Perform erosion to refine the boundaries and make them more circular
    eroded_img = ndimage.binary_erosion(dilated_img, structure=np.ones((erosion_radius, erosion_radius)))

    # Create a new PIL Image from the NumPy array
    reconstructed_img = Image.fromarray(eroded_img.astype(np.uint8) * 255)

    # Display the original, black and white, and reconstructed images if needed
    #reconstructed_img.show(title='Reconstructed Image')

    # Save the image in testing_images folder as 'reconstructured_image.png'
    reconstructed_img.save('Programming/testing_images/reconstructed_image.png')

    return reconstructed_img


def cell_counting(img):
    img_array = np.array(img)

    # Find contours in the smoothed image
    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the image for visualization purposes
    img_with_contours = img.copy()

    # Draw the contours on the image
    cv2.drawContours(np.array(img_with_contours), contours, -1, (0, 255, 0), 2)

    # Create a copy of the image for visualization purposes
    img_with_circles = img.copy()

    # Draw circles around the identified cells
    draw = ImageDraw.Draw(img_with_circles)
    for contour in contours:
        # Fit a circle around each contour
        center, radius = cv2.minEnclosingCircle(contour)
        center = tuple(map(int, center))
        radius = int(radius)
        
        # Draw the circle on the image
        draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius])

    # Display the original image and image with circles if needed
    img_with_circles.show(title='Image with Circles')

    # Count the number of detected contours (cells)
    num_cells = len(contours)
    print(f"Number of cells: {num_cells}")

    return num_cells

def cell_counting_2(img):

    img_array = np.array(img)

    cnts = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    minimum_area = 1
    average_cell_area = 650
    connected_cell_area = 650
    cells = 0
    
    image_original = cv2.imread("Programming/testing_images/3_00013.png")
    original = image_original.copy()

    for c in cnts:
        area = cv2.contourArea(c)
        if area > minimum_area:
            cv2.drawContours(original, [c], -1, (36,255,12), 2)
            if area > connected_cell_area:
                cells += math.ceil(area / average_cell_area)
            else:
                cells += 1
    print('Cells: {}'.format(cells))
    
    # Show the image with the identified cells
    cv2.imshow('original', original)
    cv2.waitKey()


def main():
    # Set the path to the folder containing the image
    folder_path = 'Programming/testing_images'

    # Set the name of the image file
    image_name = '3_00013.png'

    # Import the image and open it
    img = import_image(folder_path, image_name)
    
    # Convert the image to grayscale
    img_gray = gray_conversion(img)

    # Perform wavelet decomposition - normalisation of coefficients
    # wavelet_coefficients(img_gray)

    # Perform wavelet decomposition - removing insignificant coefficients
    img_post_wavelet = wavelet_decomposition_removal(img_gray)

    # Display the reconstructed image
    bw_converted = bw_conversion(img_post_wavelet)

    # Reconstructing image boundaries
    img_reconstructed = cell_boundary_reconstruction(bw_converted)

    # Counting the number of cells
    #cell_count = cell_counting(img_reconstructed)

    # Counting the number of cells
    cell_counting_2(img_reconstructed)

if __name__ == "__main__":
    main()