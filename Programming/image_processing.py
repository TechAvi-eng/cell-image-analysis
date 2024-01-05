import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw
import pywt # PyWavelets package
import os
import cv2

def import_image(folder_path, image_name):
    # Joining the image path
    image_path = os.path.join(folder_path, image_name)

    image = cv2.imread(image_path) # read the image
    
    cv2.imshow('Image', image) # display the image
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    # Obtain the bit depth of the image
    bit_depth = image.dtype
    print('Bit depth of the raw image: ', bit_depth)

    return image # return the image (array)


def gray_conversion(image):
    # Converting the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Converting the image to 8-bit unsigned integer
    img_gray = cv2.convertScaleAbs(img_gray)

    # Display grayscale image
    cv2.imshow('Grayscale Image', img_gray)

    # Save grayscale image
    output_path = 'Programming/images/gray_image.png'
    cv2.imwrite(output_path, img_gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Obtain the bit depth of the grayscale image
    bit_depth = image.dtype
    print('Bit depth of the grayscale image: ', bit_depth)

    # Return the grayscale image
    return img_gray


def image_info(image):
    # Display the colour information of the image
    print(f'Mean: {image.mean():.2f}')
    print(f'Minimum: {image.min()}')
    print(f'Maximum: {image.max()}')

    plt.hist(image.flatten(), bins=100, range=(0, 255))
    plt.show()


def image_normalisation(image):
    # Normalise the image
    image_normalised = image/255

    # Display the colour information of the image
    print(f'Mean: {image_normalised.mean():.2f}')
    print(f'Minimum: {image_normalised.min()}')
    print(f'Maximum: {image_normalised.max()}')

    plt.hist(image_normalised.flatten(), bins=100, range=(0, 1))
    plt.show()


def wavelet_coefficients(img_gray):
    
    # Wavlet decomposition (2 level)
    n = 2 # level of decomposition
    w = 'haar' # mother wavelet type

    coeffs = pywt.wavedec2(img_gray, wavelet=w, level=n) # perform wavelet decompositionq


    # normalize each coefficient array
    coeffs[0] = coeffs[0]/np.abs(coeffs[0]).max() # normalize the lowpass component (approximation coefficients)
    
    for detail_level in range(n): # loop through the detail levels
        coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level + 1]] # normalize the detail coefficients

    arr, coeff_slices = pywt.coeffs_to_array(coeffs) # convert the wavelet coefficients into an array 'arr', and provides information about coefficients arrangement 'coeff_slices', which is later used to reconstruct the image

    # Plot the wavelet coefficients
    plt.imshow(arr, cmap='gray', vmin = -0.25, vmax = 0.75) # display the 2D array 'arr' as an image, with the colormap 'gray_r' (reverse gray), and the range of values to be displayed is between -0.25 and 0.75
    plt.rcParams["figure.figsize"] = (8,8) # set the size of the figure
    plt.show() # display the figure
    

def wavelet_decomposition_removal(img_gray):

    # Wavlet decomposition (2 level)
    n = 4 # level of decomposition
    w = 'db18' # mother wavelet type 

    coeffs = pywt.wavedec2(img_gray, wavelet=w, level=n)

    coeff_arr, coeffs_slices = pywt.coeffs_to_array(coeffs) # convert the wavelet coefficients into an array 'coeff_arr', and provides information about coefficients arrangement 'coeff_slices', which is later used to reconstruct the image

    Csort = np.sort(np.abs(coeff_arr.reshape(-1))) # sort the wavelet coefficients in ascending order

    # for keep in (0.2, 0.1, 0.05, 0.01, 0.001): # top 10%, 5%, 1%, 0.5% of wavelet coefficients are kept
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
    keep = 0.01
    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind # filter out the wavelet coefficients that are less than the threshold

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format='wavedec2') # convert the filtered wavelet coefficients back to the format used by the pywt package

    Arecon = pywt.waverec2(coeffs_filt, wavelet=w) # perform inverse wavelet transform to reconstruct the image

    # Store the reconstructed image as img_post_wavelet
    img_post_wavelet = Image.fromarray(Arecon.astype('uint8'))

    # Display the reconstructed image
    # img_post_wavelet.show()

    return img_post_wavelet


def bw_conversion(img):
    # Display the image
    img.show()

    # Convert the image to black and white binary image using a threshold value of 128
    img_bw = img.point(lambda x: 0 if x < 123 else 255, '1')

    # Display the black and white binary image
    img_bw.show()

    return img_bw


#def cell_counting(img_bw):
    """
    This function counts the number of cells in the image, by adding ellipses around all circles detected in the image, even if they have gaps. This function then shows the image with the ellipses drawn around the circles.
    """


def main():
    # Set the path to the folder containing the image
    folder_path = 'Programming/images'

    # Set the name of the image file
    image_name = '1_00001.png'

    # Import the image
    image = import_image(folder_path, image_name)
    
    # Convert the image to grayscale
    img_gray = gray_conversion(image)

    # Display the image information
    image_info(img_gray)

    # Normalise the image pixel density from 0 to 1 and display information
    image_normalisation(img_gray)

    # Perform wavelet decomposition - normalisation of coefficients
    #wavelet_coefficients(img_gray)

    # Perform wavelet decomposition - removing insignificant coefficients
    #img_post_wavelet = wavelet_decomposition_removal(img_gray)

    # Display the reconstructed image
    #img_bw = bw_conversion(img_post_wavelet)

    # Counting the number of cells
    #cell_count = cell_counting(img_bw)

if __name__ == "__main__":
    main()