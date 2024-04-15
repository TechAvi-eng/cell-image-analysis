import time
import cv2
import os
import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# This program is designed to count the number of cells in an image using the Discrete Wavelet Transform (DWT) for image denoising
# Detailed setup instructions can be found in the README.md file
# Note: Appropriate images must be placed in the Dataset folder for the program to run correctly

def display_image(image_name, image_array):
    """
    Function to display the image
    Parameters:
        image_name (str): image name
        image_array (array): image array
    """
    cv2.imshow(image_name, image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_info(image_name, imArray):
    """
    Function to display the image information
    Parameters:
        image_name (str): image name
        imArray (array): image array
    """
    # Display the colour information of the image
    print(f'\nImage information for {image_name}:')
    print(f'Mean: {imArray.mean():.2f}')
    print(f'Minimum: {imArray.min()}')
    print(f'Maximum: {imArray.max()}')

    # Display the histogram for number of pixels against pixel intensity, include the image_name
    plt.hist(imArray.flatten(), bins=100)
    
    # Histogram Formatting
    plt.title(f'Histogram of Pixel Intensity\n {image_name}', fontname='Times New Roman', fontsize=12)
    plt.xlabel('Pixel intensity', fontname='Times New Roman', fontsize=11)
    plt.ylabel('Number of pixels', fontname='Times New Roman', fontsize=11)
    plt.xticks(fontname='Times New Roman', fontsize=10)
    plt.yticks(fontname='Times New Roman', fontsize=10)
    plt.gcf().set_size_inches(8.38/2.54, 6/2.54)
    plt.gcf().set_dpi(600)
    plt.tight_layout()
    plt.show()
    

def image_import(folder_path, image_name):
    """
    Imports image and returns image array
    Parameters:
        folder_path (str): folder path of image
        image_name (str): image name
    Returns:
        imArray (array): image array
    """

    image_path = os.path.join(folder_path, image_name) + '.png' # Image path
    imArray = cv2.imread(image_path) # Read image
    
    print(f'Bit depth: {imArray.dtype}') # Display bit depth of image

    display_image('Original Image', imArray) # Display the original image

    image_info('Original Image', imArray) # Display the image information

    return imArray


def gray_conversion(imArray):
    """
    Convert image to grayscale and convert to 8-bit integer
    Parameters:
        imArray (array): original image array
    Returns:
        imArrayG (array): grayscale 8-bit image array
    """
    imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY) # Conversion to grayscale

    imArrayG = np.uint8(imArrayG) # Conversion to 8-bit integer

    display_image('Gray 8-bit Integer Image', imArrayG)
    
    image_info('Before Pre-Processing', imArrayG)

    print("Dynamic Range Before Processing: ", imArrayG.max()-imArrayG.min())

    return imArrayG


def dynamic_range(imArrayG):
    """
    Algorithm to adjust the brightness and contrast of the image
    Parameters:
        imArrayG (array): grayscale 8-bit image array pre adjustment
    Returns:
        adjusted_image (array): grayscale 8-bit image array post adjustment
    """
    minimum_value = np.min(imArrayG)
    maximum_value = np.max(imArrayG)

    brightness = minimum_value
    contrast_ratio = 255 / (maximum_value - minimum_value)
    img_contrast = imArrayG * (contrast_ratio)

    minimum_value = np.min(img_contrast)
    img_contrast = img_contrast - brightness
    adjusted_image = np.clip(img_contrast, 0, 255)
    adjusted_image = np.uint8(adjusted_image)

    display_image('Adjusted Image (Brightness and Contrast)', adjusted_image)

    image_info('After Pre-Processing', adjusted_image)

    print("Dynamic Range After Processing: ", adjusted_image.max()-adjusted_image.min())

    return adjusted_image


def discrete_wavelet_transform(imArrayG, n, wavelet):
    """
    Complete DWT based multiresolution for denoising
    Parameters:
        imArrayG (array): grayscale 8-bit image array
        n (int): number of levels of decomposition
        wavelet (str): wavelet function
    Returns:
        coeffs (array): coefficients array from DWT
    """
    coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

    return coeffs


def coeffs_map(coeffs):
    """
    Produce coefficient map for visualisation only
    Parameters:
        coeffs (array): coefficients array
    """
    coeffs[0] = coeffs[0]/np.abs(coeffs[0]).max() # normalise the approximation coefficients

    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    display_image('Coefficients Map', arr)    


def reconstrucuted_images(coeffs, n, wavelet):
    """
    Reconstruct images using only approximation and detail coefficients, producing two images
    Parameters:
        imArrayG (array): grayscale 8-bit image array from DWT
        n (int): number of levels of decomposition
        wavelet (str): wavelet function
    Returns:
        reconstructed_image_A (array): reconstructed image using only approximation coefficients
    """
    # Setting Detail Coefficients to Zero
    coeffs_A = list(coeffs)
    for i in range(1, n+1):
        coeffs_A[i] = tuple(np.zeros_like(element) for element in coeffs[i]) # set detail coefficients to zero

    reconstructed_image_A = pywt.waverec2(tuple(coeffs_A), wavelet) # reconstruct image using inverse DWT

    reconstructed_image_A = np.uint8(reconstructed_image_A) # convert to 8-bit integer image

    display_image('Approx Coefficients Only Reconstructed Image', reconstructed_image_A)

    image_info('Approx Coefficients Only Reconstructed Image', reconstructed_image_A)
    

    # Setting Detail Coefficients to Zero (for Visualisation only)
    coeffs_D = list(coeffs)
    for i in range(0, 1):
        coeffs_D[i] = tuple(np.zeros_like(element) for element in coeffs[i]) # set approximation coefficients to zero

    reconstructed_image_D = pywt.waverec2(tuple(coeffs_D), wavelet) # reconstruct image using inverse DWT

    reconstructed_image_D = np.uint8(reconstructed_image_D) # convert to 8-bit integer image

    display_image('Detail Coefficients Only Reconstructed Image', reconstructed_image_D)

    return reconstructed_image_A


def binary_thresholding(prepared_image):
    """
    Binary thresholding using simple thresholding method
    Parameters:
        prepared_image (array): image array post inverse DWT reconstruction
    Returns:
        thresh (array): thresholded image array
    """
    threshold = prepared_image.mean() - 1/2 * prepared_image.std() - 2 # Calculate threshold value
    
    # If pixel value > threshold the  set to 0. Inverted as cv2.findContours() requires white objects on black background
    _, thresh = cv2.threshold(prepared_image, threshold, 255, cv2.THRESH_BINARY_INV)

    print('Simple Threshold Value: ' + str(threshold))

    display_image('Binary Thresholded Image', thresh)

    return thresh


def cell_identification(binary_image, imArrayG):
    """
    Morphological operations and contour detection (cell identification)
    Note: The morphological operations must be adjusted based on the cell size
    Parameters:
        binary_image (array): binary thresholded image array
        imArrayG (array): grayscale 8-bit image array
    Returns:
        result_filtered (array): image array with filtered contours
        filtered_contours (list): list of filtered contours (can be used for future analysis)
    """
    # Morphological kernel sizes (with all ones)
    kernel_open = np.ones((25, 25), np.uint8)
    kernel_dilation = np.ones((25, 25), np.uint8)
    kernel_close = np.ones((25, 25), np.uint8)

    morphed = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open) # Removes small white regions (noise in background)
    morphed = cv2.dilate(morphed, kernel_dilation, iterations = 1) # Increases white regions (joins broken cells)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel_close) # Removes small black holes (noise in cells)
    
    display_image('Morphed Image', morphed)
    
    # Contour detection: cv2.RETR_EXTERNAL retrieves only the extreme outer contours, cv2.CHAIN_APPROX_SIMPLE compresses the contour
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original colour image
    result = imArrayG.copy()
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    result = cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    display_image('Contours on Grayscale Image', result)


    # Minimum contour area threshold - removes contours smaller than the threshold
    min_contour_area = 0

    # Filter contours based on area threshold
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            filtered_contours.append(contour)
            print('Cell Area: ' + str(cv2.contourArea(contour))) # Output the area of the cells

    print('Average Cell Area: ' + str(np.mean([cv2.contourArea(contour) for contour in filtered_contours])))
    
    # Retrieve the original colour image so that the filtered contours can be drawn on it
    result_filtered = imArrayG.copy()
    result_filtered = cv2.cvtColor(result_filtered, cv2.COLOR_GRAY2BGR)

    # Draw filtered contours and number them
    for i in range(len(filtered_contours)):
        cv2.drawContours(result_filtered, filtered_contours, i, (0, 255, 0), 2)
        cv2.putText(result_filtered, str(i+1), tuple(filtered_contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)    

    print('Number of Cells Found: ' + str(len(filtered_contours)))

    display_image('Filtered Contours on Grayscale Image', result_filtered)

    return result_filtered, filtered_contours


def main():
    # Image path and name for which the cell counting is to be performed
    folder_path = 'Dataset'
    image_name = '1_00001'

    # Import image and return image array
    imArray = image_import(folder_path, image_name)

    # Convert image to grayscale and convert to 8-bit integer
    imArrayG = gray_conversion(imArray)

    # Adjust brightness and contrast to improve the dynamic range
    adjusted_image = dynamic_range(imArrayG)

    # DWT parameters (number of levels and wavelet type)
    n = 4
    wavelet = 'db12'
    
    # DWT decomposition 
    coeffs = discrete_wavelet_transform(adjusted_image, n, wavelet)

    # Produce coefficient map for visualisation only
    # coeffs_map(coeffs)

    # Reconstruct images with only approximation and detail coefficients respectively
    prepared_image = reconstrucuted_images(coeffs, n, wavelet)

    # Binary thresholding using simple thresholding method
    binary_image_simple = binary_thresholding(prepared_image)

    # Morphological operations and contour detection (cell identification)
    result_filtered, filtered_contours = cell_identification(binary_image_simple, imArrayG)
    

if __name__ == "__main__":
    main()