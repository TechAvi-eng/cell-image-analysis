import numpy as np
import pywt
import pywt.data
import matplotlib.image as mping
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import (peak_signal_noise_ratio, structural_similarity)

def display_image(image_name, image_array):
    """
    Display image
    Parameters:
        image_name (str): image name
        image_array (array): image array
    """
    cv2.imshow(image_name, image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_info(image_name, imArray):
    """
    Display image information
    Parameters:
        imArray (array): image array
    """
    # Display the colour information of the image
    print(f'\nImage information for {image_name}:')
    print(f'Mean: {imArray.mean():.2f}')
    print(f'Minimum: {imArray.min()}')
    print(f'Maximum: {imArray.max()}')

    # Display the histogram for number of pixels against pixel intensity
    plt.hist(imArray.flatten(), bins=100) #, range=(0, 255)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Number of pixels')
    plt.tight_layout()
    plt.savefig('Programming/histograms/' + image_name + '_histogram.png')
    plt.show()
    

def image_import(folder_path, image_name):
    """
    Import image and return image array
    Parameters:
        folder_path (str): folder path of image
        image_name (str): image name
    Returns:
        imArray (array): image array
    """

    image_path = os.path.join(folder_path, image_name) + '.png' # image path 
    imArray = cv2.imread(image_path)
    
    print(f'Bit depth: {imArray.dtype}')

    display_image('Original Image', imArray)

    # image_info('Original Image', imArray)


    return imArray


def gray_conversion(imArray, image_name):
    """
    Convert image to grayscale and convert to 8-bit integer
    Parameters:
        imArray (array): image array
        image_name (str): image name
    Returns:
        imArrayG (array): grayscale 8-bit image array
    """
    imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)

    imArrayG = np.uint8(imArrayG) # convert to 8-bit integer

    display_image('Gray 8-bit Integer Image', imArrayG)
    
    image_info('Gray_8_bit_Image', imArrayG)

    output_path = 'Programming/edited_images/' + image_name + '_gray.png'
    cv2.imwrite(output_path, imArrayG)

    return imArrayG


def adjust_brightness_contrast(imArrayG):
    """
    Adjust brightness and contrast
    Parameters:
        imArrayG (array): grayscale 8-bit image array
    Returns:
        adjusted_image (array): grayscale 8-bit image array
    """
    # Find the maximum pixel intensity in the image
    max_intensity = np.max(imArrayG)

    # Define the target maximum intensity
    # Determine the intensity with the most pixels
    hist = cv2.calcHist([imArrayG], [0], None, [256], [0, 256])
    peak_intensity = np.argmax(hist)
    print('Peak Intensity: ' + str(peak_intensity))
    std = imArrayG.std()
    print('Standard Deviation: ' + str(std))
    target_max_intensity = peak_intensity + 2 * std

    # Calculate the ratio to adjust brightness and contrast
    brightness_ratio = max_intensity / target_max_intensity
    print('Brightness Ratio: ' + str(brightness_ratio))
    contrast_ratio = 255 / max_intensity
    print('Contrast Ratio: ' + str(contrast_ratio))

    # Adjust brightness and contrast
    adjusted_image = cv2.convertScaleAbs(imArrayG, alpha=brightness_ratio, beta=0)
    adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=contrast_ratio, beta=0)

    # Display the adjusted image
    display_image('Adjusted Image (Brightness and Contrast)', adjusted_image)

    image_info('Adjusted_Image', adjusted_image)

    return adjusted_image


def discrete_wavelet_transform(imArrayG, n, wavelet):
    """
    Complete DWT
    Parameters:
        imArrayG (array): grayscale 8-bit image array
        n (int): number of levels
        wavelet (str): wavelet type
    Returns:
        coeffs (array): coefficients array from DWT
    """
    coeffs = pywt.wavedec2(imArrayG, wavelet, level=n) # complete DWT

    return coeffs


def reconstrucuted_images(coeffs, n, wavelet, image_name):
    """
    Reconstruct images using only approximation and detail coefficients
    Parameters:
        imArrayG (array): grayscale 8-bit image array
        n (int): number of levels
        wavelet (str): wavelet type
        image_name (str): image name
    Returns:
        reconstructed_image_A (array): reconstructed image using only approximation coefficients
    """
    # SETTING DETAIL COEFFICIENTS TO ZERO
    coeffs_A = list(coeffs)
    for i in range(1, n+1):
        coeffs_A[i] = tuple(np.zeros_like(element) for element in coeffs[i]) # set detail coefficients to zero

    reconstructed_image_A = pywt.waverec2(tuple(coeffs_A), wavelet) # reconstruct image using inverse DWT
    reconstructed_image_A = np.uint8(reconstructed_image_A) # convert to 8-bit integer image

    display_image('Approx Coefficients Only Reconstructed Image', reconstructed_image_A)

    # image_info('Approx_Coefficients_Image', reconstructed_image_A)
    
    output_path = 'Programming/edited_images/' + image_name + '_prepared.png'
    cv2.imwrite(output_path, reconstructed_image_A)

    # SETTING APPROXIMATION COEFFICIENTS TO ZERO
    coeffs_D = list(coeffs)
    for i in range(0, 1):
        coeffs_D[i] = tuple(np.zeros_like(element) for element in coeffs[i])

    reconstructed_image_D = pywt.waverec2(tuple(coeffs_D), wavelet)
    reconstructed_image_D = np.uint8(reconstructed_image_D)

    display_image('Detail Coefficients Only Reconstructed Image', reconstructed_image_D)

    return reconstructed_image_A
    

def binary_thresholding(prepared_image):
    """
    Binary thresholding using simple thresholding method
    Parameters:
        prepared_image (array): image array
    Returns:
        thresh (array): thresholded image array
    """
    threshold = prepared_image.mean() + 1 * prepared_image.std() # threshold value
    print('Mean: ' + str(prepared_image.mean()))
    print('Standard Deviation: ' + str(prepared_image.std()))
    # threshold = 37
    _, thresh = cv2.threshold(prepared_image, threshold, 255, cv2.THRESH_BINARY) # Pixel value > threshold set to 255, then inverted as cv2.findContours() requires white objects on black background

    print('Simple Threshold Value: ' + str(threshold))

    # Display thresholded image
    display_image('Binary Thresholded Image', thresh)

    # image_info('Binary_Thresholded_Image', thresh)

    return thresh


def cell_identification(binary_image, imArrayG, image_name):
    # Morphological operations
    kernel_open = np.ones((2, 2), np.uint8) # kernel with all ones
    kernel_dilation = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((2, 2), np.uint8)

    morphed = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open) # Removes small white regions (noise in background)
    morphed = cv2.dilate(morphed, kernel_dilation, iterations = 1) # Increases white regions (joins broken cells)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel_close) # Removes small black holes (noise in cells)
    
    morphed = binary_image
    display_image('Morphed Image', morphed)
    
    # Contour detection
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL retrieves only the extreme outer contours, cv2.CHAIN_APPROX_SIMPLE compresses the contour

    # Draw contours on original grayscale image
    result = imArrayG.copy()
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) # ***Only to display contour in colour

    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    display_image('Contours on Grayscale Image', result)


    # Minimum contour area threshold - removes small contours
    min_contour_area = 500

    # Filter contours based on area
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            filtered_contours.append(contour)

    # Draw contours on original grayscale image
    result_filtered = imArrayG.copy()
    result_filtered = cv2.cvtColor(result_filtered, cv2.COLOR_GRAY2BGR) # ***Only to display contour and label text in colour

    # Draw contours and number them
    for i in range(len(filtered_contours)):
        cv2.drawContours(result_filtered, filtered_contours, i, (0, 255, 0), 2)
        #cv2.putText(result_filtered, str(i+1), tuple(filtered_contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)    

        # Output the contours area
        # print('Contour ' + str(i+1) + ' area = ' + str(cv2.contourArea(filtered_contours[i])))

    print('Number of Cells Found: ' + str(len(filtered_contours)))

    display_image('Filtered Contours on Grayscale Image', result_filtered)

    # Save image
    output_path = 'Programming/edited_images/' + image_name + '_contours.png'
    cv2.imwrite(output_path, result_filtered)

    return result_filtered, filtered_contours



def main():
    folder_path = 'Programming/raw_images/'
    image_name = 'fig12b'

    # Import image
    imArray = image_import(folder_path, image_name)

    # Convert image to grayscale and convert to 8-bit integer
    imArrayG = gray_conversion(imArray, image_name)

    # Adjust brightness and contrast
    adjusted_image = adjust_brightness_contrast(imArrayG)

    n = 2
    wavelet = 'haar'
    
    # Complete DWT
    coeffs = discrete_wavelet_transform(adjusted_image, n, wavelet)

    # Reconstruct images with only approximation and detail coefficients respectively
    prepared_image = reconstrucuted_images(coeffs, n, wavelet, image_name)

    # Binary thresholding
    binary_image_simple = binary_thresholding(prepared_image)

    # Morphological operations and contour detection (cell identification)
    result_filtered, filtered_contours = cell_identification(binary_image_simple, imArrayG, image_name)
    
if __name__ == "__main__":
    main()

