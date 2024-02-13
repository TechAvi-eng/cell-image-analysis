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
    plt.savefig('Programming/edited_images' + image_name + '_histogram.png')
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

    #image_info('Original_Image', imArray)


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
    
    #image_info('Gray_8_bit_Image', imArrayG)

    output_path = 'Programming/edited_images/' + image_name + '_gray.png'
    cv2.imwrite(output_path, imArrayG)

    return imArrayG


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

    image_info('Approx_Coefficients_Image', reconstructed_image_A)
    
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


def DWT_performance(imArrayG):
    """
    Evaluate the performance before and after DWT applied, using PSNR and SSIM performance metrics
    Parameters:
        imArrayG (array): grayscale 8-bit image array
    """

    wavelet_list = pywt.wavelist(kind='discrete')
    
    # Create empty lists to store PSNR and SSIM values, with columns representing the wavelet function used and rows representing the number of levels used
    psnr_list = np.zeros((len(wavelet_list), 3))
    ssim_list = np.zeros((len(wavelet_list), 3))

    psnr_highest_name = ''
    ssim_highest_name = ''

    psnr_highest = 0
    ssim_highest = 0

    for n in range(3, 6):
        for i in wavelet_list:
            wavelet = i

            coeffs = pywt.wavedec2(imArrayG, wavelet, level=n)

            coeffs_A = list(coeffs)

            for i in range(1, n+1):
                coeffs_A[i] = tuple(np.zeros_like(element) for element in coeffs[i])

            reconstructed_image_A = pywt.waverec2(tuple(coeffs_A), wavelet)
            reconstructed_image_A = np.uint8(reconstructed_image_A)

            psnr_A = peak_signal_noise_ratio(imArrayG, reconstructed_image_A)
            ssim_A = structural_similarity(imArrayG, reconstructed_image_A)

            if psnr_A > psnr_highest:
                psnr_highest = psnr_A
                psnr_highest_name = wavelet + ' at ' + str(n) + ' levels'
            
            if ssim_A > ssim_highest:
                ssim_highest = ssim_A
                ssim_highest_name = wavelet + ' at ' + str(n) + ' levels'

            # Store PSNR and SSIM values in the list at column = wavelet and row = n
            psnr_list[wavelet_list.index(wavelet)][n-3] = psnr_A
            ssim_list[wavelet_list.index(wavelet)][n-3] = ssim_A
            
            print("Completed " + str(wavelet) + " at " + str(n) + " levels")

    # Output the wavelet function gives the highest PSNR and SSIM
    print("Highest PSNR: " + psnr_highest_name + " with PSNR of " + str(psnr_highest))
    print("Highest SSIM: " + ssim_highest_name + " with SSIM of " + str(ssim_highest))

    # Produce a graph of the PSNR against the wavelet used
    plt.figure(figsize=(10,7))
    plt.plot(wavelet_list, psnr_list)
    plt.xticks(rotation=90)
    plt.xlabel('Wavelet Function')
    plt.ylabel('Peak Signal-To-Noise Ratio (dB)')
    plt.title('Peak Signal-To-Noise Ratio (dB) against Wavelet Function')
    plt.legend(['3 levels', '4 levels', '5 levels'])
    plt.show()

    # Produce a graph of the SSIM against the wavelet used
    plt.figure(figsize=(10,7))
    plt.plot(wavelet_list, ssim_list)
    plt.xticks(rotation=90)
    plt.xlabel('Wavelet Function')
    plt.ylabel('Structural Similarity (SSIM) Index')
    plt.title('Structural Similarity (SSIM) Index against Wavelet Function')
    plt.legend(['3 levels', '4 levels', '5 levels'])
    plt.show()
    

def binary_thresholding(prepared_image):
    """
    Binary thresholding using simple thresholding method
    Parameters:
        prepared_image (array): image array
    Returns:
        thresh (array): thresholded image array
    """
    #threshold = prepared_image.mean() - 1/2 * prepared_image.std() - 2 # threshold value
    threshold = 48
    _, thresh = cv2.threshold(prepared_image, threshold, 255, cv2.THRESH_BINARY) # Pixel value > threshold set to 255, then inverted as cv2.findContours() requires white objects on black background

    print('Simple Threshold Value: ' + str(threshold))

    # Display thresholded image
    display_image('Binary Thresholded Image', thresh)

    return thresh


def otsu_thresholding(prepared_image):
    """
    Otsu's thresholding
    Parameters:
        prepared_image (array): image array
    Returns:
        otsu (array): thresholded image array
    """
    # Output the mean and standard deviation of the image
    print('Mean: ' + str(prepared_image.mean()))
    print('Standard Deviation: ' + str(prepared_image.std()))

    threshold, otsu = cv2.threshold(prepared_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Output the threshold value
    print('Otsu Threshold Value: ' + str(threshold))

    # Display thresholded image
    display_image('Otsu Thresholded Image', otsu)

    return otsu


def cell_identification(binary_image, imArrayG, image_name):
    # Morphological operations
    # kernel_open = np.ones((25, 25), np.uint8) # kernel with all ones
    # kernel_dilation = np.ones((16, 16), np.uint8)
    # kernel_close = np.ones((25, 25), np.uint8)

    # morphed = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open) # Removes small white regions (noise in background)
    # morphed = cv2.dilate(morphed, kernel_dilation, iterations = 1) # Increases white regions (joins broken cells)
    # morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel_close) # Removes small black holes (noise in cells)
    
    morphed = binary_image
    display_image('Morphed Image', morphed)
    
    # Contour detection
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL retrieves only the extreme outer contours, cv2.CHAIN_APPROX_SIMPLE compresses the contour

    # Draw contours on original grayscale image
    result = imArrayG.copy()
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) # ***Only to display contour in colour

    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    #display_image('Contours on Grayscale Image', result)


    # Minimum contour area threshold - removes small contours
    min_contour_area = 00

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
        #cv2.putText(result_filtered, str(i+1), tuple(filtered_contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)    

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
    image_name = 'fig10'

    # Import image
    imArray = image_import(folder_path, image_name)

    # Convert image to grayscale and convert to 8-bit integer
    imArrayG = gray_conversion(imArray, image_name)

    n = 4
    wavelet = 'coif17'
    
    # Complete DWT
    coeffs = discrete_wavelet_transform(imArrayG, n, wavelet)

    # Reconstruct images with only approximation and detail coefficients respectively
    prepared_image = reconstrucuted_images(coeffs, n, wavelet, image_name)

    """
    # Evaluate the performance before and after DWT applied
    DWT_performance(imArrayG)
    """

    # Binary thresholding
    binary_image_simple = binary_thresholding(prepared_image)


    # Otsu's thresholding
    # binary_image_otsu = otsu_thresholding(prepared_image)

    # Morphological operations and contour detection (cell identification)
    result_filtered, filtered_contours = cell_identification(binary_image_simple, imArrayG, image_name)
    #result_filtered, filtered_contours = cell_identification(binary_image_otsu, imArrayG, image_name)
    
if __name__ == "__main__":
    main()


