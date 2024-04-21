import time
import cv2
import os
import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# This program evaluates the performance of Discrete Wavelet Transform (DWT) functions for image denoising
# The performance is evaluated using the following metrics:
# Peak Signal-To-Noise Ratio (PSNR), Structural Similarity (SSIM) and computational time

def image_import(folder_path, image_name):
    """
    Imports image and returns image array for the image which will be used for denoising evaluation
    Parameters:
        folder_path (str): folder path of image
        image_name (str): image name
    Returns:
        imArray (array): image array
    """

    image_path = os.path.join(folder_path, image_name) + '.png' # Image path
    imArray = cv2.imread(image_path) # Read image
    
    print(f'Bit depth: {imArray.dtype}') # Display bit depth of image

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

    print("Dynamic Range After Processing: ", adjusted_image.max()-adjusted_image.min())

    return adjusted_image


def DWT_performance(imArrayG):
    """
    Evaluate the performance before and after DWT applied, using PSNR, SSIM and computing time performance metrics
    Parameters:
        imArrayG (array): grayscale 8-bit image array
    """
    # List of wavelet functions
    wavelet_list = pywt.wavelist(kind='discrete')
    
    # Empty lists to store PSNR, SSIM values, and computing time results
    psnr_list = np.zeros((len(wavelet_list)))
    ssim_list = np.zeros((len(wavelet_list)))
    comp_time_list = np.zeros((len(wavelet_list)))

    # Number of levels of DWT decomposition
    n = 4 

    for i in wavelet_list: 
        wavelet = i

        # Start the timer for computation time
        start = time.time() 

        # Complete DWT decomposition
        coeffs = pywt.wavedec2(imArrayG, wavelet, level=n) 
        coeffs_A = list(coeffs) 

        # Remove the high frequency coefficients
        for y in range(1, n+1):
            coeffs_A[y] = tuple(np.zeros_like(element) for element in coeffs[y])

        # Reconstruct the image using the inverse DWT function
        reconstructed_image_A = pywt.waverec2(tuple(coeffs_A), wavelet)
        reconstructed_image_A = np.uint8(reconstructed_image_A)

        # End timer and calculate computation time
        end = time.time()
        comp_time = end - start

        # Calculate PSNR
        psnr_A = peak_signal_noise_ratio(imArrayG, reconstructed_image_A)

        # Calculate SSIM
        ssim_A = structural_similarity(imArrayG, reconstructed_image_A)

        # Store PSNR, SSIM and computational time values in the predefined lists
        psnr_list[wavelet_list.index(wavelet)] = psnr_A
        ssim_list[wavelet_list.index(wavelet)] = ssim_A
        comp_time_list[wavelet_list.index(wavelet)] = comp_time

        # Output the progress 
        print("Completed " + str(wavelet) + " at " + str(n) + " levels")

    # Produce a graph of the PSNR against the wavelet used
    plt.figure(figsize=(10,7))
    plt.plot(wavelet_list, psnr_list)
    plt.xticks(rotation=90)
    plt.xlabel('Wavelet Function')
    plt.ylabel('Peak Signal-To-Noise Ratio (dB)')
    plt.title('Peak Signal-To-Noise Ratio (dB) against Wavelet Function')
    plt.show()

    # Produce a graph of the SSIM against the wavelet used
    plt.figure(figsize=(10,7))
    plt.plot(wavelet_list, ssim_list)
    plt.xticks(rotation=90)
    plt.xlabel('Wavelet Function')
    plt.ylabel('Structural Similarity (SSIM) Index')
    plt.title('Structural Similarity (SSIM) Index against Wavelet Function')
    plt.show()

    # Produce a graph of the computation time against the wavelet used
    plt.figure(figsize=(10,7))
    plt.plot(wavelet_list, comp_time_list)
    plt.xticks(rotation=90)
    plt.xlabel('Wavelet Function')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time (s) against Wavelet Function')
    plt.show()


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

    # Evaluate the performance before and after DWT applied
    DWT_performance(adjusted_image)


if __name__ == "__main__":
    main()