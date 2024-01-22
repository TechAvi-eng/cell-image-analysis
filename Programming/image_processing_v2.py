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

    display_image('Original Image', imArray)

    return imArray


def gray_conversion(imArray):
    """
    Convert image to grayscale and convert to 8-bit integer
    Parameters:
        imArray (array): image array
    Returns:
        imArrayG (array): grayscale 8-bit image array
    """
    imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    imArrayG = np.uint8(imArrayG) # convert to 8-bit integer

    display_image('Gray 8-bit Integer Image', imArrayG)

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


def coeffs_map(coeffs):
    """
    Produce coefficient map
    Parameters:
        coeffs (array): coefficients array
    """
    coeffs[0] = coeffs[0]/np.abs(coeffs[0]).max() # normalise the approximation coefficients

    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    display_image('Coefficients Map', arr)


def thresholding(coeffs, wavelet):
    """
    Thresholding retaining only set% of coefficients and reconstruct image
    Parameters:
        coeffs (array): coefficients array
        wavelet (str): wavelet type
    """
    coeffs[0] = coeffs[0]/np.abs(coeffs[0]).max()

    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    Csort = np.sort(np.abs(arr.reshape(-1))) # sort the coefficients from smallest to largest

    for keep in (0.5, 0.1, 0.05, 0.01, 0.005):
        thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
        ind = np.abs(arr) > thresh
        Cfilt = arr * ind

        coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

        Arecon = pywt.waverec2(coeffs_filt, wavelet) # reconstruct image

        display_image('Threshold ' + str(keep), Arecon)


def denoising(imArrayG, n, wavelet_function):
    """
    Denoising using BayesShrink and VisuShrink, and display SNR and PSNR after each method
    Parameters:
        imArrayG (array): grayscale 8-bit image array
        n (int): number of levels
        wavelet_function (str): wavelet type
    """
    sigma_est = estimate_sigma(imArrayG, average_sigmas=True)

    img_bayes = denoise_wavelet(imArrayG, method = 'BayesShrink', mode = 'soft', wavelet_levels = n, wavelet = wavelet_function, rescale_sigma = True)
    img_visushrink = denoise_wavelet(imArrayG, method = 'VisuShrink', mode = 'soft', wavelet_levels = n, wavelet = wavelet_function, sigma = sigma_est/3, rescale_sigma = True)

    psnr_bayes = peak_signal_noise_ratio(imArrayG, img_bayes)
    psnr_visushrink = peak_signal_noise_ratio(imArrayG, img_visushrink)

    display_image('Image After Bayes', img_bayes)
    display_image('Image After Visushrink', img_visushrink)

    print('PSNR of Bayes: ', psnr_bayes)
    print('PSNR of Visushrink: ', psnr_visushrink)


def reconstrucuted_images(coeffs, n, wavelet):
    """
    Reconstruct images using only approximation and detail coefficients
    Parameters:
        imArrayG (array): grayscale 8-bit image array
        n (int): number of levels
        wavelet (str): wavelet type
    """
    # SETTING DETAIL COEFFICIENTS TO ZERO
    coeffs_A = list(coeffs)
    for i in range(1, n+1):
        coeffs_A[i] = tuple(np.zeros_like(element) for element in coeffs[i])

    reconstructed_image_A = pywt.waverec2(tuple(coeffs_A), wavelet)
    reconstructed_image_A = np.uint8(reconstructed_image_A)

    display_image('Approx Coefficients Only Reconstructed Image', reconstructed_image_A)

    # SETTING APPROXIMATION COEFFICIENTS TO ZERO
    coeffs_D = list(coeffs)
    for i in range(0, 1):
        coeffs_D[i] = tuple(np.zeros_like(element) for element in coeffs[i])

    reconstructed_image_D = pywt.waverec2(tuple(coeffs_D), wavelet)
    reconstructed_image_D = np.uint8(reconstructed_image_D)

    display_image('Detail Coefficients Only Reconstructed Image', reconstructed_image_D.astype(np.uint8))


def performance(imArrayG):
    """
    Evaluate the performance before and after DWT applied, using PSNR and SSIM performance metrics
    Parameters:
        imArrayG (array): grayscale 8-bit image array
    """
    n = 4
    wavelet_list = ['haar', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12',
                     'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20']
    psnr_list = []
    ssim_list = []

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

        psnr_list.append(psnr_A)
        ssim_list.append(ssim_A)
        
    # Produce a graph of the PSNR against the wavelet used
    plt.figure(figsize=(10,7))
    plt.plot(wavelet_list, psnr_list)
    plt.xlabel('Wavelet Function')
    plt.ylabel('Peak Signal-To-Noise Ratio (dB)')
    plt.title('Peak Signal-To-Noise Ratio (dB) against Wavelet Function')
    plt.show()

    # Produce a graph of the SSIM against the wavelet used
    plt.figure(figsize=(10,7))
    plt.plot(wavelet_list, ssim_list)
    plt.xlabel('Wavelet Function')
    plt.ylabel('Structural Similarity (SSIM) Index')
    plt.title('Structural Similarity (SSIM) Index against Wavelet Function')
    plt.show()


def main():
    folder_path = 'Programming/images'
    image_name = '1_00001'

    # Import image
    imArray = image_import(folder_path, image_name)

    # Convert image to grayscale and convert to 8-bit integer
    imArrayG = gray_conversion(imArray)

    n = 4
    wavelet = 'db17'
    
    # Complete DWT
    coeffs = discrete_wavelet_transform(imArrayG, n, wavelet)

    """
    # Produce coefficient map
    coeffs_map(coeffs)
    
    # Thresholding retaining only set% of coefficients
    thresholding(coeffs, wavelet)

    # Denoising using BayesShrink and VisuShrink
    denoising(imArrayG, n, wavelet)
    """

    # Reconstruct images with only approximation and detail coefficients
    reconstrucuted_images(coeffs, n, wavelet)

    # Evaluate the performance before and after DWT applied
    performance(imArrayG)

if __name__ == "__main__":
    main()


