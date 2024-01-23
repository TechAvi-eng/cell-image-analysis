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
    
    #image_info('Gray 8-bit Integer Image', imArrayG)

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
    coeffs[0] = coeffs[0]/np.abs(coeffs[0]).max() # normalise the approximation coefficients (unsure why this is required normalising but the detail coefficients do not)

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

    image_info('Approx Coefficients Only Reconstructed Image', reconstructed_image_A)
    
    output_path = 'Programming/images/' + image_name + '_prepared.png'
    cv2.imwrite(output_path, reconstructed_image_A)

    # SETTING APPROXIMATION COEFFICIENTS TO ZERO
    coeffs_D = list(coeffs)
    for i in range(0, 1):
        coeffs_D[i] = tuple(np.zeros_like(element) for element in coeffs[i])

    reconstructed_image_D = pywt.waverec2(tuple(coeffs_D), wavelet)
    reconstructed_image_D = np.uint8(reconstructed_image_D)

    display_image('Detail Coefficients Only Reconstructed Image', reconstructed_image_D)

    return reconstructed_image_A


def performance(imArrayG):
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
    

def main():
    folder_path = 'Programming/images'
    image_name = '1_00001'

    # Import image
    imArray = image_import(folder_path, image_name)

    # Convert image to grayscale and convert to 8-bit integer
    imArrayG = gray_conversion(imArray)

    n = 4
    wavelet = 'coif12'
    
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
    prepared_image = reconstrucuted_images(coeffs, n, wavelet, image_name)

    """
    # Evaluate the performance before and after DWT applied
    performance(imArrayG)
    """
    
if __name__ == "__main__":
    main()


