import numpy as np
import pywt
import pywt.data
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import seaborn as sns


def cell_image_import(input_folder_path):
    """
    Go through folder and store names of images in list
    Parameters:
        input_folder_path (str): folder path of images
    Returns:
        image_list (list): list of image names
    """
    
    image_list = []

    for filename in os.listdir(input_folder_path):
        if filename.endswith('.png'):  
            image_list.append(filename)

    print('SUCCESS: Written list of images')

    return image_list


def discrete_wavelet_transform(input_folder_path, image_list, decomposition_level):
    """
    Complete DWT
    Parameters:
        imArrayG (array): grayscale 8-bit image array
        n (int): number of levels
        wavelet (str): wavelet type
    Returns:
        coeffs (array): coefficients array from DWT
    """
    
    wavelet = 'coif12'
    n = decomposition_level

    cell_data = []

    labels = []

    print('STARTING DWT...')
    
    # For first 5 images in the folder
    for image in image_list:
        image_path = os.path.join(input_folder_path, image)
        imArray = cv2.imread(image_path)
        imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
        imArrayG = np.uint8(imArrayG)
        
        print('Image Imported:', image)

        coeffs = pywt.wavedec2(imArrayG, wavelet, level=n) # complete DWT

        cA = coeffs[0]
        cA = cA.flatten()

        mean = np.mean(cA)
        std = np.std(cA)
        skewness = skew(cA)
        kurt = kurtosis(cA)
        median = np.median(cA)
        co_range = np.max(cA) - np.min(cA)
        mean_square = np.mean(cA ** 2)
        rms = np.sqrt(mean_square)
        entro = shannon_entropy(cA)

        new_row = [mean, std, skewness, kurt, median, co_range, rms, entro]

        # Starts at level n decomposition and ends at level 1
        for level in range(1,n+1):
            cD = coeffs[level]

            # Loops through vertical, horizontal and diagonal detail coefficients
            for i in range(3):
                details = cD[i]
                details = cD[i].flatten()

                mean = np.mean(details)
                std = np.std(details)
                skewness = skew(details)
                kurt = kurtosis(details)
                median = np.median(details)
                co_range = np.max(details) - np.min(details)

                mean_square = np.mean(details ** 2)
                rms = np.sqrt(mean_square)

                entro = shannon_entropy(details)

                new_row = new_row + [mean, std, skewness, kurt, median, co_range, rms, entro]

        # Append the label to the labels list
        label = image[0]
        label = int(label)
        labels.append(label)
        
        cell_data.append(new_row)
        
    print('SUCCESS: Completed DWT')

    cell_data = np.array(cell_data, dtype=float)
    print('Cell Data:', cell_data.shape)

    return cell_data, labels


def data_split(cell_data, labels):
    # Split dataset into training set and test set
    data_train, data_test, label_train, label_test = train_test_split(cell_data, labels, test_size=0.3,random_state=109) # 70% training and 30% test

    print('SUCCESS: Split data into training and test sets')
    print('Training Data Size:', data_train.shape)
    print('Test Data Size:', data_test.shape)

    return data_train, data_test, label_train, label_test


def kmeans_clustering(data_train, data_test, label_train, label_test):
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(data_train)

    test_clusters = kmeans.predict(data_test)

    test_accuracy = accuracy_score(label_test, test_clusters)
    
    print("Accuracy for K Means:", test_accuracy)

    return test_accuracy


def main():
    input_folder_path = '/Users/nikhildhulashia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Third Year/Individual Project/Datasets/RPE_dataset/Subwindows'
    
    # Import image names
    image_list = cell_image_import(input_folder_path)

    decomposition_levels = [1, 2, 3, 4, 5, 6]
    accuracy = []
    for i in decomposition_levels:
        # Complete DWT and return data and labels
        cell_data, labels = discrete_wavelet_transform(input_folder_path, image_list, i)

        # Split data into training and test sets
        data_train, data_test, label_train, label_test = data_split(cell_data, labels)

        # KMeans Clustering
        test_accuracy = kmeans_clustering(data_train, data_test, label_train, label_test)
        accuracy.append(test_accuracy)

    print("Accuracy :")
    print(accuracy)

if __name__ == "__main__":
    main()


