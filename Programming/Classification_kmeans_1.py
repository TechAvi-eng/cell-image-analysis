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
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import seaborn as sns
import csv


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
        # if filename endswith '.png' and starts with '1' or '2' or '3'
        # if filename.endswith('.png') and (filename.startswith('1') or filename.startswith('2') or filename.startswith('3')):
        if filename.endswith('.png'):  
            image_list.append(filename)

    print('SUCCESS: Written list of images')

    return image_list


def discrete_wavelet_transform(input_folder_path, image_list):
    """
    Complete DWT
    Parameters:
        imArrayG (array): grayscale 8-bit image array
        n (int): number of levels
        wavelet (str): wavelet type
    Returns:
        coeffs (array): coefficients array from DWT
    """
    
    wavelet = 'db2'
    n = 2

    cell_data = []

    labels = []

    print('STARTING DWT...')
    
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
    # file_name = "Elbow.csv"
    
    # inertias = []
    # # Save first 10 data points for elbow method
    # elbow_data = data_train
    # max_clusters = 10
    # with open(file_name, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for i in range(1, max_clusters):
    #         kmeans = KMeans(n_clusters=i)
    #         kmeans.fit(elbow_data)
    #         inertias.append(kmeans.inertia_)
    #         csv_writer.writerow([i, kmeans.inertia_])
    
    # plt.plot(range(1, max_clusters), inertias, marker = 'o')
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.show()

    # file_name = "Silhouette.csv"

    # silhouette_avg = []
    # with open(file_name, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for i in range(2, max_clusters):
    #         kmeans = KMeans(n_clusters=i)
    #         kmeans.fit(elbow_data)
    #         cluster_labels = kmeans.labels_
    #         silhouette_avg.append(silhouette_score(elbow_data, cluster_labels))
    #         csv_writer.writerow([i, silhouette_score(elbow_data, cluster_labels)])
        
    # plt.plot(range(2, max_clusters), silhouette_avg, marker = 'o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score') 
    # plt.title('Silhouette Method')
    # plt.show()

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data_train)

    train_cluster_labels = kmeans.labels_

    clf = LogisticRegression()
    clf.fit(train_cluster_labels.reshape(-1, 1), label_train)
    
    test_cluster_labels = kmeans.predict(data_test)
    predicted_labels = clf.predict(test_cluster_labels.reshape(-1, 1))

    accuracy = accuracy_score(label_test, predicted_labels)
    print("Accuracy for K Means:", accuracy)

    return kmeans


def main():
    input_folder_path = '/Users/nikhildhulashia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Third Year/Individual Project/Datasets/RPE_dataset/Subwindows'
    
    # Import image names
    image_list = cell_image_import(input_folder_path)

    # Complete DWT and return data and labels
    cell_data, labels = discrete_wavelet_transform(input_folder_path, image_list)

    # Split data into training and test sets
    data_train, data_test, label_train, label_test = data_split(cell_data, labels)

    # KMeans Clustering
    kmeans_clustering(data_train, data_test, label_train, label_test)

    

if __name__ == "__main__":
    main()


