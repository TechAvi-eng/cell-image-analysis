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


def cell_image_import(folder_path):
    """
    Go through folder and store names of images in list
    Parameters:
        folder_path (str): folder path of images
    Returns:
        image_list (list): list of image names
    """
    
    image_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_list.append(filename)
    
    print('SUCCESS: Written list of images')

    return image_list


def gray_conversion(image_list, folder_path, gray_folder_path):
    """
    Import images from folder and convert to grayscale, convert to 8-bit integer and save as new images
    Parameters:
        image_list (list): list of image names
        folder_path (str): folder path of raw images
        gray_folder_path (str): folder path of grayscale images    
    """
    # If the Gray Folder Path does not contai
    for image_name in image_list:
        image_path = os.path.join(folder_path, image_name)
        imArray = cv2.imread(image_path)
        imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
        imArrayG = np.uint8(imArrayG)
        output_path = gray_folder_path + '/' + image_name
        cv2.imwrite(output_path, imArrayG)

    print('SUCCESS: Converted images to grayscale')

    return


def discrete_wavelet_transform(gray_folder_path):
    """
    Complete DWT
    Parameters:
        imArrayG (array): grayscale 8-bit image array
        n (int): number of levels
        wavelet (str): wavelet type
    Returns:
        coeffs (array): coefficients array from DWT
    """
    wavelet = 'haar'
    
    cell_data = []
    labels = []

    print('STARTING DWT...')
    for image in os.listdir(gray_folder_path):

        image_path = gray_folder_path + '/' + image
        imArrayG = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        coeffs = pywt.dwt(imArrayG, wavelet)
        Approx = coeffs[0]
        
        # Make the Approximation matrix a single row
        Approx = Approx.flatten()
        
        # Append the Approximation matrix to the data list
        cell_data.append(Approx)

        # Append the label to the labels list
        label = image[0]
        label = int(label)
        labels.append(label)

    print('SUCCESS: Completed DWT')

    cell_data = np.array(cell_data)
    
    return cell_data, labels


def data_split(cell_data, labels):
    # Split dataset into training set and test set
    data_train, data_test, label_train, label_test = train_test_split(cell_data, labels, test_size=0.3,random_state=109) # 70% training and 30% test

    print('SUCCESS: Split data into training and test sets')
    print('Training Data Size:', data_train.shape)
    print('Test Data Size:', data_test.shape)

    return data_train, data_test, label_train, label_test


def svm_classifier(data_train, data_test, label_train, label_test):
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    print('SUCCESS: Created SVM Classifier')

    print('STARTING Training SVM Classifier...')
    #Train the model using the training sets
    clf.fit(data_train, label_train)

    print('SUCCESS: Trained SVM Classifier')

    #Predict the response for test dataset
    label_pred = clf.predict(data_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(label_test, label_pred))

    return


def kmeans_clustering(data_train, data_test, label_train, label_test):
    kmeans = KMeans(n_clusters=3)
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
    folder_path = '/Users/nikhildhulashia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Third Year/Individual Project/Datasets/rpe_raw_images'
    gray_folder_path = '/Users/nikhildhulashia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Third Year/Individual Project/Datasets/rpe_gray_images'

    # Import image names
    image_list = cell_image_import(folder_path)

    # Convert images to grayscale and save as new images
    gray_conversion(image_list, folder_path, gray_folder_path)

    # Complete DWT and return data and labels
    cell_data, labels = discrete_wavelet_transform(gray_folder_path)

    # Split data into training and test sets
    data_train, data_test, label_train, label_test = data_split(cell_data, labels)

    # Create a svm Classifier
    svm_classifier(data_train, data_test, label_train, label_test)

    # KMeans Clustering
    kmeans_clustering(data_train, data_test, label_train, label_test)
    
    

if __name__ == "__main__":
    main()


