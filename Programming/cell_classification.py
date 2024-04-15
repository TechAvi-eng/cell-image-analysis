import cv2
import os
import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from matplotlib.font_manager import FontProperties


# This program is used to classify the maturity of RPE cells using SVM and KMeans clustering
# The program uses Discrete Wavelet Transform to extract features from the images
# Note: The full dataset must be placed within the 'Dataset' folder for the program to work

def cell_image_import(input_folder_path):
    """
    Go through the image folder and store names of images in list
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


def discrete_wavelet_transform(input_folder_path, image_list):
    """
    Complete DWT based multiresolution analysis to generate DWT decomposition coefficients
    Parameters:
        input_folder_path (str): folder path of images
        image_list (list): list of image names
    Returns:
        coeffs (array): coefficients array from DWT
    """
    # DWT Parameters
    wavelet = 'db12'
    n = 2

    cell_data = []

    labels = []

    print('STARTING DWT...')
    
    # Loop through each image in the image list
    for image in image_list:
        image_path = os.path.join(input_folder_path, image)
        imArray = cv2.imread(image_path)
        imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
        imArrayG = np.uint8(imArrayG)
        
        print('Image Imported:', image)

        coeffs = pywt.wavedec2(imArrayG, wavelet, level=n) # Complete DWT

        # Extracting statistical features from the approximation coefficients
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
        
        # Stores the statistical features in a new row
        new_row = [mean, std, skewness, kurt, median, co_range, rms, entro]

        # Starts at level n decomposition and ends at level 1
        for level in range(1,n+1):
            cD = coeffs[level]

            # Loops through vertical, horizontal and diagonal detail coefficients, and extracts statistical features
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

                # Append the statistical features to the new row
                new_row = new_row + [mean, std, skewness, kurt, median, co_range, rms, entro]

        # Append the label to the labels list (for supervised learning)
        label = image[0]
        label = int(label)
        labels.append(label)
        
        cell_data.append(new_row)
        
    print('SUCCESS: Completed DWT')

    cell_data = np.array(cell_data, dtype=float)

    # Output the size of the input cell data
    print('Cell Data:', cell_data.shape)

    return cell_data, labels


def data_split(cell_data, labels):
    """
    Split the data into training and test sets
    Parameters:
        cell_data (array): input data containing statistical features extracted from DWT decomposition coefficients
        labels (list): list of labels extracted for the name of each image
    Returns:
        data_train (array): training data
        data_test (array): test data
        label_train (list): training labels
        label_test (list): test labels
    """
    # Split dataset into training set and test set
    data_train, data_test, label_train, label_test = train_test_split(cell_data, labels, test_size=0.3,random_state=109) # 70% training and 30% test

    print('SUCCESS: Split data into training and test sets')
    print('Training Data Size:', data_train.shape)
    print('Test Data Size:', data_test.shape)

    return data_train, data_test, label_train, label_test


def svm_classifier(data_train, data_test, label_train, label_test):
    """
    Complete SVM Classification using the training and test sets
    Parameters:
        data_train (array): training data
        data_test (array): test data
        label_train (list): training labels
        label_test (list): test labels
    """
    #Create an svm Classifier
    clf = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel, and balanced class weights for imbalanced dataset

    print('SUCCESS: Created SVM Classifier')

    print('STARTING Training SVM Classifier...')
    # Train the model using the training sets
    clf.fit(data_train, label_train)

    print('SUCCESS: Trained SVM Classifier')

    # Predict the response for test dataset
    label_pred = clf.predict(data_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(label_test, label_pred))

    return


def svm_classifier_visualisation(data_train, data_test, label_train, label_test):
    """
    Perform an SVM classification using only 2 features and visualise the decision boundaries
    Parameters:
        data_train (array): training data
        data_test (array): test data
        label_train (list): training labels
        label_test (list): test labels
    """
    # Create a svm Classifier
    classifier = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel and balanced class weights for imbalanced dataset

    # Reduce the data to 2 features
    data_train = data_train[:, :2]
    data_test = data_test[:, :2]
    
    # Train the model using the training sets
    classifier.fit(data_train, label_train)

    print('SUCCESS: Trained SVM Classifier')
    
    # Predict the response for test dataset
    label_pred = classifier.predict(data_test)

    print("Accuracy for 2 Feature Classification:",metrics.accuracy_score(label_test, label_pred))

    # Generate a grid of points to plot decision boundaries
    min_x, max_x = data_train[:, 0].min() - 1, data_train[:, 0].max() + 1
    min_y, max_y = data_train[:, 1].min() - 1, data_train[:, 1].max() + 1
    x_axis, y_axis = np.meshgrid(np.arange(min_x, max_x, 0.1),
                        np.arange(min_y, max_y, 0.1))

    # Plot decision boundaries
    boundary = classifier.predict(np.c_[x_axis.ravel(), y_axis.ravel()])
    boundary = boundary.reshape(x_axis.shape)
    plt.contourf(x_axis, y_axis, boundary, alpha=0.4)

    # Plot data points
    sns.scatterplot(x=data_train[:, 0], y=data_train[:, 1], hue=label_train, palette="Set1")

    # Graph formatting
    plt.xlabel('Mean', fontsize=11, fontname='Times New Roman')
    plt.ylabel('Standard Deviation', fontsize=11, fontname='Times New Roman')
    plt.title('SVM Decision Boundaries', fontsize=12, fontname='Times New Roman')
    plt.xticks(fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')
    legend_font = FontProperties(family='Times New Roman', size=11)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', labels=["Epithelioid", "Fusiform", "Mixed", "Cobblestone"], prop=legend_font)
    legend.set_title("Maturity Classification", prop={'size': 11, 'family': 'Times New Roman'})
    plt.gcf().set_size_inches(8.38/2.54, 6/2.54)
    plt.gcf().set_dpi(600)
    plt.tight_layout()
    plt.show()

    return


def elbow_method(data_train):
    """
    Perform the elbow method to determine the optimal number of clusters for k-means
    Parameters:
        data_train (array): training data
    """        
    inertias = []

    # Save first 10 data points for elbow method
    elbow_data = data_train
    max_clusters = 10

    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(elbow_data)
        inertias.append(kmeans.inertia_)
    
    plt.plot(range(1, max_clusters), inertias, marker = 'o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    
    return


def silhouette_method(data_train):
    """
    Perform the silhouette method to determine the optimal number of clusters for k-means
    Parameters:
        data_train (array): training data
    """
    silhouette_avg = []

    # Save first 10 data points for silhouette method
    silhouette_data = data_train
    max_clusters = 10

    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(silhouette_data)
        cluster_labels = kmeans.labels_
        silhouette_avg.append(silhouette_score(silhouette_data, cluster_labels))
        
    plt.plot(range(2, max_clusters), silhouette_avg, marker = 'o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score') 
    plt.title('Silhouette Method')
    plt.show()

    return


def kmeans_clustering(data_train, data_test, label_train, label_test):
    """
    Perform k-means clustering and then classify the test data to determine accuracy
    Parameters:
        data_train (array): training data
        data_test (array): test data
        label_train (list): training labels
        label_test (list): test labels
    """
    # Number of clusters set to 4 (maturity classes) and dandom state selected for reproducibility
    kmeans = KMeans(n_clusters=4, random_state=42) 
    kmeans.fit(data_train)

    test_clusters = kmeans.predict(data_test)

    test_accuracy = accuracy_score(label_test, test_clusters)

    print("Accuracy for K Means:", test_accuracy)

    return


def main():
    # Path to folder containing images
    input_folder_path = '/Users/nikhildhulashia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Third Year/Individual Project/Datasets/RPE_dataset/Subwindows'
    
    # Create a list containing the image names
    image_list = cell_image_import(input_folder_path)

    # Complete DWT processing and return data and labels
    cell_data, labels = discrete_wavelet_transform(input_folder_path, image_list)

    # Split data into training and test sets
    data_train, data_test, label_train, label_test = data_split(cell_data, labels)

    # Complete SVM Classification
    svm_classifier(data_train, data_test, label_train, label_test)

    # Visualise SVM decision boundaries (performs 2 parameter classification only)
    svm_classifier_visualisation(data_train, data_test, label_train, label_test)

    # Perform Elbow Method to determine optimal number of clusters
    elbow_method(data_train)

    # Perform Silhouette Method to determine optimal number of clusters
    silhouette_method(data_train)

    # Complete k-means Clustering
    kmeans_clustering(data_train, data_test, label_train, label_test)


if __name__ == "__main__":
    main()


