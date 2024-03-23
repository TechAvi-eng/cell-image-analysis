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
from sklearn.preprocessing import StandardScaler
from sklearn_som.som import SOM

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
    
    # For first 5 images in the folder
    for image in image_list[:50]:
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
    inertias = []
    # Save first 10 data points for elbow method
    elbow_data = data_train[:100]
    max_clusters = len(elbow_data)
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(elbow_data)
        inertias.append(kmeans.inertia_)
    
    plt.plot(range(1, max_clusters), inertias, marker = 'o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

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


def self_organising_map(data_train, data_test, label_train, label_test):
    n_data_points = data_train.shape[0]
    n_neurons = 5 * n_data_points ** (1 / 2)

    # Get data, select which features to analyse
    classifying_metrics = [
        "mean",
        "std",
        "skewness",
        "kurtosis",
        "median",
        "co_range",
        "rms",
        "entropy"
    ]
    
    aspect_ratio = 1.5  # width / height of som
    epochs = 15  # scan through the data more times
    max_iterations = int(1e5)  # maximum number of iterations across all epochs

    n_metrics = len(classifying_metrics)

    # Standardise data
    scaler = StandardScaler()
    data_train_normalised = scaler.fit_transform(data_train)

    # Self organising maps
    som_cols = int(np.sqrt(n_neurons * aspect_ratio))  # make a map with specified aspect ratio
    som_rows = int(n_neurons / som_cols)
    som_size = (som_rows, som_cols)  # set size
    som_height, som_width = som_size
    n_clusters = som_width * som_height
    som = SOM(som_height, som_width, n_metrics, max_iter=max_iterations)
    labels = som.fit_predict(data_train_normalised, epochs=epochs)
    print("itertations", som.n_iter_)

    # Cluster centres
    centers = som.cluster_centers_
    # display(centers)
    centers = som.cluster_centers_.reshape((n_clusters, n_metrics))
    # print(centers)
    centers = scaler.inverse_transform(centers)

    # print("centers:", centers)

    # Plotting heatmaps for each feature
    columns = 3
    rows = n_metrics / columns
    if rows.is_integer() is False:
        rows = int(rows) + 1  # round up if there aren't enough rows
    rows = max(2, rows)
    fig, axes = plt.subplots(int(rows), columns, figsize=(12 * aspect_ratio, 12))

    for i, feature in enumerate(classifying_metrics):
        row, col = i // columns, i % columns
        feature_values = centers[:, i].reshape(som_size)
        # print(centers[:, i])
        ax:plt.Axes = axes[row, col]
        ax.imshow(feature_values, cmap='turbo', aspect='equal')
        ax.set_title(f'{feature.capitalize()}', fontsize=25)
        ax.set_xlabel("Node number", fontsize=16)
        ax.set_ylabel("Node number", fontsize=16)
        ax.set_xticks(range(som_width))
        ax.set_yticks(range(som_height))

    plt.tight_layout()
    plt.show()

    return


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

    # Self Organising Maps
    self_organising_map(data_train, data_test, label_train, label_test)


if __name__ == "__main__":
    main()