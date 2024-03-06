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

def cell_image_import(folder_path):
    """
    Go through folder and store names of images in list
    Parameters:
        folder_path (str): folder path of images
    Returns:
        image_list (list): list of image names
    """

    raw_images_data = []
    image_labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):

            imArray = cv2.imread(folder_path + '/' + filename)
            imArrayG = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
            imArrayG = np.uint8(imArrayG)
            raw_pixels = imArrayG.flatten()

            raw_mean = np.mean(raw_pixels)
            raw_std = np.std(raw_pixels)

            raw_images_data.append([raw_mean, raw_std])
            label = filename[0]
            label = int(label)
            image_labels.append(label)

    print('SUCCESS: Written list of images')

    raw_images_data = np.array(raw_images_data)
        
    print('SUCCESS: Gathered Image Information')

    # print number of lables with 1, 2 and 3
    print('Number of labels with 1:', image_labels.count(1))
    print('Number of labels with 2:', image_labels.count(2))
    print('Number of labels with 3:', image_labels.count(3))

    return raw_images_data, image_labels


def data_split(cell_data, labels):
    # Split dataset into training set and test set
    data_train, data_test, label_train, label_test = train_test_split(cell_data, labels, test_size=0.3,random_state=0) # 70% training and 30% test

    print('SUCCESS: Split data into training and test sets')
    print('Training Data Size:', data_train.shape)
    print('Test Data Size:', data_test.shape)

    return data_train, data_test, label_train, label_test


def svm_classifier(data_train, data_test, label_train, label_test):
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel

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


def main():
    folder_path = '/Users/nikhildhulashia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Third Year/Individual Project/Datasets/rpe_raw_images'

    # Import image
    images_data, image_labels = cell_image_import(folder_path)

    # Split data into training and test sets
    data_train, data_test, label_train, label_test = data_split(images_data, image_labels)

    # Create a svm Classifier
    svm_classifier(data_train, data_test, label_train, label_test)
    
    

if __name__ == "__main__":
    main()


