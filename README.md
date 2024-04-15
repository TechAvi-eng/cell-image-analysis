# Cell Image Analysis Pipeline
## Overview
This repository contains the pipeline developed for the UCL Mechanical Engineering Third Year Project titled "Automated Processing of Cell Images Using the Discrete Wavelet Transform and Machine Learning".

This project involved designing a pipeline for cell image analysis including pre-processing, cell identification, cell counting, and cell maturity classification.

The pipeline has been developed and tested using Retinal pigment epithelium (RPE) cell images, and the BioMediTech RPE published dataset has been specifically used. Credit is given to the following paper for the RPE cell dataset: L. Nanni, M. Paci, F. L. C. Santos, H. Skottman, K. Juuti-Uusitalo, J. Hyttinen, Texture descriptors ensembles enable image-based classification of maturation of human stem cell-derived retinal pigmented epithelium, Plos One 2016.

This dataset is publicly available at: https://figshare.com/articles/dataset/BioMediTech_RPE_dataset/2070109

## Setup and Configuration Instructions
1. Download and open the full repository within an IDE
2. Copy the full dataset of interest into the folder named `Dataset` (Can use the BioMediTech RPE dataset cited above or alternate cell data, but this may require parameter adjustment)
   - Note: Some sample images are included within the `Dataset` extracted from the BioMediTech RPE dataset. Please replace these with the full dataset before attempting classification</li>
3. Install the code by entering `pip install -e .` into the command line.
4. Install the depdencies by entering 'pip install -r requirements.txt' into the command line
5. Run any `Cell_Counting.py`, `Cell_Classification.py` or `DWT_Performance.py` files (an overview of each file is provided below
6. For specific applications, e.g. only running the Elbow and Silhouette methods, please comment out the unnecessary functions in the 'main' function, which will avoid running the full pipelines functionality
7. Docstring comments are provided under each function describing its purpose</li>

## Python Files
Below a list is provided for the different functionality provided within each Python file contained within the repository.
1. Cell_Counting.py
   1. Dynamic Range adjustment algorithm
   2. DWT based multiresolution analysis for denoising
   3. Binarisation thresholding
   4. Morphological operation application
   5. Contour identification
   6. Cell mapping
2. Cell_Classification.py
   1. DWT based multiresolution analysis for generating DWT decomposition coefficients
   2. Extracting statistical features from both raw pixel values and DWT decomposition coefficients
   3. SVM cell classification (supervised ML, therefore, labelled data required)
   4. SVM decision boundary visualistation (completing 2 parameter classification
   5. Elbow and Silhouette methods for optimal number of clusters determination</li>
   6. K-means clustering cell classification (unsupervised ML, therefore, labelled data not required)
3. DWT_Performance.py
   - Following metrics are calculated for different DWT wavlet functions and vanishing moments
   1. Structural Similarity Index Measure (SSIM)
   2. Peak Signal-to-Noise Ratio (PSNR)
   3. Computational time calculation for different DWT functions and vanishing moments

## Results Achieved
