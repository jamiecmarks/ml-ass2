# TRAFFIC SIGN CLASSIFICATION – COMP30027 PROJECT 2 (2025)

## Overview:
This project aims to classify German traffic signs into one of 43 classes using extracted features. 

## Directory Structure:
The project is organized as follows:

- train\
    Folder with training image
	- train_metadata.csv
    		Metadata for training images (ids, image filenames and class labels)
	- Features\
    		Contains:
      		- color_histogram.csv
      		- hog_pca.csv
      		- additional_features.csv

- test\
    Folder with test images 
	- test_metadata.csv
    		Metadata for test images (ids, image filenames only, no labels)
	- Features\
    		Contains:
      		- color_histogram.csv
      		- hog_pca.csv
      		- additional_features.csv

- README.txt
    This file

## Data:
- 5488 training images with class labels
- 2353 test images without labels
- 43 total traffic sign classes
- Provided features:
    * HOG (Histogram of Oriented Gradients) - PCA reduced
    * Color histograms
    * Additional features (edge density, texture variance, mean RGB)


## Submission Format (Kaggle) (CSV):
Final submission should follow this structure:

Id,ClassId
67.jpg,4
94,2
...
521.jpg,12

## Code Setup

### (Macbook) First setup virtual environment and install required packages
```bash
# The setup below includes GPU boosting
python3 -m venv tf-metal
source tf-metal/bin/activate
pip install -r requirements.txt
```

### Notebooks
- `exploration.ipynb` - initial exploration with SVM, Logistic Regression, KNN and Stacking, with a brief CNN exploration. This notebook also contains the code to produce many of the figures used in the report
- `tuning.ipynb` - contains code for hyperparameter tuning of SVM, Logistic Regressionm, and KNN. Also tried stacking with tuned base learners
- `best_cnn.ipynb` - contains code for how we trained our best CNN model

### Other files
- `top_cnn_model.h5` - a saved version of our top performing model
- `best_cnn_model.h5` - a saved version of the best model after training
- `final.csv` - highest performing predictions by CNN, submitted to Kaggle
- `cnn_predictions.csv` - predictions made by CNN after running the notebook for the test set
- `lr_predictions.csv` - predictions made by logistic regression after running the notebook for the test set
- `stacking_predictions.csv` - predictions made by stacking model after running the notebook for the test set
- `knn_predictions.csv` - predictions made by KNN model after running the notebook for the test set
- `svm_predictions.csv` - predictions made by svm model after running the notebook for the test set
- `custom_test_data.csv` - contains custom features for training and testing
- `mismatched_images.png` - outputs misclassified images from the CNN notebook
- `mismatched_images.csv` - outputs misclassified instances from the CNN notebook

## Running the code

All of notebooks are runnable. Just open any and run all blocks.