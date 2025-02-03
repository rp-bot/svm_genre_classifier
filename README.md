# SVM and Random Forest Genre Classifier of Drum Loops

The objective of this project is to take survey Machine Learning Models for the purpose of genre classifications.
The questions addressed in this project are:
1. What are Support Vector Machines (SVM)? What are the types of SVM?
2. What are the features that we can extract that constitutes drums?
3. How do we perform dimensionality reduction?
4. What is Random Forest?

# Methodology
## Dataset
A custom dataset was created using the [Waiveops](https://www.patchbanks.com/waivops/) dataset created by the people at Patchbanks.
Each audio file was loaded using librosa of 5 second duration. And in order to not have any distortion. I added a 0.5 second fadeout. 
`make_csv.py` creates an annotations file for each folder. `extract_features.py` extract all the features specified and stores it as a `.npz` file.

In order to maintain equal contribution from each class, the `_dataloader.py` script creates equal slices and concatnates it to create one set. Which is later split into test and train sets. 

## Feature Selection
Feature selection is the most crucial part.
1. 13 MFCC coefficients were extracted from the 5 second clip as the main feature.
2. Onset Strength
3. Zero Crossing Rate.

The features were first normalized using a simple min-max normalization equation. 
To construct the feature vector, the mean over time produced a 1 x 15 vector. 

## Results
| Model    | Accuracy | Precision | Recall| F1-Score| 
| -------- | ------- | --------| -----|------|
|Linear SVM | 0.83| 0.84 | 0.83 | 0.83 |
|Radial Basis Function SVM| 0.98 | 0.98 | 0.98|0.98|
|Random Forest| 0.97 | 0.97 | 0.97|0.97|



