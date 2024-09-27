**Project Overview**
This project focuses on predicting engine transmission ratings by performing regression analysis on the dataset. The primary objective was to clean the data by identifying and removing outliers, followed by implementing regression models to predict ratings effectively. The dataset contains features related to engine transmission, and the target variable is the engine transmission rating.

**Steps Performed**
1. Exploratory Data Analysis (EDA)
Initial analysis to understand the distribution of features.
Visualizations and summary statistics to identify patterns, correlations, and anomalies in the data.
2. Feature Selection
Selected relevant features based on correlation analysis and domain knowledge to improve model performance and reduce dimensionality.
3. Outlier Removal
Identified and removed outliers using statistical techniques and visualization tools such as boxplots to ensure model robustness.
4. Feature Extraction
Engineered new features that could provide additional predictive power.
Extracted important characteristics from existing features, including transformations to enhance model performance.
5. Data Preprocessing
Handled missing data using appropriate imputation techniques.
Standardized and normalized features to ensure that they are on the same scale, especially for distance-based models.
6. Model Implementation
Implemented two machine learning models: Decision Tree and Linear Regression.
Trained models using the training data and validated their performance on cross-validation data.
Final model evaluation was performed on a separate test dataset.
7. Model Evaluation
Mean Log Squared Error (MLSE) was used as the primary evaluation metric to measure model performance.
KDE plots were generated to visualize and compare the distribution of actual and predicted ratings.
8. Pipeline for Unseen Data
Created a full end-to-end pipeline to handle new, unseen data.
The pipeline includes all steps from preprocessing, feature extraction, and model prediction.

**Results**
The models were evaluated on the test dataset, and the performance was reported using MLSE.
KDE plots revealed how well the predicted values matched the actual ratings.
**Requirements**
The project includes a requirements file (requirements.txt) that lists all the necessary libraries and dependencies for running the code.
**Problem Objective**
The primary objective of this project was to identify outliers in the data and improve rating predictions by implementing regression techniques.
