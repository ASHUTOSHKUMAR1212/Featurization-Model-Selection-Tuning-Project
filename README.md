# Featurization-Model-Selection-Tuning-Project

Semiconductor Manufacturing Process: Yield Prediction Classifier
Project Objective

The goal of this project is to build a machine learning classifier to predict the Pass/Fail yield of a semiconductor manufacturing process entity. The data contains 591 features, and feature selection techniques will be applied to identify the most relevant signals for improving the classifier's accuracy and efficiency.
Data Description

    File: signal-data.csv
    Shape: (1567, 592) — 1567 datapoints with 591 features.
    Target: The label corresponds to pass (-1) or fail (1) for in-house line testing.
    The timestamp in the dataset represents the specific test point for the process entity.

Steps and Tasks
1. Data Understanding

    Load the Data: Import signal-data.csv as a Pandas DataFrame.
    Summary Statistics: Print the 5-point summary (min, max, mean, etc.) and share observations on the data distribution.

2. Data Cleansing

    Remove Features with High Null Values: Create a loop to remove all features with more than 20% missing values. Impute the rest of the missing values with the mean of the feature.
    Drop Constant Features: Identify and remove features with a constant value across all rows.
    Feature Elimination: Drop additional features using domain knowledge or functional reasoning.
    Multi-collinearity Check: Identify highly correlated features and take necessary actions (e.g., dropping one of the correlated features).
    Final Modifications: Apply any other relevant data cleaning techniques based on logical assumptions or domain expertise.

3. Data Analysis & Visualization

    Univariate Analysis: Perform detailed univariate analysis, exploring the distribution of individual features and commenting on the insights gained.
    Bivariate/Multivariate Analysis: Explore relationships between pairs and multiple variables to identify significant patterns.

4. Data Pre-processing

    Segregate Predictors and Target: Separate the features (predictors) from the target attribute.
    Target Balancing: Check for class imbalance in the target variable and apply techniques such as oversampling/undersampling if necessary.
    Train-Test Split: Split the data into training and testing sets (e.g., 80/20).
    Standardization/Normalization: Standardize or normalize the data to ensure all features are on the same scale.
    Train-Test Comparison: Compare the statistical characteristics of the train and test sets to ensure they resemble the original data.

5. Model Training, Testing, and Tuning

    Supervised Learning: Train a model using any supervised learning technique (e.g., Decision Tree, Random Forest, SVM).
    Cross-Validation: Use cross-validation techniques (e.g., K-Fold, Stratified K-Fold) to evaluate the model’s generalizability.
    Hyperparameter Tuning: Apply hyperparameter tuning techniques such as GridSearchCV to improve model performance.
    Performance Enhancement: Consider additional techniques like dimensionality reduction, feature selection, target balancing, and data standardization to further enhance performance.
    Classification Report: Display and explain the classification report with metrics like precision, recall, and F1-score.

6. Post-Training and Conclusion

    Model Comparison: Compare the accuracy of all the models trained on both training and test datasets.
    Select Final Model: Choose the best-performing model based on performance metrics and provide a detailed explanation for its selection.
    Pickle the Model: Save the best-trained model as a pickle file for future use.
    Conclusion: Write a conclusion summarizing the results and insights from the analysis and model performance.

Performance Metrics

For each model, the following metrics will be evaluated:

    Accuracy
    Precision
    Recall
    F1-Score
    ROC-AUC Score

Tools and Libraries Used

    Python: Programming language for model development.
    Pandas: Data manipulation and cleaning.
    NumPy: Numerical operations.
    Matplotlib/Seaborn: Data visualization.
    Scikit-learn: Machine learning models, cross-validation, hyperparameter tuning, and performance metrics.

Conclusion

This project will provide insights into which features are most important in predicting the Pass/Fail yield of a semiconductor manufacturing process. By comparing multiple models, we aim to select the most efficient classifier for production use. The final model will be stored for future predictions and analysis.
