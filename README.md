# Stroke Prediction Project

## Table of Contents

1. [Project Overview, Dataset Exploration, and Data Loading](#project-overview-dataset-exploration-and-data-loading)  
2. [Step 2: Exploratory Data Analysis (EDA) and Missing Values Handling](#step-2-exploratory-data-analysis-eda-and-missing-values-handling)  
3. [Step 3: Feature Engineering](#step-3-feature-engineering)  
4. [Step 4: Build and Train Machine Learning Models](#step-4-build-and-train-machine-learning-models)  
5. [Step 5: Evaluate and Tune Threshold](#step-5-evaluate-and-tune-threshold)  
6. [Step 6: Results Summary and Key Learnings](#step-6-results-summary-and-key-learnings)  
7. [Step 7: Next Steps](#step-7-next-steps)

---

## Project Overview, Dataset Exploration, and Data Loading

Stroke is a leading cause of death and disability worldwide. Early prediction can improve patient outcomes by enabling timely intervention. This project builds machine learning models to predict stroke risk from demographic, clinical, and lifestyle data.

The dataset contains approximately 5,000 patient records with the following features:

- Age (years)  
- BMI (Body Mass Index), with some missing values  
- Average Glucose Level  
- Hypertension (binary)  
- Heart Disease (binary)  
- Work Type (categorical)  
- Smoking Status (categorical)  
- Residence Type (urban/rural)  
- Marital Status (ever married or not)  
- Stroke (target variable)

Initial data loading reveals a class imbalance: about 5% positive stroke cases.

---

## Step 2: Exploratory Data Analysis (EDA) and Missing Values Handling

Stroke cases are a minority (~5%), making the dataset imbalanced. BMI has missing values, imputed using mean imputation to preserve dataset size.

Key EDA insights:

- Stroke patients are generally older.  
- Elevated glucose levels are more common in stroke cases.  
- Hypertension and heart disease show higher prevalence in stroke patients.  
- Work type and smoking status differ between stroke and non-stroke groups.  
- Minor differences in residence and marital status.

---

## Step 3: Feature Engineering

New features introduced to improve model performance:

- **Age Squared:** to capture nonlinear age effects.  
- **BMI-Glucose Interaction:** combined effect of BMI and glucose.  
- **Binned Glucose:** categorized glucose levels into clinical bins.

These features helped models capture complex stroke risk patterns.

---

## Step 4: Build and Train Machine Learning Models

Trained and compared multiple classifiers:

- Logistic Regression with class weighting.  
- Random Forest to capture nonlinearities.  
- Gradient Boosting for accuracy improvements.

Training used stratified splits, numeric scaling, and one-hot encoding. Models were evaluated with precision, recall, F1-score, and ROC AUC.

---

## Step 5: Evaluate and Tune Threshold

Because of imbalance, the default classification threshold of 0.5 was not optimal. Precision-recall curves guided threshold tuning to maximize F1-score and improve stroke detection.

---

## Step 6: Results Summary and Key Learnings

- Achieved ~91% accuracy and ROC AUC around 0.83–0.84.  
- Stroke recall improved to ~46% after threshold tuning.  
- Age, hypertension, and glucose level are top predictive features.  
- Feature engineering and imbalance handling were essential.

---

## Step 7: Next Steps

- Explore advanced imbalance methods like SMOTE.  
- Experiment with neural networks or ensemble methods.  
- Develop clinical decision support tools.  
- Perform model explainability analyses.

---
