# Stroke Prediction with Machine Learning

---

## Table of Contents

1. [Introduction & Dataset Overview](#1-introduction--dataset-overview)  
2. [Data Cleaning and Preprocessing](#2-data-cleaning-and-preprocessing)  
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)  
4. [Feature Engineering](#4-feature-engineering)  
5. [Model Selection and Development](#5-model-selection-and-development)  
6. [Handling Class Imbalance](#6-handling-class-imbalance)  
7. [Model Evaluation](#7-model-evaluation)  
8. [Interpretability and Explainability](#8-interpretability-and-explainability)  
9. [Deployment](#9-deployment)  
10. [Challenges and Lessons Learned](#10-challenges-and-lessons-learned)  
11. [Future Work and Acknowledgements](#11-future-work-and-acknowledgements)  
12. [Conclusion](#12-conclusion)  

---

## 1. Introduction & Dataset Overview

### 1.1 Background and Motivation

Stroke remains one of the foremost causes of death and long-term disability worldwide. It occurs when the blood supply to the brain is interrupted or reduced, leading to brain cell death due to lack of oxygen and nutrients. According to the World Health Organization, approximately 15 million people suffer a stroke each year, with nearly 5 million deaths and 5 million survivors experiencing permanent disabilities.

This project explores the application of ML techniques to predict stroke occurrence using clinical and demographic data from the Kaggle Stroke Prediction Dataset. The goal is to build an accurate, interpretable, and deployable model that can aid healthcare providers and patients alike in stroke risk assessment.

---

### 1.2 Dataset Description

The dataset used in this study is publicly available on Kaggle and contains records for 5,110 patients, each described by 12 features, including the binary target variable indicating whether the patient had a stroke.

The features are summarized below:

| Feature           | Description                                                    |
|-------------------|----------------------------------------------------------------|
| **id**            | Unique patient identifier                                      |
| **gender**        | Patient’s gender: Male, Female, or Other                       |
| **age**           | Patient age in years                                           |
| **hypertension**  | Binary indicator: 1 if patient has hypertension, else 0       |
| **heart_disease** | Binary indicator: 1 if patient has heart disease, else 0      |
| **ever_married**  | Marital status: Yes or No                                      |
| **work_type**     | Patient’s work type: Govt_job, Private, Self-employed, Children, Never_worked |
| **Residence_type**| Urban or Rural residence                                       |
| **avg_glucose_level** | Average blood glucose level                                 |
| **bmi**           | Body Mass Index (some values missing)                         |
| **smoking_status**| Smoking status: formerly smoked, never smoked, smokes, unknown |
| **stroke**        | Target variable: 1 if stroke occurred, 0 otherwise            |

---

### 1.3 Initial Observations and Challenges

The dataset presents several important characteristics and challenges that shaped the modeling approach:

#### 1.3.1 Imbalanced Target Classes

- Stroke cases represent approximately 4.9% (249 out of 5,110) of the dataset, making the classification task highly imbalanced. This imbalance could cause models to be biased towards predicting the majority “no stroke” class if unaddressed.

#### 1.3.2 Missing Values

- The **bmi** feature contains missing values for about 4% of the records (201 out of 5,110). Since BMI is a crucial health indicator associated with stroke risk, effective imputation methods are necessary to avoid bias and loss of data.

#### 1.3.3 Categorical Variables

- Several features such as gender, work_type, smoking_status, and residence_type are categorical and require encoding into numerical representations that machine learning algorithms can process.

#### 1.3.4 Outliers and Noisy Data

- Certain observations show extreme or inconsistent values, particularly in BMI and glucose levels. These outliers need to be carefully handled during preprocessing to prevent distortion of model learning.

#### 1.3.5 Moderate Dataset Size

- With just over 5,000 samples, the dataset is sufficient to train classical machine learning models and small neural networks but not large-scale deep learning models.

---

### 1.4 Project Objectives

Given these characteristics, the project is designed to accomplish the following:

- **Develop predictive models** capable of accurately classifying patients’ stroke risk, prioritizing recall to minimize missed stroke cases.
- **Effectively handle missing and noisy data** through rigorous preprocessing and imputation strategies.
- **Apply feature engineering** to capture non-linear relationships and interactions that may improve predictive performance.
- **Address class imbalance** using resampling methods, class weighting, and threshold tuning.
- **Provide model interpretability** through tools like SHAP to foster clinical trust and usability.
- **Deploy the model** in an accessible interactive application for end users including clinicians and patients.

---


## 2. Data Cleaning and Preprocessing

### 2.1 Introduction

Data preprocessing is a critical step in any machine learning project. The quality and format of the input data greatly influence the performance and reliability of predictive models. In this project, preprocessing involved handling missing data, addressing outliers, encoding categorical variables, and scaling numerical features.

The raw stroke prediction dataset contained several imperfections, including missing BMI values, inconsistent categorical entries, and extreme outliers. This section describes the methods applied to clean and transform the data into a machine-learning-friendly format.

---

### 2.2 Handling Missing Values

#### 2.2.1 Missing BMI Values

The **bmi** feature had missing values in approximately 5% of the records. Since BMI is an important indicator of health and stroke risk, it was necessary to impute these missing values rather than drop affected rows, which would reduce data size and potentially bias the dataset.

Several imputation techniques were evaluated:

- **Mean Imputation:** Replacing missing values with the mean BMI value of the dataset.
- **Median Imputation:** Using the median BMI, less sensitive to outliers.
- **Regression Imputation:** Predicting missing BMI using other related features (such as age and average glucose) through linear or polynomial regression.

##### Regression-Based Imputation

Regression imputation was found to be more effective in preserving the relationship between BMI and other features. A quadratic regression model using age and average glucose level predicted missing BMI values, accounting for nonlinear associations observed during exploratory data analysis.

This method involved:

- Training a polynomial regression model on records with known BMI.
- Predicting BMI for records with missing values using this model.
- Validating imputation quality by comparing imputed values with observed distributions.

---

### 2.3 Outlier Detection and Column Removal

Certain extreme values in **bmi** and **avg_glucose_level** were identified as outliers that could skew model training:

- BMI values greater than 70 were considered implausible for the population and likely data entry errors. These rows were removed.
- Glucose levels above 300 mg/dL, which are physiologically extreme, were examined and handled case-by-case, with some removed and others capped at reasonable upper limits.
- The `gender` column contained a small number of records labeled as `"Other"`. Due to their scarcity and unclear definition, these records were removed.
- The `work_type` category `"Never_worked"` had very few samples and was also removed to avoid noise.
- The `id` column was dropped as it does not contribute to prediction.
---

## 3-Exploratory-Data-Analysis



Exploratory Data Analysis (EDA) is a fundamental step to understand the characteristics of the dataset and to identify relationships between variables that will inform feature engineering and model development.

### 3.1 Target Variable Distribution and Class Imbalance

The dataset exhibits a strong class imbalance:

| Stroke Status | Count | Percentage |
|---------------|-------|------------|
| No Stroke (0) | 4,939 | 96.7%      |
| Stroke (1)    | 171   | 3.3%       |

This imbalance poses a risk that naive models may simply predict the majority class, leading to misleadingly high accuracy but poor recall for stroke cases. Therefore, metrics like F1-score, precision, and recall must be emphasized in model evaluation.

### 3.2 Univariate Analysis of Key Features

![Alt text describing the graph](stroke_feature_comparison.png)

#### Age

- Age is distributed approximately normally but skewed slightly toward older age groups.
- Histogram and kernel density estimates (KDE) showed that stroke incidence increases sharply with age, particularly beyond 60 years.
- Median age for stroke patients is notably higher compared to non-stroke patients.

#### Average Glucose Level

- Glucose levels span a broad range, with a mean around 106 mg/dL.
- The distribution is right-skewed with a long tail of high glucose values.
- Stroke risk markedly increases when glucose exceeds approximately 175 mg/dL, suggesting a non-linear relationship.

#### Body Mass Index (BMI)

- BMI data showed a near-normal distribution with some skewness.
- The relationship with stroke is non-linear and somewhat inconsistent, with stroke risk fluctuating across BMI ranges.
- Outliers with extremely high BMI were removed prior to analysis.

### 3.3 Categorical Variable Analysis

- **Gender:** After excluding rare 'Other' category, stroke prevalence was similar across male and female patients.
- **Hypertension:** Strong positive association with stroke; hypertensive patients had significantly higher stroke incidence.
- **Heart Disease:** Similarly, heart disease presence correlated strongly with stroke.
- **Smoking Status:** Current and former smokers had higher stroke rates compared to never smokers.
- **Work Type and Residence Type:** No substantial difference in stroke prevalence detected.

### 3.4 Bivariate and Interaction Analysis

- Investigated interactions between features, particularly between continuous variables.
- The product term `age * bmi` showed a moderate correlation with stroke, indicating that the combined effect of aging and BMI might be relevant.
- Polynomial terms such as `age²` and `glucose²` captured observed non-linear effects.
- Visualizations such as scatter plots and boxplots were used to inspect these interactions.

### 3.5 Correlation Matrix

- Calculated Pearson correlation coefficients for numerical features and the target.
- `age` and `avg_glucose_level` showed positive correlations with stroke.
- `bmi` showed a weak positive correlation.
- Checked feature-to-feature correlations to detect multicollinearity risks.

### 3.6 Feature Binning

- Created bins for continuous variables to discretize them into meaningful groups:
  - **Age:** Divided into <40 years, 40–60 years, and >60 years categories to align with clinical risk brackets.
  - **Avg Glucose Level:** Grouped into normal (<140 mg/dL), prediabetic (140–199 mg/dL), and diabetic (≥200 mg/dL) ranges.
- Binning helped simplify complex relationships and was useful for models that benefit from categorical inputs.


## Feature Importance Analysis

To better understand which features contribute most to predicting stroke, we trained a Random Forest classifier with class weighting to handle imbalance and extracted feature importances.

The top 20 features ranked by their importance in the model are shown below:

![Top 20 Feature Importances from Random Forest](images/rf_feature_importance.png)

### Insights from Feature Importance

- **Age** emerged as the most important predictor, consistent with medical knowledge that stroke risk increases with age.
- **Average glucose level** was also highly predictive, reflecting the impact of blood sugar on stroke risk.
- Medical history indicators such as **hypertension** and **heart disease** showed strong contributions.
- Interaction features and engineered polynomial terms, where included, also showed meaningful influence.

This analysis informed further feature engineering and model tuning efforts.
