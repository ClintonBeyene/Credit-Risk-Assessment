**Credit Risk Assessment Project**
=====================================

**Table of Contents**
-----------------

1. [Introduction](#introduction)
2. [Task 1: Understanding Credit Risk](#task-1-understanding-credit-risk)
3. [Task 2: Exploratory Data Analysis (EDA)](#task-2-exploratory-data-analysis-eda)
4. [Task 3: Feature Engineering](#task-3-feature-engineering)
5. [Task 4: Modelling](#task-4-modelling)
6. [Task 5: Model Serving API Call](#task-5-model-serving-api-call)
7. [Getting Started](#getting-started)
8. [Contributing](#contributing)

**Introduction**
---------------

This project aims to build a credit risk assessment model using machine learning techniques. The goal is to predict the likelihood of a customer defaulting on a loan based on their credit history and other relevant factors.

**Task 1: Understanding Credit Risk**
------------------------------------

📚 Read and understand the concept of credit risk and its importance in the financial industry.

* Key references:
	+ [Credit Scoring Approaches Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
	+ [How to Develop a Credit Risk Model and Scorecard](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
	+ [Credit Risk](https://www.risk-officer.com/Credit_Risk.htm)

**Task 2: Exploratory Data Analysis (EDA)**
------------------------------------------

📊 Explore the dataset and understand its structure, summary statistics, and distribution of numerical and categorical features.

* Tasks:
	+ Understand the structure of the dataset
	+ Calculate summary statistics (mean, median, mode, etc.)
	+ Visualize the distribution of numerical features
	+ Analyze the distribution of categorical features
	+ Perform correlation analysis
	+ Identify missing values and outliers

**Task 3: Feature Engineering**
------------------------------

💡 Create new features that can help improve the performance of the model.

* Tasks:
	+ Create aggregate features (e.g., total transaction amount, average transaction amount)
	+ Extract features (e.g., transaction hour, transaction day)
	+ Encode categorical variables (e.g., one-hot encoding, label encoding)
	+ Handle missing values (e.g., imputation, removal)
	+ Normalize/standardize numerical features

**Task 4: Modelling**
---------------------

📈 Train machine learning models to predict credit risk.

* Tasks:
	+ Split the data into training and testing sets
	+ Choose at least two models (e.g., logistic regression, decision trees, random forest)
	+ Train the models on the training data
	+ Perform hyperparameter tuning (e.g., grid search, random search)
	+ Evaluate model performance (e.g., accuracy, precision, recall, F1 score, ROC-AUC)

**Task 5: Model Serving API Call**
----------------------------------

📱 Create a REST API to serve the trained machine-learning models for real-time predictions.

* Tasks:
	+ Choose a framework (e.g., Flask, FastAPI, Django REST framework)
	+ Load the trained model
	+ Define API endpoints
	+ Handle requests and make predictions
	+ Return predictions as a response to the API call
	+ Deploy the API to a web server or cloud platform

**Getting Started**
-------------------

1. Clone the repository: `git clone https://github.com/ClintonBeyene/credit-risk-assessment.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the EDA notebook: `jupyter notebook 01-data-exploration.ipynb`
4. Run the feature engineering notebook: `jupyter notebook 02-feature-engineering.ipynb`
5. Run the modelling notebook: `jupyter notebook 03-modell-training.ipynb`

**Contributing**
---------------

Contributions are welcome! Please submit a pull request with your changes.

## Author: Clinton Beyene