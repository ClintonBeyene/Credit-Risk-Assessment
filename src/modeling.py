# modeling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self, X_train=None, X_test=None, y_train=None, y_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.best_models = {}

    def load_data(self, data_path, target_col, highly_correlated=[]):
        logging.info("Loading data from CSV file.")
        data = pd.read_csv(data_path)
        
        # Print the column names to verify
        logging.info(f"Columns in the data: {data.columns.tolist()}")
        
        # Ensure highly_correlated is a list
        if not isinstance(highly_correlated, list):
            raise ValueError("highly_correlated must be a list.")
        
        # Check if the target column exists
        if target_col not in data.columns:
            raise KeyError(f"Target column '{target_col}' not found in the data.")
        
        # Check if all highly correlated columns exist
        missing_columns = [col for col in highly_correlated if col not in data.columns]
        if missing_columns:
            raise KeyError(f"Columns {missing_columns} not found in the data.")
        
        # Drop the target column and highly correlated columns
        columns_to_drop = [target_col] + highly_correlated
        X = data.drop(columns=columns_to_drop)
        y = data[target_col]
        return X, y

    def split_data(self, X, y, test_size=0.3, random_state=42, stratify=None):
        logging.info("Splitting data into training and testing sets.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def initialize_models(self):
        logging.info("Initializing models.")
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }

    def train_models(self):
        logging.info("Training models on the training data.")
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            logging.info(f"{name} trained.")

    def evaluate_models(self):
        logging.info("Evaluating models on the training data.")
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_train)
            y_pred_prob = model.predict_proba(self.X_train)[:, 1]
            results[name] = {
                'Accuracy': accuracy_score(self.y_train, y_pred),
                'Precision': precision_score(self.y_train, y_pred),
                'Recall': recall_score(self.y_train, y_pred),
                'F1 Score': f1_score(self.y_train, y_pred),
                'ROC-AUC': roc_auc_score(self.y_train, y_pred_prob)
            }
        return results

    def define_hyperparameter_grids(self):
        logging.info("Defining hyperparameter grids for each model.")
        self.param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }

    def perform_grid_search(self):
        logging.info("Performing Grid Search for each model.")
        for name, model in self.models.items():
            grid_search = GridSearchCV(model, self.param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            self.best_models[name] = grid_search.best_estimator_
            logging.info(f"{name} Best Parameters: {grid_search.best_params_}")

    def evaluate_best_models(self):
        logging.info("Evaluating the best models on the testing data.")
        results = {}
        for name, model in self.best_models.items():
            y_pred = model.predict(self.X_test)
            y_pred_prob = model.predict_proba(self.X_test)[:, 1]
            results[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_prob)
            }
        return results

    def plot_roc_curves(self):
        logging.info("Plotting ROC-AUC curves for each model.")
        plt.figure(figsize=(10, 7))
        for name, model in self.best_models.items():
            y_pred_prob = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def save_best_models(self, save_dir='models'):
        logging.info("Saving best models for further analysis and deployment.")
        
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for name, model in self.best_models.items():
            file_path = os.path.join(save_dir, f"{name}_best_model.pkl")
            try:
                joblib.dump(model, file_path)
                logging.info(f"Model {name} saved to {file_path}.")
            except Exception as e:
                logging.error(f"Failed to save model {name}: {e}")