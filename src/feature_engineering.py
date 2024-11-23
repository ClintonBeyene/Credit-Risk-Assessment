# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
import scorecardpy as sc
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def encode_boolean_features(self):
        boolean_cols = self.df.select_dtypes(include=['bool']).columns
        for col in boolean_cols:
            self.df[col] = self.df[col].astype(int)
        return self.df
    
    def extract_features(self, cols, features=None):
        """
        Extract essential temporal features from datetime columns.

        Parameters:
        cols (list): List of datetime columns to process.
        features (list): List of features to extract. Defaults to optimized temporal features.

        Returns:
        pd.DataFrame: DataFrame with additional temporal features.
        """
        
        # Convert each column in 'cols' to datetime
        for col in cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')  # Handle errors gracefully

        if features is None:
            # Optimized list of necessary features
            features = ['Month', 'Day', 'Dayofweek', 
                        'Is_month_end', 'Is_year_start', 
                        'Hour', 'Minute']
        
        for col in cols:
            for feature in features:
                try:
                    if feature.startswith('Is_'):
                        self.df[col + '_' + feature] = getattr(self.df[col].dt, feature.lower())
                    else:
                        self.df[col + '_' + feature] = getattr(self.df[col].dt, feature.lower())
                except AttributeError as e:
                    print(f"Skipping feature {feature} for column {col}: {e}")
            
            # Elapsed time since epoch
            self.df[col + '_Elapsed'] = self.df[col].view('int64') // 10 ** 9
        
        return self.df

    # Compute correlation matrix
    def correlation_matrix(self):
        # Select only numeric columns
        numeric_data = self.df.select_dtypes(include=['number'])

        correlation_matrix = numeric_data.corr()

        # Visualize correlation to identify redundant features
        plt.figure(figsize=(22, 14))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.show()

        # Drop one of each highly correlated pair based on domain knowledge

    def encode_categorical_features(self):
        # Identify categorical columns (excluding boolean columns)
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Apply One-Hot Encoding to categorical columns
        if len(categorical_cols) > 0:
            encoded = encoder.fit_transform(self.df[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
            
            # Join the encoded columns back to the original DataFrame
            self.df = self.df.join(encoded_df)
            
            # Drop the original categorical columns
            self.df.drop(categorical_cols, axis=1, inplace=True)
        
        # Debug: Check for NaN values
        if self.df.isnull().values.any():
            print("NaN values found in the encoded DataFrame:")
            print(self.df[self.df.isnull().any(axis=1)])
        else:
            print("No NaN values found in the encoded DataFrame.")
        
        return self.df

    def apply_label_encoding(self, columns):
        """
        Applies Label Encoding to the specified columns of the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of column names to label encode.
            
        Returns:
            pd.DataFrame: Updated DataFrame with label-encoded columns.
        """
        encoder = LabelEncoder()
        for col in columns:
            if col in self.df.columns:
                self.df[col] = encoder.fit_transform(self.df[col])
                print(f"Column '{col}' encoded with classes: {list(encoder.classes_)}")
        return self.df

    def normalize_and_standardize(self, normalize_cols, standardize_cols):
        if normalize_cols:
            min_max_scaler = MinMaxScaler()
            self.df[normalize_cols] = min_max_scaler.fit_transform(self.df[normalize_cols])
        
        if standardize_cols:
            standard_scaler = StandardScaler()
            self.df[standardize_cols] = standard_scaler.fit_transform(self.df[standardize_cols])
        
        return self.df

    def balance_classes(self, target_col, over_sampling_ratio=0.5, under_sampling_ratio=1.0):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        oversample = RandomOverSampler(sampling_strategy=over_sampling_ratio, random_state=42)
        X_train_oversampled, y_train_oversampled = oversample.fit_resample(X_train, y_train)

        undersample = RandomUnderSampler(sampling_strategy=under_sampling_ratio, random_state=42)
        X_train_balanced, y_train_balanced = undersample.fit_resample(X_train_oversampled, y_train_oversampled)

        balanced_data = pd.concat(
            [pd.DataFrame(X_train_balanced, columns=X.columns), pd.DataFrame(y_train_balanced, columns=[target_col])],
            axis=1
        )

        return balanced_data, X_test, y_test
    

    def visualize_class_distribution(self, data, target_col):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        sns.countplot(x=target_col, data=self.df, palette='coolwarm')
        plt.title('Original Data Distribution')

        plt.subplot(1, 2, 2)
        sns.countplot(x=target_col, data=data, palette='viridis')
        plt.title('Balanced Data Distribution')

        plt.tight_layout()
        plt.show()

    def calculate_iv_woe(self, data, target_col, bins=10, show_woe=False):
        def calculate_iv_woe(data, target_col, bins, show_woe):
            newDF, woeDF = pd.DataFrame(), pd.DataFrame()
            cols = data.columns

            for ivars in cols[~cols.isin([target_col])]:
                if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
                    binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
                    d0 = pd.DataFrame({'x': binned_x, 'y': data[target_col]})
                else:
                    d0 = pd.DataFrame({'x': data[ivars], 'y': data[target_col]})
                d0 = d0.astype({"x": str})
                d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
                d.columns = ['Cutoff', 'N', 'Events']
                d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
                d['Non-Events'] = d['N'] - d['Events']
                d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
                d['WoE'] = d['% of Non-Events'] / d['% of Events']
                d['IV'] = d['WoE'] * (d['% of Non-Events'] - d['% of Events'])
                d.insert(loc=0, column='Variable', value=ivars)
                temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
                newDF = pd.concat([newDF, temp], axis=0)
                woeDF = pd.concat([woeDF, d], axis=0)
                
                if show_woe:
                    print(d)
            return newDF, woeDF

        iv, woe = calculate_iv_woe(data, target_col, bins, show_woe)
        iv_summary = iv.copy()
        woe_summary = woe.groupby('Variable').agg({'WoE': lambda x: x.tolist()}).reset_index()
        final_summary = pd.merge(iv_summary, woe_summary, on='Variable')

        key_points = []
        for index, row in final_summary.iterrows():
            iv_value = row['IV']
            variable = row['Variable']
            if iv_value < 0.02:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Not useful for prediction.")
            elif 0.02 <= iv_value < 0.1:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Medium predictive power.")
            elif 0.1 <= iv_value < 0.3:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Medium predictive power.")
            elif 0.3 <= iv_value < 0.5:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Strong predictive power.")
            elif iv_value >= 0.5:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Suspicious predictive power. Check for overfitting or data leakage.")

        return final_summary, key_points

    def drop_unused_columns(self, final_summary, threshold=0.02):
        cols_to_drop = final_summary[final_summary['IV'] < threshold]['Variable'].tolist()
        return cols_to_drop