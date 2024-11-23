import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scorecardpy as sc

class DefaultEstimator:
    def __init__(self, df):
        """
        Initialize the processor with a DataFrame.
        """
        self.df = df
        self.rfms = None
    
    def calculate_rfms(self):
        """
        Calculate Recency, Frequency, Monetary, and Sociality metrics.
        """
        # Convert `TransactionStartTime` to datetime and make it timezone-naive
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime']).dt.tz_localize(None)

        # Get the current timestamp as timezone-naive
        now = pd.Timestamp.utcnow().replace(tzinfo=None)

        # Calculate Recency (days since last transaction)
        self.df['LastAccess'] = self.df.groupby('CustomerId')['TransactionStartTime'].transform('max')
        self.df['Recency'] = (now - self.df['LastAccess']).dt.days

        # Calculate Frequency (number of transactions)
        self.df['Frequency'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')

        # Calculate Monetary (total transaction amount)
        self.df['Monetary'] = self.df.groupby('CustomerId')['Amount'].transform('sum')

        # Placeholder for Sociality (e.g., standard deviation of transaction amounts)
        self.df['Sociality'] = self.df.groupby('CustomerId')['Amount'].transform('std').fillna(0)

        # Combine RFMS metrics into a DataFrame
        self.rfms = self.df[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Sociality']].drop_duplicates()

        return self.rfms

    def display_header(self):
        """
        Display the header of the RFMS DataFrame.
        """
        print(self.rfms.head())
    
    def plot_rfms(self):
        """
        Plot Recency vs. Frequency and Frequency vs. Monetary.
        """
        sns.scatterplot(data=self.rfms, x='Recency', y='Frequency')
        plt.title("Recency vs. Frequency")
        plt.show()
        
        sns.scatterplot(data=self.rfms, x='Frequency', y='Monetary')
        plt.title("Frequency vs. Monetary")
        plt.show()
    
    """
    def calculate_rfms_score(self):
        
        # Calculate the RFMS score based on weighted metrics.
        
        self.rfms['RFMS_Score'] = (
            (1 / (1 + self.rfms['Recency'])) * 0.4 +  # Lower Recency → Higher score
            (self.rfms['Frequency'] / self.rfms['Frequency'].max()) * 0.3 +  # Normalize Frequency
            (self.rfms['Monetary'] / self.rfms['Monetary'].max()) * 0.2 +  # Normalize Monetary
            (self.rfms['Sociality'] / self.rfms['Sociality'].max()) * 0.1  # Normalize Sociality
        )
    """
    
    def calculate_rfms_score(self):
        """
        Calculate RFMS scores after normalizing RFMS metrics.
        """
        from sklearn.preprocessing import MinMaxScaler

        # Normalize RFMS metrics
        rfms_columns = ['Recency', 'Frequency']
        scaler = MinMaxScaler()
        self.rfms[rfms_columns] = scaler.fit_transform(self.rfms[rfms_columns])

        # Calculate RFMS Score
        self.rfms['RFMS_Score'] = (
            (1 - self.rfms['Recency']) * 0.4 +  # Lower Recency → Higher score
            self.rfms['Frequency'] * 0.3 +
            self.rfms['Monetary'] * 0.2 +
            self.rfms['Sociality'] * 0.1
        )

    def assign_risk_labels(self):
        """
        Assign risk labels based on RFMS score thresholds.
        """
        # Calculate thresholds for RFMS scores
        high_threshold = self.rfms['RFMS_Score'].quantile(0.75)
        low_threshold = self.rfms['RFMS_Score'].quantile(0.5)

        # Assign 'Good' or 'Bad' based on the thresholds
        self.rfms['Risk_Label'] = self.rfms['RFMS_Score'].apply(
            lambda x: 'Good' if x >= low_threshold else 'Bad'
        )
        return self.rfms

    def visualize_risk_label_distribution(self):
        """
        Visualize the distribution of risk labels.
        """
        sns.countplot(data=self.rfms, x='Risk_Label')
        plt.title("Distribution of Risk Labels")
        plt.show()

    def plot_woe_binning(self, bins, feature):
        """
        Plot WoE binning results for a specific feature using scorecardpy.
        """
        plt.figure(figsize=(10, 6))
        sc.woebin_plot(bins[feature])
        plt.title(f'WoE Binning Plot for {feature}')
        plt.tight_layout()
        plt.show()

    def perform_woe_binning(self):
        """
        Perform WOE binning on the RFMS metrics using scorecardpy and visualize results.
        """
        # Convert 'Risk_Label' to binary numeric values: 'Good' → 1, 'Bad' → 0
        self.rfms['Risk_Label'] = self.rfms['Risk_Label'].map({'Good': 1, 'Bad': 0})

        # Prepare data for scorecardpy
        data = self.rfms[['Recency', 'Frequency', 'Monetary', 'Sociality', 'Risk_Label']]

        # Perform WoE binning using scorecardpy
        bins = sc.woebin(data, y='Risk_Label')

        # Plot WoE for each metric
        for feature in ['Recency', 'Frequency', 'Monetary', 'Sociality']:
            self.plot_woe_binning(bins, feature)

    
    def merge_data(self):
        """
        Merge RFMS with the original DataFrame for predictive modeling, avoiding duplicate columns.
        """
        # Ensure `CustomerId` is unique in `rfms`
        self.rfms = self.rfms.drop_duplicates(subset='CustomerId', keep='first')

        # Drop existing RFMS columns from the original DataFrame if they exist
        rfms_columns = ['Recency', 'Frequency', 'Monetary', 'Sociality', 'RFMS_Score', 'Risk_Label']
        self.df = self.df.drop(columns=[col for col in rfms_columns if col in self.df.columns], errors='ignore')

        # Merge with RFMS DataFrame
        self.df = pd.merge(self.df, self.rfms, on='CustomerId', how='left')

        return self.df.sample(5)
    
    def get_processed_data(self):
        """
        Return the processed DataFrame after merging RFMS metrics.
        """
        if self.df is None:
            raise ValueError("Data has not been processed yet. Run the required methods first.")
        return self.df
    
    def iv_woe_summary(self, data, target, bins=10, show_woe=False):
        """
        Calculate IV and WoE for each feature in the dataset and provide a summary with key points and suggestions.
        
        Parameters:
        - data: DataFrame containing the dataset.
        - target: The target variable column name.
        - bins: Number of bins to use for binning continuous variables.
        - show_woe: Boolean to indicate whether to print the WoE table for each feature.
        
        Returns:
        - final_summary: DataFrame with the Variable, IV, and WoE for each feature.
        - key_points: List of key points and suggestions based on the IV values.
        """
        def calculate_iv_woe(data, target, bins, show_woe):
            newDF, woeDF = pd.DataFrame(), pd.DataFrame()
            cols = data.columns
            
            for ivars in cols[~cols.isin([target])]:
                if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
                    binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
                    d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
                else:
                    d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
                d0 = d0.astype({"x": str})
                d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
                d.columns = ['Cutoff', 'N', 'Events']
                d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
                d['Non-Events'] = d['N'] - d['Events']
                d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
                d['WoE'] = np.log(d['% of Non-Events'] / d['% of Events'])
                d['IV'] = d['WoE'] * (d['% of Non-Events'] - d['% of Events'])
                d.insert(loc=0, column='Variable', value=ivars)
                temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
                newDF = pd.concat([newDF, temp], axis=0)
                woeDF = pd.concat([woeDF, d], axis=0)
                
                if show_woe:
                    print(d)
            return newDF, woeDF
        
        # Calculate IV and WoE
        iv, woe = calculate_iv_woe(data, target, bins, show_woe)
        
        # Summarize the results
        iv_summary = iv.copy()
        woe_summary = woe.groupby('Variable').agg({'WoE': lambda x: x.tolist()}).reset_index()
        
        # Merge the IV and WoE summaries
        final_summary = iv_summary.merge(woe_summary, on='Variable')
        
        # Generate key points and suggestions
        key_points = []
        for index, row in final_summary.iterrows():
            iv_value = row['IV']
            variable = row['Variable']
            if iv_value < 0.02:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Not useful for prediction.")
            elif 0.02 <= iv_value < 0.1:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Weak predictive power.")
            elif 0.1 <= iv_value < 0.3:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Medium predictive power.")
            elif 0.3 <= iv_value < 0.5:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Strong predictive power.")
            elif iv_value >= 0.5:
                key_points.append(f"{variable}: IV = {iv_value:.6f} - Suspicious predictive power. Check for overfitting or data leakage.")
        
        return final_summary, key_points

    
    def drop_unused_columns(self, final_summary, threshold=0.02):
        """
        Drop columns with IV below the threshold.
        
        Parameters:
        - final_summary (pd.DataFrame): The DataFrame containing IV values for each feature.
        - threshold (float): The threshold for IV to consider a feature useful for prediction.
        
        Returns:
        - list: List of columns to drop.
        """
        # Find columns with IV less than the threshold
        cols_to_drop = final_summary[final_summary['IV'] < threshold]['Variable'].tolist()
        return cols_to_drop