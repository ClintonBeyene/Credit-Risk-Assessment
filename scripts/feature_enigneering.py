import numpy as np
import pandas as pd
import scorecardpy as sc


# Define the function
def extract_features(df, cols, features=None):
    if features is None:
        features = ['Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start',
                'Hour', 'Minute', 'Second']
    
    for col in cols:        
        for feature in features:
            if feature == 'Week':
                df[col + '_' + feature] = df[col].dt.isocalendar().week
            elif feature == 'Is_month_end':
                df[col + '_' + feature] = df[col].dt.is_month_end
            elif feature == 'Is_month_start':
                df[col + '_' + feature] = df[col].dt.is_month_start
            elif feature == 'Is_quarter_end':
                df[col + '_' + feature] = df[col].dt.is_quarter_end
            elif feature == 'Is_quarter_start':
                df[col + '_' + feature] = df[col].dt.is_quarter_start
            elif feature == 'Is_year_end':
                df[col + '_' + feature] = df[col].dt.is_year_end
            elif feature == 'Is_year_start':
                df[col + '_' + feature] = df[col].dt.is_year_start
            elif feature == 'Dayofweek':
                df[col + '_' + feature] = df[col].dt.dayofweek
            elif feature == 'Dayofyear':
                df[col + '_' + feature] = df[col].dt.dayofyear
            else:
                df[col + '_' + feature] = getattr(df[col].dt, feature.lower())
        
        df[col + '_Elapsed'] = df[col].astype(np.int64) // 10 ** 9


def aggregate_data(group_columns, value_columns, agg_functions, df, fillna=True, use_na=False):
    """
    Aggregate data by group_columns.

    Parameters:
        group_columns (list): List of column names to group by.
        value_columns (list): List of column names to aggregate.
        agg_functions (list): List of aggregation functions to apply.
        df (pd.DataFrame): Input DataFrame.
        fillna (bool, optional): Whether to fill missing values. Defaults to True.
        use_na (bool, optional): Whether to use NaN for missing values. Defaults False.

    Returns:
        pd.DataFrame: The input DataFrame with aggregated columns.
    """
    for value_column in value_columns:
        for agg_function in agg_functions:
            aggregated_column_name = '_'.join(group_columns + [value_column, agg_function])
            df[aggregated_column_name] = df.groupby(group_columns)[value_column].transform(agg_function)

            if use_na:
                df.loc[df[aggregated_column_name] == -1, aggregated_column_name] = np.nan

            if fillna:
                df[aggregated_column_name] = df[aggregated_column_name].fillna(-1)

            print(f"'{aggregated_column_name}'", ', ', end='')

    return df

# Frequency Encoding
def encode_FE(df, cols):
    for col in cols:
        nm = col + '_FE'
        df_comb = pd.concat([df[col], df[col]]).value_counts().to_dict()
        df[nm] = df[col].map(df_comb).astype('float32')
        print(nm, ', ', end='')

# Label Encoding
def encode_LE(df, cols):
    for col in cols:
        df_comb = pd.concat([df[col], df[col]], axis=0)
        df_comb, _ = df_comb.factorize(sort=True)
        df[col] = df_comb[:len(df)].astype('int32')
        print(col, ', ', end='')

# Group Aggregation
def encode_AG(df, group, main_columns, aggregations):
    for main_column in main_columns:
        for agg_type in aggregations:
            new_col_name = group + '_' + main_column + '_' + agg_type
            agg_values = df.groupby(group)[main_column].agg(agg_type)
            if agg_values.isnull().all():  # Check if all values are NaN
                df[new_col_name] = 0  # If all NaN, fill with 0
            else:
                df[new_col_name] = df[group].map(agg_values).fillna(0)
            df[new_col_name] = df[new_col_name].astype('float32')
            print(new_col_name, ', ', end='')

# Combine Features
def encode_CB(df, col1, col2):
    nm = col1 + '_' + col2
    df[nm] = df[col1].astype(str) + '_' + df[col2].astype(str)
    print(nm, ', ', end='')

# Group Aggregation NUnique
def encode_AG2(df, group, main_columns):
    for main_column in main_columns:
        new_col_name = group + '_' + main_column + '_ct'
        df[new_col_name] = df.groupby(group)[main_column].nunique().to_dict()
        df[new_col_name] = df[group].map(df[new_col_name]).astype('float32')
        print(new_col_name, ', ', end='')


def rfms_score(df):
    """
    Calculate RFMS score for each customer.
    RFMS: Recency, Frequency, Monetary, Size
    """
    # Assuming TransactionStartTime is already in datetime format
    current_date = df['TransactionStartTime'].max()
    
    customer_metrics = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (current_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': ['sum', 'mean'],  # Monetary and Size
    })
    
    customer_metrics.columns = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg']
    
    # Normalize the metrics
    for col in customer_metrics.columns:
        customer_metrics[f'{col}_Normalized'] = (customer_metrics[col] - customer_metrics[col].min()) / (customer_metrics[col].max() - customer_metrics[col].min())
    
    # Calculate RFMS score (you may adjust the weights as needed)
    customer_metrics['RFMS_Score'] = (
        0.25 * (1 - customer_metrics['Recency_Normalized']) +  # Inverse of Recency
        0.25 * customer_metrics['Frequency_Normalized'] +
        0.25 * customer_metrics['MonetaryTotal_Normalized'] +
        0.25 * customer_metrics['MonetaryAvg_Normalized']
    )
    
    return customer_metrics

def assign_good_bad_label(df, rfms_scores, threshold=0.4):
    """
    Assign good/bad labels based on RFMS score per customer
    """
    # Merge RFMS scores with the original dataframe
    customer_labels = rfms_scores['RFMS_Score'].reset_index()
    customer_labels['label'] = np.where(customer_labels['RFMS_Score'] > threshold, 'good', 'bad')
    
    # Merge labels back to the original dataframe
    df = df.merge(customer_labels[['CustomerId', 'RFMS_Score', 'label']], on='CustomerId', how='left')
    
    return df


def woe_binning(df, target_col, features):
    """
    Perform Weight of Evidence (WoE) binning on specified features.
    
    :param df: DataFrame containing the features and target variable
    :param target_col: Name of the target column
    :param features: List of feature names to perform WoE binning on
    :return: DataFrame with WoE binned features
    """
    bins = sc.woebin(df, y=target_col, x=features)
    woe_df = sc.woebin_ply(df, bins)

    # Extract and print IV for each feature
    iv_values = {}
    for feature in features:
        iv_values[feature] = bins[feature]['total_iv'].values[0]
        print(f"IV for {feature}: {iv_values[feature]}")


    return woe_df, iv_values
