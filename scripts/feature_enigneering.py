import numpy as np
import pandas as pd

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


