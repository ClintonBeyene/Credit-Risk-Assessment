import numpy as np
import pandas as pd

# Define the function
def extract_features(df, cols, features=None):
    if features is None:
        features = ['Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start',
                'Hour', 'Minute', 'Second']
    
    for col in cols:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%SZ')
        
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
            df[new_col_name] = df.groupby(group)[main_column].transform(agg_type)
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
