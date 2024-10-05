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