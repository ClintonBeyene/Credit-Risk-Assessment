import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the percentage of missing values in the data set 
def calculat_missing_percentage(dataframe):
    # Determine the total number of element in dataframe
    total_elements = np.prod(dataframe.shape)

    # Calculate the total numeber of missing values in each columns
    missing_values = dataframe.isna().sum()

    # Sum of the total number of missing values
    total_missing = missing_values.sum()

    # Compute the percentage of missing values 
    percentage_missing = (total_missing / total_elements) * 100

    # Print the result, rounded to two decimal
    print(f"The dataset has {round(percentage_missing, 2)}% missing values.")

# Check missing values
def check_missing_values(df):
    # check missing values in the dataset 
    missing_values = df.isnull().sum()
    missing_percentage = 100 * df.isnull().sum() / len(df)
    column_data_dtypes = df.dtypes
    missing_table = pd.concat([missing_values, missing_percentage, column_data_dtypes], axis=1,
                              keys=['Missing Values',
                                    '% of Total Values',
                                     'Data Types'])
    return missing_table.sort_values('% of Total Values', ascending=False).round(2)


# Check outlier box
def check_outliers(df, columns):
    for column in columns:
        plt.figure(figsize=(10,6))
        sns.boxplot(x=df[column])
        plt.title(f"Box plot of {column}")
        plt.show()

# Drop missing values 
def drop_high_missing_values(df, threshold=50):
    # Missing percentage in columns
    missing_series = df.isnull().sum() / len(df) * 100
    # Identify missing values with greater than 50%
    columns_to_drop = missing_series[missing_series > threshold].index
    # Drop columns with high missing values 
    df_cleaned = df.drop(columns=columns_to_drop)
    # print the result 
    print(f"Dropped column: list{columns_to_drop}")
    return df_cleaned

# Drop high missing values 
def drop_high_missing_rows(df, threhold=50):
    """
    Drop rows with a high percentage of missing values.
    :param df: pandas DataFrame
    :param threshold: percentage threshold for dropping rows (default 50%)
    :return: DataFrame with high-missing rows dropped
    """
    missing_series = df.isnull().sum(axis=1) * 100 / len(df)
    rows_to_drop = missing_series[missing_series > threhold].index
    df_cleaned = df.drop(rows=rows_to_drop)
    # Print the result 
    print(f"Dropped rows: list{rows_to_drop}")
    return df_cleaned