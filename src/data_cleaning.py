import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    """
    A class to handle data cleaning operations on a DataFrame.

    Methods:
        - apply_column_prefix_removal: Removes the prefix from columns in the DataFrame.
        - check_data_info: Logs the data types and missing values of the DataFrame.
        - identify_duplicates: Identifies duplicate rows based on specified columns.
        - remove_duplicates: Removes duplicate rows based on specified columns.
        - check_for_duplicates: Checks if any duplicates still exist in the DataFrame.
        - get_data_shape: Returns the shape of the DataFrame.
        - convert_to_datetime: Converts specified columns to datetime format.
    """

    def __init__(self, df):
        """
        Initializes the DataCleaner with a DataFrame.

        :param df: pandas DataFrame to be cleaned.
        """
        self.df = df

    def apply_column_prefix_removal(self):
        """
        Removes the prefix from columns in the DataFrame if the prefix matches any column name.

        :return: The updated DataFrame.
        """
        logging.info("Applying column prefix removal...")
        self.df = self.df.applymap(
            lambda x: x.split('_', 1)[1] if isinstance(x, str) and x.split('_', 1)[0] in self.df.columns else x
        )
        logging.info("Column prefix removal completed.")
        return self.df.sample(5)
     
    def check_data_info(self):
        """
        Logs the data types and missing values of the DataFrame.

        :return: The DataFrame (unchanged).
        """
        logging.info("Checking data types and missing values...")
        logging.info(self.df.info())


    def identify_duplicates(self, subset_columns=None):
        """
        Identifies duplicate rows in the DataFrame.

        :param subset_columns: List of columns to consider for identifying duplicates.
                            If None, all columns are considered.
        :return: DataFrame containing duplicate rows.
        """
        logging.info("Identifying duplicates...")
        
        # Default to all columns if subset_columns is not specified
        if subset_columns is None:
            subset_columns = self.df.columns.tolist()
            logging.info("No subset columns specified. Considering all columns.")
        
        # Identify duplicate rows
        duplicate_rows = self.df[self.df.duplicated(subset=subset_columns, keep=False)]
        
        # Logging the results
        if duplicate_rows.empty:
            logging.info("No duplicate rows found.")
        else:
            logging.info("Number of duplicate rows found: %d", duplicate_rows.shape[0])
        
        return duplicate_rows

    def remove_duplicates(self, subset_columns=None, keep='first'):
        """
        Removes duplicate rows based on specified columns.

        :param subset_columns: List of columns names to consider for removing duplicates.
                               If None, all columns are considered.
        :param keep: Determines which duplicates to keep. Default is 'first'.
        :return: The updated DataFrame.
        """
        logging.info("Removing duplicates based on specified columns...")
        self.df = self.df.drop_duplicates(subset=subset_columns, keep=keep)
        logging.info("Duplicates removed.")
        return self.df

    def check_for_duplicates(self):
        """
        Checks if any duplicates still exist in the DataFrame.

        :return: DataFrame containing any remaining duplicate rows.
        """
        logging.info("Checking for any remaining duplicates...")
        remaining_duplicates = self.df[self.df.duplicated()]
        logging.info("Remaining duplicates: %s", remaining_duplicates.shape)
        return remaining_duplicates

    def get_data_shape(self):
        """
        Returns the shape of the DataFrame.

        :return: Tuple containing the number of rows and columns, and the DataFrame.
        """
        logging.info("Getting data shape...")
        shape = self.df.shape
        logging.info("Data shape: %s", shape)
        return shape
    
    def convert_to_datetime(self, column_name, date_format='%Y-%m-%dT%H:%M:%SZ'):
        """
        Converts a specified column to datetime format.

        :param column_name: The name of the column to be converted.
        :param date_format: The format of the datetime string in the column.
        :return: The updated DataFrame.
        """
        logging.info("Converting column '%s' to datetime format...", column_name)
        try:
            # Ensure the column is treated as a string before conversion
            self.df[column_name] = self.df[column_name].astype(str)
            self.df[column_name] = pd.to_datetime(self.df[column_name], format=date_format)
            logging.info("Column '%s' successfully converted to datetime.", column_name)
        except Exception as e:
            logging.error("Error converting column '%s' to datetime: %s", column_name, str(e))
        return self.df

    # Calculate the percentage of missing values in the data set 
    def calculat_missing_percentage(self):
        # Determine the total number of element in dataframe
        total_elements = np.prod(self.df.shape)

        # Calculate the total numeber of missing values in each columns
        missing_values = self.df.isna().sum()

        # Sum of the total number of missing values
        total_missing = missing_values.sum()

        # Compute the percentage of missing values 
        percentage_missing = (total_missing / total_elements) * 100

        # Print the result, rounded to two decimal
        print(f"The dataset has {round(percentage_missing, 2)}% missing values.")

    # Check missing values
    def check_missing_values(self):
        # check missing values in the dataset 
        missing_values = self.df.isnull().sum()
        missing_percentage = 100 * self.df.isnull().sum() / len(self.df)
        column_data_dtypes = self.df.dtypes
        missing_table = pd.concat([missing_values, missing_percentage, column_data_dtypes], axis=1,
                                keys=['Missing Values',
                                        '% of Total Values',
                                        'Data Types'])
        return missing_table.sort_values('% of Total Values', ascending=False).round(2)


    # Check outlier box
    def check_outliers(self, columns):
        for column in columns:
            plt.figure(figsize=(10,6))
            sns.boxplot(x=self.df[column])
            plt.title(f"Box plot of {column}")
            plt.show()

    # Drop missing values 
    def drop_high_missing_values(self, threshold=50):
        # Missing percentage in columns
        missing_series = self.df.isnull().sum() / len(df) * 100
        # Identify missing values with greater than 50%
        columns_to_drop = missing_series[missing_series > threshold].index
        # Drop columns with high missing values 
        df_cleaned = self.df.drop(columns=columns_to_drop)
        # print the result 
        print(f"Dropped column: list{columns_to_drop}")
        return df_cleaned

    # Drop high missing values 
    def drop_high_missing_rows(self, threhold=50):
        """
        Drop rows with a high percentage of missing values.
        :param df: pandas DataFrame
        :param threshold: percentage threshold for dropping rows (default 50%)
        :return: DataFrame with high-missing rows dropped
        """
        missing_series = self.df.isnull().sum(axis=1) * 100 / len(self.df)
        rows_to_drop = missing_series[missing_series > threhold].index
        df_cleaned = self.df.drop(rows=rows_to_drop)
        # Print the result 
        print(f"Dropped rows: list{rows_to_drop}")
        return df_cleaned