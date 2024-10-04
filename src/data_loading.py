import numpy as np
import gdown
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def download_file_from_google_drive(file_ids, output_dir, file_names):
    """
    Download a file from Google Drive.

    Parameters:
    file_id (str): The ID of the file to download.
    output_path (st): The path where the file will be saved.
    """
    for file_id, file_name in zip(file_ids, file_names):
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = os.path.join(output_dir, file_name)
        gdown.download(url, output_path, quiet=False)

def load_data(file_path):

    """
    Load data from a CSV file.

    Parameters: 
    file_path (str): The path to the csv file.

    Return:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """

    data = pd.read_csv(file_path)
    return data


def split_data(data, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Parameters:
    data(pd.DataFrame): The data to split.
    test_size (float): The proportion of the data to include in the test split.
    random_state (int): The seed used by random number generator.

    Returns:
    pd.DataFrame, pd.DataFrame: The training and testing Data. 
    """

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data