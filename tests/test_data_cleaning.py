import unittest
import pandas as pd
import numpy as np
from io import StringIO
from scripts.data_cleaning import DataCleaner

class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame for testing
        data = {
            'A': [1, 2, 2, 4],
            'B': ['a', 'b', 'b', 'd'],
            'C': ['2023-01-01T12:00:00Z', '2023-01-02T15:30:00Z', '2023-01-02T15:30:00Z', '2023-01-03T10:00:00Z'],
            'D': [np.nan, 1, 2, np.nan]
        }
        self.df = pd.DataFrame(data)
        self.cleaner = DataCleaner(self.df)

    def test_apply_column_prefix_removal(self):
        # Add prefixes to column data
        self.cleaner.df['A'] = 'A_' + self.cleaner.df['A'].astype(str)
        self.cleaner.df['B'] = 'B_' + self.cleaner.df['B']
        modified_df = self.cleaner.apply_column_prefix_removal()
        self.assertTrue(modified_df is not None, "Column prefix removal failed.")
        self.assertFalse(modified_df['A'].str.startswith('A_').any(), "Prefixes not removed.")

    def test_identify_duplicates(self):
        duplicates = self.cleaner.identify_duplicates()
        self.assertEqual(len(duplicates), 2, "Failed to identify duplicate rows.")

    def test_remove_duplicates(self):
        cleaned_df = self.cleaner.remove_duplicates()
        self.assertEqual(len(cleaned_df), 3, "Failed to remove duplicates.")

    def test_check_for_duplicates(self):
        self.cleaner.remove_duplicates()
        remaining = self.cleaner.check_for_duplicates()
        self.assertTrue(remaining.empty, "Failed to remove all duplicates.")

    def test_convert_to_datetime(self):
        cleaned_df = self.cleaner.convert_to_datetime('C', '%Y-%m-%dT%H:%M:%SZ')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df['C']), "Column not converted to datetime.")

    def test_calculate_missing_percentage(self):
        percentage = self.cleaner.calculat_missing_percentage()
        self.assertAlmostEqual(percentage, 16.67, 2, "Incorrect missing value percentage calculation.")

    def test_check_missing_values(self):
        missing_table = self.cleaner.check_missing_values()
        self.assertEqual(missing_table.loc['D', 'Missing Values'], 2, "Incorrect missing values count.")

    def test_check_outliers(self):
        # Outlier test is visual; ensure it does not raise exceptions
        try:
            self.cleaner.check_outliers(['A'])
        except Exception as e:
            self.fail(f"Outlier check raised an exception: {e}")

    def test_drop_high_missing_values(self):
        cleaned_df = self.cleaner.drop_high_missing_values(threshold=50)
        self.assertNotIn('D', cleaned_df.columns, "Column with high missing values not dropped.")

    def test_drop_high_missing_rows(self):
        cleaned_df = self.cleaner.drop_high_missing_rows(threhold=50)
        self.assertEqual(len(cleaned_df), 2, "Rows with high missing values not dropped.")

    def test_get_data_shape(self):
        shape = self.cleaner.get_data_shape()
        self.assertEqual(shape, (4, 4), "Incorrect data shape returned.")

if __name__ == '__main__':
    unittest.main()

# python -m unittest test_data_cleaning.py
