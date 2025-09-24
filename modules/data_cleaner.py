"""
DataCleaner: A class for cleaning and preprocessing pandas DataFrames.
Methods include dropping/filling missing values, trimming whitespace, normalizing column names,
removing duplicates, type conversion, and custom cleaning pipelines.
"""

import pandas as pd
from typing import Optional, Union, List, Dict

class DataCleaner:
    """
    Utility to perform data cleaning steps on pandas DataFrames.

    Parameters:
        df : pd.DataFrame
            The DataFrame to clean.
    """
    def __init__(self, df: pd.DataFrame):
        # Work on a copy to avoid mutating the original
        self.df = df.copy()

    def drop_missing(self, subset: Optional[List[str]] = None, how: str = 'any') -> 'DataCleaner':
        """
        Drop rows with missing values.

        Parameters:
            
            subset : list of column names, optional
                Only consider these columns for NA.
            how : 'any' or 'all'
                Drop if any or all values in subset are NA.

        Returns:
           DataCleaner
        """
        self.df = self.df.dropna(subset=subset, how=how)
        return self

    def fill_missing(self, value: Union[int, float, str, Dict[str, Union[int, float, str]]] = 0) -> 'DataCleaner':
        """
        Fill missing values with a constant or dict of column-specific values.

        Parameters:
            value : scalar or dict
                Value(s) to use for filling NA.

        Returns:
            DataCleaner
        """
        self.df = self.df.fillna(value)
        return self

    def trim_whitespace(self, columns: Optional[List[str]] = None) -> 'DataCleaner':
        """
        Strip leading/trailing whitespace from string columns.
        If columns is None, applies to all object-dtype columns.

        Parameters:
            columns : list of column names, optional

        Returns:
            DataCleaner
        """
        cols = columns or self.df.select_dtypes(include='object').columns.tolist()
        for col in cols:
            self.df[col] = self.df[col].astype(str).str.strip()
        return self

    def normalize_column_names(self) -> 'DataCleaner':
        """
        Normalize column names: lowercase, replace spaces and special characters with underscores.

        Returns:
            DataCleaner
        """
        def _clean(name: str) -> str:
            return (
                name.strip()
                    .lower()
                    .replace(' ', '_')
                    .replace('-', '_')
                    .replace('.', '')
            )
        self.df.columns = [_clean(col) for col in self.df.columns]
        return self

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: Union[str, bool] = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows.

        Parameters:
            subset : list of column names, optional
                Columns to identify duplicates.
            keep : 'first', 'last', or False
                Which duplicates to keep.

        Returns:
            DataCleaner
        """
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self

    def convert_dtypes(self, dtype_map: Dict[str, Union[str, type]]) -> 'DataCleaner':
        """
        Convert columns to specified data types.

        Parameters:
            dtype_map : dict
             Mapping of column names to target dtype.

        Returns:
            DataCleaner
        """
        self.df = self.df.astype(dtype_map)
        return self

    def pipeline(self, steps: List[Dict[str, Union[str, Dict]]]) -> 'DataCleaner':
        """
        Apply a custom pipeline of cleaning steps defined by method names and kwargs.

        Parameters:
            steps : list of dicts
                Each dict: {'method': method_name, 'kwargs': {...}}

        Returns:
            DataCleaner
        """
        for step in steps:
            method = getattr(self, step['method'])
            self = method(**step.get('kwargs', {}))
        return self

    def get_df(self) -> pd.DataFrame:
        """
        Return a copy of the cleaned DataFrame.
        """
        return self.df.copy()
    

    @staticmethod
    def human_readable_size(size_bytes: int) -> str:
        """
        Convert a file size in bytes into a human-readable string, using binary units.
        """
        units = ["bytes", "KiB", "MiB"]
        idx = 0
        size = float(size_bytes)
        while size >= 1024 and idx < len(units) - 1:
            size /= 1024
            idx += 1
        if idx == 0:
            return f"{int(size)} {units[idx]}"
        return f"{size:.2f} {units[idx]}"

    @staticmethod
    def max_decimal_places(series: pd.Series) -> int:
        """
        Determine the maximum number of decimal places in numeric-like strings of a Series.
        """
        vals = series.dropna().astype(str)
        dec_counts = vals.map(lambda s: len(s.split(".", 1)[1]) if "." in s else 0)
        return int(dec_counts.max()) if not dec_counts.empty else 0

    @staticmethod
    def most_common_string(series: pd.Series) -> str:
        """
        Return the most frequent non-null string in a Series.
        """
        non_null = series.dropna()
        return non_null.mode().iloc[0] if not non_null.empty else ""

    @staticmethod
    def least_common_string(series: pd.Series) -> str:
        """
        Return the least frequent non-null string in a Series.
        """
        counts = series.dropna().value_counts()
        return counts.index[-1] if not counts.empty else ""

    def missing_value_summary(self) -> pd.DataFrame:
        """
        Summarize missing value counts and percentages, unique counts, and dtypes for the DataFrame.
        """
        na_counts = self.df.isnull().sum()
        summary = pd.DataFrame({
            "Column": self.df.columns,
            "Missing Count": na_counts.values,
            "Missing %": (na_counts.values / len(self.df) * 100).round(2),
            "Data Type": self.df.dtypes.values,
            "Unique Values": self.df.nunique().values
        })
        return summary.sort_values(by="Missing Count", ascending=False)