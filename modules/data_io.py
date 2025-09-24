"""
DataIO: A utility class for loading and exporting pandas DataFrames in CSV, Excel, and JSON formats.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any

class DataIO:
    """
    Parameters:
        df : pd.DataFrame
            The DataFrame to work with.
    """
    def __init__(self, df: pd.DataFrame):
        # Store a copy to avoid mutating the original
        self.df = df.copy()

    @classmethod
    def load_csv(cls, path: Union[str, Path], **kwargs: Any) -> 'DataIO':
        """
        Load a CSV file into a DataIO instance.

        Parameters:
            path : str or Path
                Path to the CSV file.
            **kwargs
                Keyword arguments passed to pd.read_csv.

        Returns:
            DataIO
        """
        df = pd.read_csv(path, **kwargs)
        return cls(df)

    @classmethod
    def load_excel(cls, path: Union[str, Path], sheet_name: Union[str, int] = 0, **kwargs: Any) -> 'DataIO':
        """
        Load an Excel file into a DataIO instance.

        Parameters
        ----------
        path : str or Path
            Path to the Excel file.
        sheet_name : str or int, default 0
            Sheet name or index passed to pd.read_excel.
        **kwargs
            Keyword arguments passed to pd.read_excel.

        Returns
        -------
        DataIO
        """
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        return cls(df)

    @classmethod
    def load_json(cls, path: Union[str, Path], orient: str = 'records', **kwargs: Any) -> 'DataIO':
        """
        Load a JSON file into a DataIO instance.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.
        orient : str, default 'records'
            JSON orientation passed to pd.read_json.
        **kwargs
            Keyword arguments passed to pd.read_json.

        Returns
        -------
        DataIO
        """
        df = pd.read_json(path, orient=orient, **kwargs)
        return cls(df)

    def to_csv(self, path: Optional[Union[str, Path]] = None, **kwargs: Any) -> Optional[str]:
        """
        Export DataFrame to CSV.

        Parameters
        ----------
        path : str or Path, optional
            If provided, writes to file. Otherwise returns CSV string.
        **kwargs
            Keyword arguments passed to pd.DataFrame.to_csv.

        Returns
        -------
        str or None
        """
        csv_str = self.df.to_csv(index=False, **kwargs)
        if path:
            Path(path).write_text(csv_str, encoding='utf-8')
            return None
        return csv_str

    # def to_excel(self, path: Union[str, Path], sheet_name: str = 'Sheet1', **kwargs: Any) -> None:
    #     """
    #     Export DataFrame to an Excel file.

    #     Parameters
    #     ----------
    #     path : str or Path
    #         Path to output .xlsx file.
    #     sheet_name : str, default 'Sheet1'
    #     **kwargs
    #         Keyword arguments passed to pd.DataFrame.to_excel.
    #     """
    #     self.df.to_excel(path, sheet_name=sheet_name, index=False, **kwargs)

    # def to_excel_bytes(self, sheet_name: str = 'Sheet1', **kwargs: Any) -> bytes:
    #     """
    #     Export DataFrame to Excel as bytes (for Shiny downloads).
    #     """
    #     import io
    #     buffer = io.BytesIO()
    #     self.df.to_excel(buffer, sheet_name=sheet_name, index=False, **kwargs)
    #     buffer.seek(0)
    #     return buffer.getvalue()


    def to_json(self, path: Optional[Union[str, Path]] = None, orient: str = 'records', **kwargs: Any) -> Optional[str]:
        """
        Export DataFrame to JSON.

        Parameters
        ----------
        path : str or Path, optional
            If provided, writes to file. Otherwise returns JSON string.
        orient : str, default 'records'
        **kwargs
            Keyword arguments passed to pd.DataFrame.to_json.

        Returns
        -------
        str or None
        """
        json_str = self.df.to_json(orient=orient, **kwargs)
        if path:
            Path(path).write_text(json_str, encoding='utf-8')
            return None
        return json_str

    def to_xml(self, root_name: str = 'data', row_name: str = 'row') -> str:
        """
        Export DataFrame to XML string without external dependencies.
        
        Parameters
        ----------
        root_name : str, default 'data'
            Name of the root element.
        row_name : str, default 'row'
            Name of each row element.
        
        Returns
        -------
        str
        """
        # Manual XML generation - no dependencies needed
        xml_str = f'<?xml version="1.0" encoding="UTF-8"?>\n<{root_name}>\n'
        for _, row in self.df.iterrows():
            xml_str += f'  <{row_name}>\n'
            for col, val in row.items():
                # Handle None/NaN values
                if pd.isna(val):
                    val_str = ''
                else:
                    # Escape special XML characters
                    val_str = str(val).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
                # Make column names XML-safe (replace spaces, special chars)
                safe_col = str(col).replace(' ', '_').replace('-', '_')
                xml_str += f'    <{safe_col}>{val_str}</{safe_col}>\n'
            xml_str += f'  </{row_name}>\n'
        xml_str += f'</{root_name}>'
        return xml_str
    
    ##needs tabulate library, therefore on hold for now
    # def to_markdown(self) -> str:
    #     """Export DataFrame to Markdown table."""
    #     return self.df.to_markdown(index=False)

    def get_df(self) -> pd.DataFrame:
        """
        Return a copy of the underlying DataFrame.
        """
        return self.df.copy()
