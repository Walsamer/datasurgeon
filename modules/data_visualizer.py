"""
DataVisualizer: A class for generating plots from string and tabular data.
Methods:
  - plot_token_length_distribution(tokens: List[str]) -> matplotlib.figure.Figure
  - (future) plot_numeric_distribution(numbers: List[float])
"""

import io
from collections import Counter
from functools import wraps
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure



def try_dirty(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Figure:
        from shiny import ui
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            ui.notification_show(
                f"Plot error: {e}. NaNs may need cleaning first.",
                type="error",
                duration=12
            )
            import plotly.graph_objects as go
            return go.Figure().update_layout(title=f"Plot failed: {e}")
    return wrapper



class DataVisualizer:
    """
    Helper for creating and exporting matplotlib plots.
    """

    def plot_token_length_distribution(self, tokens: List[str]) -> plt.Figure:
        """
        Generate a bar plot of token length frequencies.
        Parameters
        ----------
        tokens : list of str
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        counts = Counter(len(t) for t in tokens)
        if not counts:
            # return empty figure or raise
            fig, _ = plt.subplots()
            return fig

        min_len, max_len = min(counts), max(counts)
        x = list(range(min_len, max_len + 1))
        y = [counts.get(length, 0) for length in x]

        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set_xlabel("Token Length")
        ax.set_ylabel("Count")
        ax.set_title("Token Length Distribution")
        ax.set_xticks(x)
        fig.tight_layout()
        return fig

    def export_fig_to_svg(self, fig: plt.Figure) -> bytes:
        """
        Export a Matplotlib figure to SVG bytes.
        Returns
        -------
        svg_bytes : bytes
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="svg")
        buf.seek(0)
        return buf.getvalue()

    def plot_numerical_values(self, values: List[float]) -> go.Figure:
        x = [str(v) for v in values]
        y = values

        fig = go.Figure()
        fig.add_bar(x=x, y=y, marker_color="cornflowerblue")

        fig.update_layout(
            title="Extracted Numerical Values",
            yaxis_title="Value",
            bargap=0.2
        )
        return fig



    @try_dirty
    def plot_histogram(
        self,
        df: pd.DataFrame,
        x: str,
        group_by: Optional[str] = None,
        orientation: str = 'v',
        auto_size: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Figure:
        """
        Create a Plotly histogram for column `x` in `df`, optionally grouped.

        Parameters
        ----------
        df : pd.DataFrame
        x : str
        group_by : str or None
        auto_size : bool
            If True, allow Plotly to auto-size; if False, use provided width/height.
        width : int, optional
            Width in pixels when auto_size is False.
        height : int, optional
            Height in pixels when auto_size is False.

        Returns
        -------
        Figure or None
        """
        if df is None or df.empty:
            return go.Figure().update_layout(title="No data available.")

        columns_to_select = [x]
        if group_by and group_by != "None" and group_by in df.columns and group_by != x:
            columns_to_select.append(group_by)

        plot_df = df[columns_to_select].dropna()
        if plot_df.empty:
            return None

        color_arg = group_by if group_by and group_by != "None" else None
        is_horizontal = orientation == 'h'
        axis_param = 'y' if is_horizontal else 'x'
        x_title = "Count" if is_horizontal else x
        y_title = x if is_horizontal else "Count"

        plot_args = {
            axis_param: x,
            'color': color_arg,
            'nbins': 50,
            'orientation': orientation,
            'data_frame': plot_df
        }
        
        fig = px.histogram(**plot_args)
        
        layout_kwargs = {
            "title": {"text": f"Histogram of {x}", "x": 0.5},
            "xaxis_title": x_title,
            "yaxis_title": y_title
        }

        if not auto_size:
            if width is not None:
                layout_kwargs["width"] = width
            if height is not None:
                layout_kwargs["height"] = height
        fig.update_layout(**layout_kwargs)
        return fig

    @try_dirty
    def plot_boxplot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        points: Union[str, bool] = 'all',
        auto_size: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Figure:
        """
        Create a Plotly boxplot for `x` vs `y`, with optional point display.

        Parameters
        ----------
        df : pd.DataFrame
        x : str
        y : str
        points : 'all', 'outliers', 'suspectedoutliers', False
        auto_size : bool
            If True, allow Plotly to auto-size; if False, use provided width/height.
        width : int, optional
        height : int, optional

        Returns
        -------
        Figure or None
        """
        if df is None or df.empty:
            return go.Figure().update_layout(title="No data available.")

        plot_df = df[[x, y]].dropna()
        if plot_df.empty:
            return None

        if isinstance(points, str) and points == 'False':
            points = False

        fig = px.box(
            data_frame=plot_df,
            x=x,
            y=y,
            points=points
        )
        layout_kwargs = {
            "title": {"text": f"Boxplot of {x} vs {y}", "x": 0.5},
            "xaxis_title": x,
            "yaxis_title": y
        }
        if not auto_size:
            if width is not None:
                layout_kwargs["width"] = width
            if height is not None:
                layout_kwargs["height"] = height
        fig.update_layout(**layout_kwargs)
        return fig