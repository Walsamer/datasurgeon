
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import re

from shiny import reactive
from shiny.express import input, render, ui, expressify
from shinywidgets import render_widget 
from collections import Counter
from pandas.api import types as ptypes
from decimal import Decimal
from functools import wraps

app = expressify()
raw_data = reactive.value(None) #datacleaner
cleaned_data = reactive.value(None) #datacleaner
column_choices = reactive.value([]) #plotting

ui.page_opts(title="Data Science toolbox", full_width=True)


with ui.card():
    with ui.layout_columns():
        ui.input_file("file", "", accept=".csv", placeholder="Upload CSV file")
        @render.text
        def show_file_size():
            if not input.file():
                return None
            return f"Size: {human_readable_size(input.file()[0]['size'])}"

    with ui.navset_tab(id="tab", selected = "Plotting"):
        with ui.nav_panel("String Analyzer"):

            default_string = "Adelie	Torgersen	18.7	181.0	2007 Adelie	Torgersen	17.4	186.0	2007"   #####SAMPLESTRING#####
            input_string = default_string
            with ui.layout_sidebar():
                with ui.sidebar(width=300):
                    ui.input_text_area("astring", "", value= default_string)
                    ui.input_switch(id="text_metrics", label="Analyze Text Metrics", value = True)
                    ui.input_switch(id="replace", label="Replace Special Characters")
                    ui.input_switch(id="extract_num", label="Extract And Analyze Numbers")
                    ui.input_switch(id="plot_token_lengths", label="Plot Token Lengths")
                        
                @render.express
                def analyze():
                    #if input.astring() != default_string: #replace == with !=
                        input_string = input.astring()
                        with ui.card(class_="astring"):#show raw string
                            ui.card_header("Overview")
                            ui.help_text("Raw String:")
                            ui.p(html_escape(input.astring()), class_="text-box")
                            
                            if input.replace() == True: #show string with placeholders
                                if visualize_special_chars(input_string) != input_string:
                                    ui.help_text("String with replaced special characters:")
                                    ui.p(visualize_special_chars(input_string), class_="text-box")
                                else:
                                    _ = ui.notification_show(
                                            f"No replaceable special characters found.",
                                            type="error",
                                            duration=5,
                                        )
                            
                            numerical_result = find_numerical(input_string)
                            
                            if input.extract_num(): #show extracted numbers
                                if numerical_result["numbers"]:
                                    ui.help_text("Extracted Numbers")
                                    ui.p(numerical_result["joined_string"], class_="text-box")
                                else:
                                    _ = ui.notification_show(
                                        "No standalone numbers found.",
                                        type="error",
                                        duration=5,
                                    )
                        if input.text_metrics(): #show text metrics
                            with ui.card(): # Text Metrics Box
                                ui.card_header("Text Metrics")
                                with ui.layout_columns():
                                    with ui.card(): #show length
                                        ui.card_header("Length")
                                        ui.p(string_length(input_string))
                                    
                                    result = detect_trim_characters(input_string)
                                    case = result["case"]

                                    if case == "both":  #start and end trimmed
                                        with ui.card():
                                            ui.card_header("Trimmed Length")
                                            ui.p(result["length_after_trim"])
                                        with ui.card():
                                            ui.card_header("Trimmable end")
                                            ui.p(result["trailing"])
                                        with ui.card():
                                            ui.card_header("Trimmable start")
                                            ui.p(result["leading"])
                                    if case == "leading_only": #only trimmed at the beginning
                                        with ui.card():
                                            ui.card_header("Trimmed Length")
                                            ui.p(result["length_after_trim"])
                                        with ui.card():
                                            ui.card_header("Trimmable start")
                                            ui.p(result["leading"])
                                    if case == "trailing_only": #only trimmed at the end
                                        with ui.card():
                                            ui.card_header("Trimmed Length")
                                            ui.p(result["length_after_trim"])
                                        with ui.card():
                                            ui.card_header("Trimmable end")
                                            ui.p(result["trailing"])
                                    if case == "none": #not trimmed
                                        with ui.card():
                                            ui.card_header("Trimmed Length")
                                            ui.p(result["length_after_trim"])
                                    if "\r" in input_string or "\n" in input_string: #line count
                                        with ui.card():
                                            ui.card_header("Lines")
                                            ui.p(line_count(input_string))
                                    if " " in input_string: #white space count
                                        num_white_spaces = input_string.count(" ")
                                        with ui.card():
                                            ui.card_header("White Spaces")
                                            ui.p(num_white_spaces)
                                    if "\t" in input_string: #tab count
                                        num_tabs = input_string.count("\t")
                                        with ui.card():
                                            ui.card_header("Tabs")
                                            ui.p(num_tabs)
                                    separator_dict = detect_separator(input_string)
                                    if separator_dict:#shows most likely separator and count
                                        with ui.card():
                                            ui.card_header("Separator count")
                                            ui.p(separator_dict["count"])
                                        with ui.card():
                                            ui.card_header("Most Likely Separator")
                                            ui.p(separator_dict["separator"])
                                    if int(len(find_numerical(input_string)) > 0): #shows count of numerical values
                                        with ui.card():
                                                ui.card_header("Standalone Numbers")
                                                ui.p(len(find_numerical(input_string)['numbers']))


                        stats = summarize_numbers(numerical_result["numbers"])

                        if input.extract_num(): #Summary statistics for numerical values
                            with ui.card():
                                ui.card_header(f"Summary Statistics On Extracted Numbers:")

                                with ui.layout_columns():
                                    for k, v in stats.items():
                                        with ui.card():
                                            ui.card_header(k.capitalize())
                                            ui.p(f"{v:.3f}" if isinstance(v, float) else str(v))

                        if input.plot_token_lengths(): #plot token length distribution
                            with ui.card():
                                ui.card_header("Token Length Distribution")
                                @render.plot
                                def token_length_distribution():
                                    tokens = tokenize(input.astring())  # Always pull fresh input!
                                    if not tokens:
                                        return
                                    plot_token_lengths_distr(tokens)
                                @render.download(
                                        label = "Download Plot",
                                        filename= "token_length_distribution.svg"
                                )
                                def download_token_plot():
                                    #recompute the plot
                                    tokens = tokenize(input.astring())  
                                    if not tokens:
                                        return
                                    counts = Counter(len(t) for t in tokens)
                                    min_len, max_len = min(counts), max(counts)
                                    x = list(range(min_len, max_len + 1))
                                    y = [counts.get(length, 0) for length in x]

                                    fig, ax = plt.subplots()
                                    ax.bar(x, y, color="skyblue")
                                    ax.set_xlabel("Token Length")
                                    ax.set_ylabel("Token Count")
                                    ax.set_title("Token Length Distribution")
                                    ax.set_xticks(x)
                                    fig.tight_layout()
                                    
                                    #write to an in‑memory buffer
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="svg")
                                    buf.seek(0)
                                    yield buf.getvalue()

        with ui.nav_panel("Data Cleaner"):
            with ui.layout_sidebar():
                with ui.sidebar(width=300):
                    ui.input_selectize(
                        id = "remove_cols",
                        label = "Remove Columns",
                        choices = [],
                        multiple = True
                    )

                    ui.hr()

                    ui.h5("Deal with NaNs")

                    ui.input_select(
                        id = "chosen_cols",
                        label = "Column scope",
                        choices=["Only numerical", "Only string", "Custom"],
                        selected="Only numerical"
                    )
                    @render.express
                    def show_custom_cols():
                        if input.chosen_cols() == "Custom":
                            ui.input_selectize(
                                id = "cols_to_modify",
                                label = "Custom column list",
                                choices=[],  # dynamically updated
                                multiple=True
                    )

                    @render.express
                    def strategy_input():
                        scope = input.chosen_cols()
                        choices = []

                        if scope == "Only numerical":
                            choices = [
                                "No change",
                                "Replace with 0",
                                "Replace with mean",
                                "Replace with median",
                                "Replace with custom value",
                                "Drop rows"
                            ]
                        elif scope == "Only string":
                            choices = [
                                "No change",
                                "Replace with most common string",
                                "Replace with custom value",
                                "Drop rows"
                            ]
                        elif scope == "Custom":
                            # mixed: show all strategies
                            choices = [
                                "No change",
                                "Replace with 0",
                                "Replace with mean",
                                "Replace with median",
                                "Replace with custom value",
                                "Replace with most common string",
                                "Drop rows"
                            ]

                        ui.input_select(
                            id="missing_values_strat",
                            label="Missing value strategy",
                            choices=choices,
                            selected="No change"
                        )

                    @render.express
                    def show_custom_nan_filler():
                        if input.missing_values_strat() == "Replace with custom value":
                            #with ui.card():
                                    ui.input_text(
                                    id = "custom_nan_filler",
                                    label = "Set your custom value:",
                                    value = ""
                                )
                    
                    ui.hr()
                    
                    # ui.input_selectize(
                    #     id = "transform_cols",
                    #     label = "Columns to transform",
                    #     choices = [],
                    #     multiple = True
                    # )
                    # ui.input_select(
                    #     id = "transform_method",
                    #     label = "Transform Strategy",
                    #     choices = ["No change", "Normalization", "Standardization"],
                    #     selected = "No change"
                    # ) 
                    
                    # ui.hr()


                    ui.input_action_button( # clean button
                        id = "clean_butt",
                        label = "Clean"
                    )

                    @render.download(label="Download cleaned data", filename="data_cleaned.csv")
                    def download_data():
                        df = cleaned_data.get()
                        if cleaned_data.get() is not None:
                            with io.StringIO() as output:
                                cleaned_data.get().to_csv(output, index=False)
                                yield output.getvalue()
                        else:
                            with io.StringIO() as output:
                                pd.DataFrame.to_csv(output, index=False)
                                yield output.getvalue()
                    
                    ui.input_action_button( # reset button
                        id = "reset_butt",
                        label = "Reset"
                    )
                with ui.navset_pill():
                    with ui.nav_panel("Overview"):
                        
                        @render.data_frame
                        def missing_summary():
                            if cleaned_data is None:
                                df = raw_data.get()
                            else:
                                df = cleaned_data.get()
                            if df is not None:
                                na_counts = df.isnull().sum()
                                summary = pd.DataFrame({
                                    "Column": df.columns,
                                    "Missing values": na_counts.values,
                                    "Missing %": np.round((na_counts.values / len(df)) * 100, 2),
                                    "Data type": df.dtypes.values,
                                    "Nr. of unique values": df.nunique().values,
                                    "Zero count": df.apply(lambda col: (col == 0).sum() if pd.api.types.is_numeric_dtype(col) else np.nan),
                                    "Is constant": df.nunique() == 1,
                                    "Most common": df.apply(lambda col: col.mode().iloc[0] if not col.mode().empty else np.nan)
                                })
                                return summary.sort_values(by="Missing values", ascending=False)
                    with ui.nav_panel("Data"):

                        @reactive.effect
                        @reactive.event(input.file)
                        def load_file():
                            file = input.file()
                            if not file:
                                return
                            try:
                                df = pd.read_csv(file[0]["datapath"])
                                raw_data.set(df)
                                cleaned_data.set(df.copy())
                                cols = df.columns.tolist()

                                ui.update_text("fileinfo",)
                                ui.update_selectize("remove_cols", choices=df.columns.tolist(), selected=[])
                                ui.update_selectize("cols_to_modify", choices=df.columns.tolist(), selected=[])
                                ui.update_select("nan_strategy", selected="No change")
                                ui.update_selectize("transform_columns", choices=df.select_dtypes(include='number').columns.tolist(), selected=[])
                                ui.update_selectize("x_axis", choices=cols, selected= cols[0] if cols else None)
                                ui.update_selectize("y_axis", choices=cols, selected=cols[1] if len(cols) > 1 else (cols[0] if cols else None))

                            except Exception as e:
                                ui.notification_show(f"Error loading file: {e}", duration = 5, type="error")

                                raw_data.set(None)
                                cleaned_data.set(None)
                                ui.update_selectize("remove_cols", choices=[], selected=[])

                        @reactive.effect
                        @reactive.event(input.reset_btn)
                        def reset_all():
                            df = raw_data.get()
                            if df is not None:
                                cleaned_data.set(df.copy())
                                ui.update_selectize("remove_cols", selected=[])
                                ui.update_selectize("cols_to_modify", selected=[])
                                ui.update_select("nan_strategy", selected="No change")
                                ui.update_selectize("transform_columns", selected=[])
                                ui.update_select("transform_strategy", selected="No change")
                                ui.update_text("fileinfo", value="")
                                ui.update_select("x_axis", choices=df.columns.tolist())
                                ui.update_select("y_axis", choices=df.columns.tolist())

                            else:
                                cleaned_data.set(None) 
                                ui.update_selectize("remove_cols", selected=[])
                                ui.update_select("nan_strategy", selected="No change")
                                ui.update_selectize("transform_columns", selected=[])
                                ui.update_select("transform_strategy", selected="No change")
                                ui.update_text("fileinfo", value="")

                        @reactive.effect
                        @reactive.event(input.chosen_cols)
                        def update_cols_to_modify_on_scope_change():
                            if input.chosen_cols() == "Custom":
                                df = cleaned_data.get()
                                if df is None:
                                    df = raw_data.get()
                                if df is not None:
                                    ui.update_selectize("cols_to_modify", choices=df.columns.tolist(), selected=[])

                        @render.data_frame
                        def render_df():
                            df = cleaned_data.get()
                            if df is not None:
                                return df
                            df_raw = raw_data.get()
                            if df_raw is not None:
                                return df_raw
                            return pd.DataFrame() #returning an empty dataframe if no data is loaded.
                        
                        @reactive.effect
                        @reactive.event(cleaned_data)
                        def update_plot_inputs():
                            df = cleaned_data.get()
                            if df is not None:
                                cols = df.columns.tolist()
                                ui.update_selectize("x_axis", choices=cols)
                                ui.update_selectize("y_axis", choices=cols)
                                ui.update_select("group_by", choices=["None"] + cols)


                        # Clean button
                        @reactive.effect
                        @reactive.event(input.clean_butt)
                        def clean_data():
                            # always work with a copy of the data so that the original data is not modified
                            df = cleaned_data.get() 
                            if df is None:
                                df = raw_data.get()
                                if df is None:
                                    return
                            df = df.copy()

                            #removing columns
                            cols_to_drop = input.remove_cols()
                            ex_cols = []
                            if cols_to_drop:
                                for cols in cols_to_drop:
                                    if cols in df.columns:
                                        ex_cols.append(cols)
                                if ex_cols:
                                    df = df.drop(columns=ex_cols)

                            #fill missing values (NaNs)
                            strategy = input.missing_values_strat()

                            scope = input.chosen_cols()

                            if scope == "Only numerical":
                                chosen_cols = df.select_dtypes(include="number").columns.tolist()
                            elif scope == "Only string":
                                chosen_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
                            elif scope == "Custom":
                                chosen_cols = list(input.cols_to_modify())

                                if not chosen_cols:
                                    ui.notification_show("No custom columns selected. Please select at least one.", type="warning", duration=5)
                                    return

                            num_cols = df[chosen_cols].select_dtypes(include="number").columns.tolist()
                            str_cols = df[chosen_cols].select_dtypes(include=["object", "string"]).columns.tolist()

                            if strategy == "No change":
                                pass
                            elif strategy == "Replace with 0":
                                if num_cols:
                                    df[num_cols] = df[num_cols].fillna(0)
                                if str_cols:
                                    ui.notification_show("Cannot fill strings with 0. Skipped string columns.", type="warning", duration=5)

                            elif strategy == "Replace with mean":
                                for col in num_cols:
                                    s = df[col]
                                    non_null = s.dropna()
                                    if non_null.empty:
                                        continue
                                    fill_val = non_null.mean()
                                    filled = s.fillna(fill_val)
                                    if ptypes.is_integer_dtype(s.dtype):
                                        df[col] = filled.round(0).astype(int)
                                    else:
                                        decs = max_decimal_places(non_null)
                                        df[col] = filled.round(decs)
                                if str_cols:
                                    ui.notification_show("Mean only applies to numeric columns. Skipped string columns.", type="warning", duration=5)

                            elif strategy == "Replace with median":
                                for col in num_cols:
                                    s = df[col]
                                    non_null = s.dropna()
                                    if non_null.empty:
                                        continue
                                    fill_val = non_null.median()
                                    filled = s.fillna(fill_val)
                                    if ptypes.is_integer_dtype(s.dtype):
                                        df[col] = filled.round(0).astype(int)
                                    else:
                                        decs = max_decimal_places(non_null)
                                        df[col] = filled.round(decs)
                                if str_cols:
                                    ui.notification_show("Median only applies to numeric columns. Skipped string columns.", type="warning", duration=5)

                            elif strategy == "Replace with custom value":
                                try:
                                    val = input.custom_nan_filler()
                                    try:
                                        fill_val = float(val)
                                        if num_cols:
                                            df[num_cols] = df[num_cols].fillna(fill_val)
                                    except ValueError:
                                        pass  # not a float
                                    if str_cols:
                                        df[str_cols] = df[str_cols].fillna(val)
                                except Exception as e:
                                    ui.notification_show(f"Error applying custom fill: {e}", type="error", duration=5)

                            elif strategy == "Replace with most common string":
                                if not str_cols:
                                    ui.notification_show("No string columns to fill.", type="warning", duration=5)
                                for col in str_cols:
                                    df[col] = df[col].fillna(most_common_string(df[col]))

                            elif strategy == "Drop rows":
                                df = df.dropna(subset=chosen_cols)

                                                    

                            



                            cleaned_data.set(df)
        
        with ui.nav_panel("Plotting"):
            with ui.layout_sidebar():
                with ui.sidebar(width=300):
                    ui.input_select("plot_type", "Plot Type", choices=["Histogram", "Boxplot"], selected="Histogram") #, "Scatter", "Heatmap"
                    
                    @render.express
                    def show_x_axis_field():
                        if input.plot_type() in ["Histogram", "Boxplot"]: #, "Scatter", "Heatmap"
                            df = cleaned_data.get()
                            if df is None:
                                return
                            cols = df.columns.tolist()
                            ui.input_selectize("x_axis", "X-Axis", choices = cols, selected = cols[0] if cols else None)

                    @render.express
                    def show_y_axis_field():
                        if input.plot_type() in ["Boxplot"]: #, "Scatter"     // == "Boxplot" or input.plot_type() == "Scatter":
                            df = cleaned_data.get()
                            if df is None:
                                return
                            cols = df.columns.tolist()
                            ui.input_selectize("y_axis", "Y-Axis", choices = cols, selected=cols[1] if len(cols) > 1 else (cols[0] if cols else None))

                    @render.express
                    def show_group_by():
                        if input.plot_type() in ["Histogram"]:
                            df = cleaned_data.get()
                            if df is None:
                                return
                            cols = df.columns.tolist()
                            ui.input_select("group_by", "Group by", choices= ["None"] + cols, selected="None")
                   
                    @render.express
                    def show_underlaying_points():
                        if input.plot_type() in ["Boxplot"]:
                            df = cleaned_data.get()
                            if df is None:
                                return
                            ui.input_select("show_points", "Show underlying data points", choices = ['all', 'outliers', 'suspectedoutliers', False])
                   
                    # ui.input_select("histfunc", "Aggregation Function", choices=["count", "sum", "avg", "min", "max"], selected="count")

                    ui.input_switch("plot_dimensions", "Auto-size plot", value = True)
                    ui.help_text("Scales the plot automatically to fit the window size")
                    @render.express
                    def show_help_text_plotting():
                        if not input.plot_dimensions():
                            ui.help_text("Caution: Large plot dimension may not fit within the visible area")
                    @render.express
                    def show_size_settings():
                        if not input.plot_dimensions():
                            ui.input_slider("x_width", "Set width (px)", min=100, max=800, value=400, step=10)
                            ui.input_slider("y_height", "Set height (px)", min=100, max=800, value=400, step=10)


                with ui.card(style="overflow-x: auto; overflow-y: auto; max-height: 800px;"):


                    @render_widget
                    def chose_right_plot():
                        plot_type = input.plot_type()
                        if plot_type == "Histogram":
                            return plot_histogram()
                        elif plot_type == "Boxplot":
                            return plot_boxplot()
                        # elif plot_type == "Scatter":
                        #     return plot_scatter()
                        # elif plot_type == "Heatmap":
                        #     return plot_heatmap()
                        else:
                            fig = None
                        if fig is None:
                            # Build an empty figure with a “help” annotation
                            fig = go.Figure()
                            fig.add_annotation(
                                text=(
                                    "🚧 Plot unavailable\n"
                                    "• Make sure you’ve uploaded a CSV\n"
                                    "• Selected valid X (and Y) columns\n"
                                    "• Dropped or filled missing values"
                                ),
                                xref="paper", yref="paper",
                                x=0.5, y=0.5,
                                showarrow=False,
                                font=dict(size=14),
                                align="center",
                            )
                            fig.update_layout(
                                xaxis={"visible": False},
                                yaxis={"visible": False},
                                margin=dict(t=20, b=20, l=20, r=20),
                            )

                        return fig



        # with ui.nav_menu("JSON/CSV converter"):
        # with ui.nav_panel("JSON to CSV"):
        #     "Coming soon..."
        # with ui.nav_panel("CSV to JSON"):
        #    "Coming soon..."


        ui.nav_spacer()
        with ui.nav_control():
            ui.input_dark_mode(id="mode")


        @reactive.effect
        @reactive.event(input.make_light)
        def _():
            ui.update_dark_mode("light")


        @reactive.effect
        @reactive.event(input.make_dark)
        def _():
            ui.update_dark_mode("dark")


## functions for String inspector
#string analyzer functions:
def string_length(string):
    """Returns the length of the string."""
    slength = int(len(string))
    return slength

#trimming function
def detect_trim_characters(string)-> dict:
    """Detects leading and trailing whitespace characters in a string."""
    leading_trim = string[:len(string) - len(string.lstrip())]
    trailing_trim = string[len(string.rstrip()):]
    full_string = string 

    def describe(chars):
        symbols = {
            "\t": "[TAB]",
            "\n": "[LF]",
            "\r": "[CR]",
            " ": "[SPACE]"
        }
        
        labels = []
        for c in chars:
            if c in symbols:
                label = symbols[c]
            else:
                label = f"[{repr(c)}]"
            labels.append(label)
        return "".join(labels)
    
    if leading_trim and trailing_trim:
        case = "both"
    elif leading_trim:
        case = "leading_only"
    elif trailing_trim:
        case = "trailing_only"
    else:
        case = "none"

    result = {
        "case": case,
        "length_after_trim": len(string.strip()),
    }
    if leading_trim:
        result["leading"] = describe(leading_trim)
    if trailing_trim:
        result["trailing"] = describe(trailing_trim)

    return result

#show whitespace and other anoying stuff
def visualize_special_chars(string) -> str:
    """Replaces special characters in a string with placeholders."""
    return (
        string.replace("\t", "[TAB]")
         .replace("\n", "[LF]")
         .replace("\r", "[CR]") #falls wer den String mit ner Schreibmaschiene geschrieben hat xD
         .replace(" ", "[SPACE]")
    )

#counts the lines
def line_count(string) -> str:
    """Counts the number of lines in a string."""
    return len(string.splitlines()) + 1

#helper function using html to supress collapsing html/ white space
def html_escape(s: str) -> str:
    """Escapes HTML special characters in a string."""
    return html.escape(s)

#helper function to detect separators
def detect_separator(string: str, candidates=None) -> dict:
    """Detects the most common separator in a string."""
    if candidates is None:
        candidates = [",", ";", "|", "\t", " ", ":", "-", "~", "#"]

    counts = {sep: string.count(sep) for sep in candidates}
    counts = {sep: count for sep, count in counts.items() if count > 0}

    if not counts:
        return None

    #sort by most common
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    best_sep, best_count = sorted_counts[0]

    #dynamic threshold, maybe not useful(?)
    threshold = max(5, len(string) // 20)
    if best_count >= threshold:
        if best_sep == " ":
            return {"count" : best_count, "separator" : "[SPACE]"}
        else: 
            return {"count" : best_count, "separator" : best_sep}
    else:
        return None

#tokenize the string
def tokenize(string: str) -> list[str]:
    """Tokenizes a string into words."""
    #default separator is [SPACE]
    if detect_separator(string) is None or detect_separator(string)["separator"] == "[SPACE]":
        return string.split()
    else:
        return string.split(sep = detect_separator(string)["separator"])

#finding numerical values: 
def find_numerical(string: str) -> dict[str, str | list[str]]:
    """Finds standalone numerical values in a string, including floats and scientific notation."""

    if not string:
        return {"numbers": [], "joined_string": ""}

    
    tokens = re.split(r"[^\d\.\-+eE]+", string) # Split on non-number characters

    def is_valid_num(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    numerical_values = [s for s in tokens if s and is_valid_num(s)]

    return {
        "numbers": numerical_values,
        "joined_string": ", ".join(numerical_values)
    }


#Summary statistics for numerical values
def summarize_numbers(numbers: list[str]) -> dict[str, float]:
    """Calculates summary statistics for a list of numbers."""
    floats = np.array([float(n) for n in numbers], dtype=np.float64)
    if floats.size == 0:
        return {}

    return {
        "count": int(floats.size),
        "min": float(np.min(floats)),
        "max": float(np.max(floats)),
        "mean": float(np.mean(floats)),
        "std": float(np.std(floats, ddof=1)) if floats.size > 1 else 0.0,
        "q1": float(np.percentile(floats, 25)),
        "median": float(np.percentile(floats, 50)),
        "q3": float(np.percentile(floats, 75)),
    }

#Plot token lengths (distribution)
def plot_token_lengths_distr(tokens: list[str]):
    """
    Create a bar plot of the lengths of the tokens, including
    zero‑count bars for missing lengths.
    Returns a Matplotlib Figure.
    """
    #Count the token‑length frequencies
    counts = Counter(len(t) for t in tokens)
    if not counts:
        return
    #Determine the full integer range
    min_len = min(counts.keys())
    max_len = max(counts.keys())
    x = list(range(min_len, max_len + 1))
    y = [counts.get(length, 0) for length in x]

    fig, ax = plt.subplots()
    ax.bar(x, y, color="skyblue")
    ax.set_xlabel("Token Length")
    ax.set_ylabel("Token Count")
    ax.set_title("Token Length Distribution")
    ax.set_xticks(x)
    fig.tight_layout()

    return fig

#newly added function - not implemented in the UI yet
def missing_heatmap(df):
    """Create a heatmap of missing values in a DataFrame."""
    plt.imshow(df.isnull(), aspect='auto', cmap='gray_r')
    plt.title("Missing Value Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()








###functions for Data Cleaner

def human_readable_size(size_bytes: int) -> str:
    """
    Convert a file size in bytes into a human-readable string,
    using binary (KiB, MiB) units.
    """
    units = ["bytes", "KiB", "MiB"]
    idx = 0
    size = float(size_bytes)
    # keep dividing by 1024 while we can
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    # bytes should be an integer, larger units show 2 decimal places
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.2f} {units[idx]}"

def max_decimal_places(series):
    vals = series.dropna().astype(str)
    dec_counts = vals.map(lambda s: len(s.split(".", 1)[1]) if "." in s else 0)
    return int(dec_counts.max()) if not dec_counts.empty else 0

def most_common_string(series: pd.Series) -> str:
    return series.dropna().mode().iloc[0] if not series.dropna().empty else ""

def least_common_string(series: pd.Series) -> str:
    counts = series.dropna().value_counts()
    return counts.index[-1] if not counts.empty else ""

def missing_value_summary(df):
    summary = pd.DataFrame({
        "Dtype": df.dtypes,
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(1),
        "Unique Values": df.nunique()
    })
    return summary.sort_values("Missing %", ascending=False)


###functions for Plotting:


def plot_histogram():
    df = cleaned_data.get()
    if df is None or df.empty:
        return go.Figure().update_layout(title="No data available.")
    
    x = input.x_axis()
    color_argument = input.group_by() if input.group_by() != "None" else None

    # 1) ensure we have a clean frame
    plot_df = df[[x, y]].dropna()
    if plot_df.empty:
        return None


    if color_argument == None:
        pass
    

    layout_kwargs = {
        "title": {"text": f"Histogram of {x}", "x": 0.5},
        "yaxis_title": "Count",
        "xaxis_title": x,
    }

    if not input.plot_dimensions():
        layout_kwargs["width"] = input.x_width()
        layout_kwargs["height"] = input.y_height()

    fig = px.histogram(
        data_frame=df,
        x=input.x_axis(),
        color=color_argument,
        nbins=50
    ).update_layout(**layout_kwargs)

    return fig

def plot_boxplot():
    df = cleaned_data.get()
    if df is None or df.empty:
        return go.Figure().update_layout(title="No data available.")
    
    x = input.x_axis()
    y = input.y_axis()
    points = input.show_points() #always a string
    
    # 1) ensure we have a clean frame
    plot_df = df[[x, y]].dropna()
    if plot_df.empty:
        return None

        # map the string "False" back to the Python False, leave everything else alone
    if points == "False":
        points = False
    else:
        pass


    layout_kwargs = {
        "title": {"text": f"Boxplot of {x}", "x": 0.5},
        "xaxis_title": x,
        "yaxis_title": y,
    }

    if not input.plot_dimensions():
        layout_kwargs["width"] = input.x_width()
        layout_kwargs["height"] = input.y_height()

    fig = px.box(df, x=x, y=y,points=points).update_layout(**layout_kwargs)
    
    return fig



#styling of the string boxes
ui.tags.style("""
    .text-box {
        background-color: #f1f1f1;
        color: #111;
        padding: 0.75rem;
        border-radius: 0.375rem;
        white-space: pre-wrap;
        font-size: 0.875rem;
    }
""")