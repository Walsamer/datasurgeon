import io
from pathlib import Path

import pandas as pd
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget
from dotenv import load_dotenv

from modules.string_inspector import StringInspector
from modules.data_io import DataIO
from modules.data_cleaner import DataCleaner
from modules.data_visualizer import DataVisualizer
from modules.llm_chat import ChatModule


raw_data = reactive.value(None)
cleaned_data = reactive.value(None)

ui.page_opts(title="Data Science Toolbox", full_width=True)

with ui.card():
    with ui.layout_columns():
        ui.input_file("file", "Upload CSV/Excel/JSON", accept=[".csv", ".xlsx", ".json"])
        @render.text
        def show_file_size():
            file = input.file()
            if not file:
                return None
            return DataCleaner.human_readable_size(file[0]["size"])
    
    with ui.navset_tab(id="tab", selected="String Analyzer"):
        with ui.nav_panel("String Analyzer"):
            default = "This is a sample string: -123, 234, 3456"
            with ui.layout_sidebar():
                with ui.sidebar(width=300):    
                    ui.input_text_area("astring", "", value=default)
                    ui.input_switch("analyze_metrics", "Analyze Text Metrics", value=True)
                    ui.input_switch("replace_special", "Replace Special Characters")
                    ui.input_switch("extract_numbers", "Extract & Analyze Numbers")
                    ui.input_switch("plot_tokens", "Plot Token Lengths")

                    @render.express
                    def show_switch():
                        if input.extract_numbers():      
                            ui.input_switch("plot_value_distr", "Plot extracted numbers")

                # Overview card - always shown
                with ui.card():
                    ui.card_header("Overview")
                    @render.express
                    def show_overview():
                        inspector = StringInspector(input.astring())
                        ui.help_text("Raw String:")
                        ui.p(inspector.html_escape(), class_="text-box")
                        
                        if input.replace_special():
                            special_chars = inspector.visualize_special_chars()
                            if special_chars != input.astring():
                                ui.help_text("Special Characters Replaced:")
                                ui.p(special_chars, class_="text-box")

                # Text metrics card
                @render.express
                def show_text_metrics():
                    if input.analyze_metrics():
                        inspector = StringInspector(input.astring())
                        metrics = inspector.get_text_metrics()
                        with ui.card():
                            ui.card_header("Text Metrics")
                            with ui.layout_column_wrap():
                                for key, value in metrics.items():
                                    with ui.card():
                                        ui.card_header(key)
                                        ui.p(str(value))

                # Numerical analysis card
                @render.express
                def show_numerical_analysis():
                    if input.extract_numbers():
                        inspector = StringInspector(input.astring())
                        num_metrics = inspector.get_numerical_metrics()
                        
                        if num_metrics:
                            with ui.card():
                                ui.card_header("Numerical Analysis")
                                with ui.layout_column_wrap():
                                    for key, value in num_metrics.items():
                                        with ui.card():
                                            ui.card_header(key)
                                            if isinstance(value, float):
                                                ui.p(f"{value:.3f}")
                                            else:
                                                ui.p(str(value))

                # Token length distribution
                @render.express
                def show_token_plot():
                    if input.plot_tokens():
                        inspector = StringInspector(input.astring())
                        tokens = inspector.tokenize()
                        if tokens:
                            with ui.card():
                                ui.card_header("Token Length Distribution")
                                @render.plot
                                def token_plot():
                                    viz = DataVisualizer()
                                    return viz.plot_token_length_distribution(tokens)

                # Numerical values distribution
                @render.express  
                def show_numerical_plot():
                    # Check if the switch exists and is turned on
                    if hasattr(input, "plot_value_distr") and input.plot_value_distr():
                        inspector = StringInspector(input.astring())
                        nums = inspector.find_numerical()["numbers"]
                        if nums:
                            with ui.card():
                                ui.card_header("Distribution of Numerical Values")
                                @render_widget
                                def value_plot():
                                    viz = DataVisualizer()
                                    return viz.plot_numerical_values([float(n) for n in nums])

        with ui.nav_panel("Data Cleaner"):
            with ui.layout_sidebar():
                with ui.sidebar(width=300):
                    ui.input_selectize("remove_cols", "Remove Columns", choices=[], multiple=True)
                    ui.hr()
                    
                    ui.h5("Deal with Missing Values")
                    ui.input_select(
                        "missing_strategy",
                        "Strategy",
                        choices=["No change", "Drop rows", "Fill with 0", "Fill with mean", "Fill with median"],
                        selected="No change"
                    )
                    ui.input_selectize(
                        "missing_subset",
                        "Apply to columns",
                        choices=[],
                        multiple=True
                    )
                    
                    ui.hr()
                    ui.input_switch("normalize_names", "Normalize column names")
                    ui.input_switch("trim_strings", "Trim whitespace from strings")
                    ui.input_switch("remove_duplicates", "Remove duplicate rows")
                    
                    ui.hr()
                    ui.input_action_button("clean_btn", "Clean Data", class_="btn-primary")
                    ui.input_action_button("reset_btn", "Reset", class_="btn-warning")
                    
                    @render.download(
                        label="Download Cleaned Data",
                        filename=lambda: f"data_cleaned.{'xlsx' if input.download_format() == 'Excel' else input.download_format().lower()}"
                    )
                    def download_data():
                        df = cleaned_data.get()
                        if df is None or df.empty:
                            yield b""
                            return
                        
                        data_io = DataIO(df)
                        format_type = input.download_format()
                        
                        if format_type == "CSV":
                            yield data_io.to_csv().encode("utf-8")
                        elif format_type == "JSON":
                            yield data_io.to_json().encode("utf-8")
                        elif format_type == "XML":
                            yield data_io.to_xml().encode("utf-8")
                        # elif format_type == "MARKDOWN":       #on hold, needs tabulate library.
                        #     yield data_io.to_markdown().encode("utf-8")

                    ui.input_select(
                        "download_format",
                        "Download Format",
                        choices=["CSV", "JSON", "XML", #"MARKDOWN"   #on hold, needs tabulate library.
                        ],
                        selected="CSV"
                    )




                with ui.navset_pill():
                    with ui.nav_panel("Overview"):
                        @render.data_frame
                        def missing_summary():
                            df = cleaned_data.get()
                            if df is not None:
                                cleaner = DataCleaner(df)
                                return cleaner.missing_value_summary()
                            return pd.DataFrame()
                    
                    with ui.nav_panel("Data"):
                        @render.data_frame
                        def show_cleaned():
                            df = cleaned_data.get()
                            if df is not None:
                                return df
                            return pd.DataFrame()


            @reactive.effect
            @reactive.event(input.file)
            def load_file():
                file = input.file()
                if not file:
                    return
                    
                path = file[0]["datapath"]
                ext = Path(path).suffix.lower()
                
                try:
                    if ext == ".csv":
                        loader = DataIO.load_csv(path)
                    elif ext in [".xlsx", ".xls"]:
                        loader = DataIO.load_excel(path)
                    elif ext == ".json":
                        loader = DataIO.load_json(path)
                    else:
                        ui.notification_show(f"Unsupported file type: {ext}", type="error")
                        return
                    
                    df = loader.get_df()
                    raw_data.set(df)
                    cleaned_data.set(df.copy())
                    
                    # Update UI elements
                    cols = df.columns.tolist()
                    ui.update_selectize("remove_cols", choices=cols)
                    ui.update_selectize("missing_subset", choices=cols)
                    ui.update_selectize("x_axis", choices=cols, selected=(cols[0] if cols else None))
                    ui.update_selectize("y_axis", choices=cols, selected=(cols[1] if len(cols) > 1 else (cols[0] if cols else None)))
                    ui.update_select("group_by", choices=["None"] + cols, selected="None")
                    
                    
                except Exception as e:
                    ui.notification_show(f"Error loading file: {e}", type="error", duration=5)

            @reactive.effect
            @reactive.event(input.clean_btn)
            def apply_cleaning():
                df = cleaned_data.get()
                if df is None:
                    return
                    
                cleaner = DataCleaner(df)
                steps = []
                
                # Remove columns
                cols_to_drop = input.remove_cols()
                ex_cols = []
                if cols_to_drop:
                    for cols in cols_to_drop:
                        if cols in df.columns:
                            ex_cols.append(cols)
                    if ex_cols:
                        df = df.drop(columns=ex_cols)
                        cleaner = DataCleaner(df)  
                # Reinitialize with updated df
                # Handle missing values
                strategy = input.missing_strategy()
                subset = input.missing_subset() or None
                
                if strategy == "Drop rows":
                    steps.append({"method": "drop_missing", "kwargs": {"subset": subset}})
                elif strategy == "Fill with 0":
                    steps.append({"method": "fill_missing", "kwargs": {"value": 0}})
                elif strategy == "Fill with mean":
                    if subset:
                        for col in subset:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                mean_val = df[col].mean()
                                steps.append({"method": "fill_missing", "kwargs": {"value": {col: mean_val}}})
                elif strategy == "Fill with median":
                    if subset:
                        for col in subset:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                median_val = df[col].median()
                                steps.append({"method": "fill_missing", "kwargs": {"value": {col: median_val}}})
                
                # Other cleaning operations
                if input.normalize_names():
                    steps.append({"method": "normalize_column_names", "kwargs": {}})
                    
                if input.trim_strings():
                    steps.append({"method": "trim_whitespace", "kwargs": {}})
                    
                if input.remove_duplicates():
                    steps.append({"method": "remove_duplicates", "kwargs": {}})
                
                # Apply pipeline
                if steps:
                    cleaner = cleaner.pipeline(steps)
                
                cleaned_data.set(cleaner.get_df())
                ui.notification_show("Data cleaned successfully!", type="success", duration=3)

            @reactive.effect
            @reactive.event(input.reset_btn)
            def reset_cleaning():
                if raw_data.get() is not None:
                    cleaned_data.set(raw_data.get().copy())
                    ui.update_selectize("remove_cols", selected=[])
                    ui.update_selectize("missing_subset", selected=[])
                    ui.update_select("missing_strategy", selected="No change")
                    ui.notification_show("Data reset to original", type="info", duration=3)

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
                            ui.input_switch("horizontal_hist", "Switch orientation", value=False)
                    @render.express
                    def show_underlaying_points():
                        if input.plot_type() in ["Boxplot"]:
                            df = cleaned_data.get()
                            if df is None:
                                return
                            ui.input_select("show_points", "Show underlying data points", choices = ['all', 'outliers', 'suspectedoutliers', False])
                   
                    # ui.input_select("histfunc", "Aggregation Function", choices=["count", "sum", "avg", "min", "max"], selected="count")

                    ui.input_switch("plot_dimensions", "Auto-size plot", value=True)
                    # Sliders are always present; ignored when Auto-size = True
                    ui.input_slider("x_width", "Width (px)", min=200, max=1400, value=700, step=10)
                    ui.input_slider("y_height", "Height (px)", min=200, max=900, value=450, step=10)
                with ui.card():
                    @render_widget
                    def show_plot():
                        df = cleaned_data.get()
                                                
                        if df is None or (hasattr(df, "empty") and df.empty):
                            import plotly.graph_objects as go
                            return go.Figure().update_layout(title="No data available.")
                        
                        plot_type = input.plot_type()
                        auto = input.plot_dimensions()
                        width = None if auto else input.x_width()
                        height = None if auto else input.y_height()
                        
                        viz = DataVisualizer()
                        
                        if plot_type == "Histogram":
                            try:
                                x = input.x_axis()
                                group = input.group_by()
                                orientation = 'h' if input.horizontal_hist() else 'v'
                                
                                print(f"Histogram inputs - X: {x}, Group: {group}")#debug
                                print(f"Available columns: {df.columns.tolist()}")#debug
                                
                                # Validate x column exists in data
                                if not x:
                                    import plotly.graph_objects as go
                                    return go.Figure().update_layout(title="Select X-Axis for Histogram.")
                                
                                if x not in df.columns:
                                    import plotly.graph_objects as go
                                    return go.Figure().update_layout(title="Column not found. Updating...")
                                
                                return viz.plot_histogram(
                                    df,
                                    x=x,
                                    group_by=(None if group == "None" else group),
                                    orientation=orientation,
                                    auto_size=auto,
                                    width=width,
                                    height=height,
                                )
                                
                            except Exception as e:
                                import plotly.graph_objects as go
                                return go.Figure().update_layout(title="Loading...")
                        
                        elif plot_type == "Boxplot":
                            try:
                                x = input.x_axis()
                                y = input.y_axis()
                                points = input.show_points()
                                
                                # Validate inputs
                                if not x or not y:
                                    import plotly.graph_objects as go
                                    return go.Figure().update_layout(title="Select X-Axis and Y-Axis for Boxplot.")
                                
                                if x not in df.columns or y not in df.columns:
                                    import plotly.graph_objects as go
                                    return go.Figure().update_layout(title="Columns not found. Updating...")
                                
                                pts = False if points == "False" else points
                                return viz.plot_boxplot(
                                    df,
                                    x=x,
                                    y=y,
                                    points=pts,
                                    auto_size=auto,
                                    width=width,
                                    height=height,
                                )
                                
                            except Exception as e:
                                import plotly.graph_objects as go
                                return go.Figure().update_layout(title="Loading...")
                        
                        # Default fallback
                        import plotly.graph_objects as go
                        return go.Figure().update_layout(title="Unknown plot type")

        with ui.nav_panel("Chat with Data"):
            # Initialize chat module
            try:
                chat_module = ChatModule()
                chat_enabled = True
            except Exception as e:
                chat_module = None
                chat_enabled = False
                print(f"Chat module initialization failed: {e}") # For debugging purposes
            
            if chat_enabled:
                # Create chat interface
                chat = ui.Chat(id="chat")
                chat.ui(
                    messages=[{
                        "content": "Hello! How can I assist you with your data today?",
                        "role": "assistant"
                    }],
                    width="100%",
                )
                
                # Store the last processed message index
                last_processed = reactive.value(0)
                
                # Handle messages
                @reactive.effect
                @reactive.event(chat.messages)
                async def respond():
                    all_messages = chat.messages()
                    
                    # Check if there's a new user message
                    if len(all_messages) > last_processed.get():
                        # Look for unprocessed user messages
                        for i in range(last_processed.get(), len(all_messages)):
                            msg = all_messages[i]
                            if msg["role"] == "user":
                                # Process this user message
                                user_text = msg["content"]
                                
                                # Update last processed index
                                last_processed.set(len(all_messages))
                                
                                # Check if we have data loaded
                                df = cleaned_data.get()
                                if df is not None:
                                    chat_module.set_dataframe(df)
                                
                                # Get response from LLM - pass the actual text, not the list!
                                response = chat_module.chat(user_text)
                                
                                # Add response to chat
                                await chat.append_message({
                                    "content": response,
                                    "role": "assistant"
                                })
                                
                                # Update counter again after adding our message
                                last_processed.set(len(all_messages) + 1)
                                
                                # Only process one user message at a time
                                break
                    else:
                        ui.card()
                        ui.card_header("Setup Required"),
                        ui.p("Add GOOGLE_API_KEY to your .env file")
                        





#############---------------------------------------
                    ui.p("Upload a dataset and ask questions about it. ")
            with ui.card():
                ui.p("Note: A google api key is needed for this function to work properly.")
        ui.nav_spacer()
        with ui.nav_control():
            ui.input_dark_mode(id="mode")

ui.tags.style("""
    .text-box {
        background-color: #f1f1f1;
        color: #111;
        padding: 0.75rem;
        border-radius: 0.375rem;
        white-space: pre-wrap;
        font-size: 0.875rem;
    }
    .btn-primary {
        background-color: #007bff;
        color: white;
    }
    .btn-warning {
        background-color: #ffc107;
        color: black;
    }
""")