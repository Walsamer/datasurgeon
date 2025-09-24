<p align="center">
  <img src="https://github.com/Walsamer/datasurgeon/raw/main/assets/data_surgeon_image.png" alt="Data Surgeon Banner" width="600"/>
</p>

# datasurgeon

**datasurgeon** is a precision toolkit for inspecting, cleaning, and prepping messy data — from raw strings all the way to CSV/DataFrame workflows.  
Ideal for anyone working in data science, machine learning, or ETL pipelines.

**Try the app here**: [datasurgeon on shinyapps.io](https://walsamer-datasurgeon.share.connect.posit.cloud)



## Features

- **String Analyzer**
  - Extract standalone numbers (ints, floats, scientific notation) from text
  - Visualize whitespace, tabs, and control characters (CR/LF/TAB/SPACE)
  - Detect leading/trailing characters and report **Case (Leading/Trailing Trim)** + length after trim
  - Tokenize and **plot token-length** (“word-length”)

- **Data Cleaner**
  - Drop rows with missing values or **fill NA** with `0`, **mean**, or **median** (optionally per-column)
  - **Normalize column names** (trim → lowercase → underscores; remove dots)
  - **Trim** whitespace across string columns
  - **Remove duplicate rows**
  - Live **missing-value summary** (count, %, dtype, unique values)

- **Plotting (Shiny Express + Plotly)**
  - **Histogram**: choose **X-Axis**; optional **Group by** (color)
  - **Boxplot**: choose **X-Axis** (categorical) and **Y-Axis** (numeric); toggle underlying points (`all | outliers | suspectedoutliers | False`)
  - **Auto-size plot** toggle; manual **Width/Height** sliders when off
  - Robust error handling: invalid inputs/missing columns show a friendly Plotly figure + UI notification (no blank panes)

- **Download**
  - Export cleaned data as **CSV / JSON / XML** (filename adapts to selected format)

- **Architecture**
  - Modular code in `modules/`: `data_io.py`, `data_cleaner.py`, `data_visualizer.py`, `string_inspector.py`
  - Built with [Shiny for Python (Express)](https://shiny.posit.co/py/) + `shinywidgets`
  - Dark mode toggle 🌙/☀️

## Screenshots

<p align="center">
  <img src="https://github.com/Walsamer/datasurgeon/raw/main/assets/deal_with_missing_values.png" alt="Deal with missing values" width="700"/>
  <br/>
  <em>Handle missing data: drop rows or fill with 0/mean/median; live missing-value summary.</em>
</p>

<p align="center">
  <img src="https://github.com/Walsamer/datasurgeon/raw/main/assets/histogram_grouped_horizontal.png" alt="Histogram grouped by category" width="700"/>
  <br/>
  <em>Histogram with optional Group by (color) and manual sizing controls in darkmode.</em>
</p>

<p align="center">
  <img src="https://github.com/Walsamer/datasurgeon/raw/main/assets/boxplot_underlaying_points.png" alt="Boxplot with points" width="700"/>
  <br/>
  <em>Boxplot with underlying data points: all, outliers, suspected outliers, or none.</em>
</p>

## Why?
Because data is rarely clean. Get a quick overview of your string and dataset, visualize issues fast, clean them, and export the results.  
**datasurgeon** gives you clarity before modeling begins — helping you diagnose and prepare your data quickly.

## Requirements
shiny
shinywidgets
pandas
plotly
matplotlib


```bash
pip install -r requirements.txt
```
Run locally:
```bash
python -m shiny run --reload app.py
```

## Coming soon
- Additional plot types (scatter, heatmap)
- One-click data profiling report (Markdown/HTML)
- Tabular RAG-LLM integration: retrieve-augmented generation over your DataFrame (column-aware prompting, per-row context windows, and guided summaries) to answer questions about your dataset directly inside the app.

## Project structure
```
datasurgeon/
├─ app.py
├─ modules/
│  ├─ data_cleaner.py
│  ├─ data_io.py
│  ├─ data_visualizer.py
│  └─ string_inspector.py
├─ assets/
├─ requirements.txt
└─ readme.md
```
