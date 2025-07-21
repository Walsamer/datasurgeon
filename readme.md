<p align="center">
  <img src="https://github.com/Walsamer/datasurgeon/raw/main/assets/data_surgeon_image.png" alt="Data Surgeon Banner" width="600"/>
</p>

# datasurgeon

**datasurgeon** is a precision toolkit for inspecting, cleaning, and prepping messy data — starting with raw strings and expanding toward CSV and DataFrame workflows.  
Ideal for anyone working in data science, machine learning, or ETL pipelines.

**Try the app here**: [datasurgeon on shinyapps.io](https://walsamer-datasurgeon.share.connect.posit.cloud)

## Features

- Extract standalone numbers (ints, floats, scientific notation) from text
- Visualize whitespace, tabs, and control characters
- Detect leading/trailing characters and line breaks
- Identify likely separators in raw text (e.g., commas, pipes, spaces)
- Extraction of numerical values (including integers, fixed-point decimals and scientific notation)
- Summary statistics (count, mean, std, quartiles, etc.) on extracted values
- Plotting of Token length ("word"-length) + Export
- Dark mode toggle 🌙/☀️
- Built with [Shiny for Python](https://shiny.posit.co/py/)


## Coming soon

- DataFrame inspection and cleaning (inlc. plotting)
- CSV and JSON support
- Export cleaned results



## Why?

Because data is rarely clean. Get a quick overview over your string, your dataset and export key-metrics!

**datasurgeon** gives you clarity before modeling begins — helping you diagnose and prepare your data fast.


## Requirements

shiny>=0.6.0
numpy>=1.25.0
matplotlib>=3.8.0
html5lib>=1.1

```bash
pip install -r requirements.txt
