# datasurgeon

**datasurgeon** is a precision toolkit for inspecting, cleaning, and prepping messy data — starting with raw strings and expanding toward CSV and DataFrame workflows.  
Ideal for anyone working in data science, machine learning, or ETL pipelines.

**Try the app here**: [datasurgeon on shinyapps.io](https://walsamer-datasurgeon.share.connect.posit.cloud)

## Features

- Extract standalone numbers (ints, floats, scientific notation) from text
- Visualize whitespace, tabs, and control characters
- Detect leading/trailing characters and line breaks
- Identify likely separators in raw text (e.g., commas, pipes, spaces)
- Summary statistics (count, mean, std, quartiles, etc.)
- Dark mode toggle 🌙/☀️
- Built with [Shiny for Python](https://shiny.posit.co/py/)


## Coming soon

- CSV and JSON support
- DataFrame inspection and cleaning
- Export cleaned results
- Regex presets and token-level views


## Why?

Because data is rarely clean.  
**datasurgeon** gives you clarity before modeling begins — helping you diagnose and prepare your data fast.


## Requirements

shiny>=0.6.0
numpy>=1.25.0
html5lib>=1.1

```bash
pip install -r requirements.txt
