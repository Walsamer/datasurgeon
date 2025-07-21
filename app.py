
import html
import numpy as np
import matplotlib.pyplot as plt
import io

from shiny import reactive
from shiny.express import input, render, ui, expressify
from collections import Counter

app = expressify()

ui.page_opts(title="Data Science toolbox", full_width=True)

with ui.navset_tab(id="tab"):
    with ui.nav_panel("String Analyzer"):

        default_string = "1 1 1 1 22 22 22 222 22 22 222 22  33333 3 3 3 123 123 123 123 431234 1234 1234 1234 1234 0987654 0987654 098765431 9870123098 09817230987123090 0987120987309 0981723098123 0918230981723 09187230198273 09187230918273 01983271039287 1039278 1039287 1039287 1230978 1230 987"   #####SAMPLESTRING#####
        input_string = default_string
        with ui.layout_sidebar():
            with ui.sidebar(width=300):
                ui.input_text_area("astring", "", value= default_string)
                ui.input_switch(id="text_metrics", label="Analyze Text Metrics")
                ui.input_switch(id="replace", label="Replace Special Characters")
                ui.input_switch(id="extract_num", label="Extract And Analyze Numbers")
                ui.input_switch(id="plot_token_lengths", label="Plot Token Lengths", value=True)
                    
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
    """Finds standalone numerical values in a string."""
    def is_valid_float_or_scientific(s: str) -> bool:
        try:
            float(s)
            return all(c.isdigit() or c in ".-+eE" for c in s)
        except ValueError:
            return False

    numerical_values = [word for word in string.split() if is_valid_float_or_scientific(word)]

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

# Plot token lengths (distribution)
import matplotlib.pyplot as plt
from collections import Counter

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
