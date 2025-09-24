"""
String Inspector: A class for inspecting and analyzing raw string data.
Methods cover metrics, trimming, numeric extraction, tokenization and separator detection.
"""


import re
import html
import numpy as np
from typing import List, Dict, Optional, Union

class StringInspector:
    """
    Parameters:
    text : str
        the input string to analyze.
    """

    def __init__(self, text: str):
        self.text = text

    def get_text_metrics(self) -> dict:
        """Return all text metrics as a flat dict of label -> value."""
        m = {}
        trim = self.detect_trim_characters()
        m["Case"] = trim["case"]
        m["Length after trim"] = trim["length_after_trim"]

        if "leading" in trim:
            m["Trimmable start"] = trim["leading"]
        if "trailing" in trim:
            m["Trimmable end"] = trim["trailing"]

        if "\n" in self.text or "\r" in self.text:
            m["Lines"] = self.line_count()
        if " " in self.text:
            m["White Spaces"] = self.whitespace_count()
        if "\t" in self.text:
            m["Tabs"] = self.tab_count()

        sep = self.detect_separator()
        if sep:
            m["Most Likely Separator"] = sep["separator"]
            m["Separator count"] = sep["count"]

        # Standalone numbers count
        nums = self.find_numerical()["numbers"]
        if nums:
            m["Standalone Numbers"] = len(nums)

        return m


    def get_numerical_metrics(self) -> Dict[str, Union[str, int, float]]:
        nums = self.find_numerical()
        num_summary = self.summarize_numbers()

        metrics = {}

        if nums["numbers"]:
            metrics["Extracted Numbers"] = nums["joined_string"]
            for key, value in num_summary.items():
                metrics[f"{key.capitalize()}"] = value

        return metrics

    def length(self) -> int:
        """return total number of characters in the string."""
        return len(self.text)
    
    def count_lines(self) -> int:
        """Counts the number of lines in a string."""
        return len(self.text.splitlines()) +1
    
    def detect_trim_characters(self) -> Dict[str, Union[str, int]]:
        """Detects leading/trailing whitespaces and control characters.
        
        Returns a dict with:
            - case: 'none', 'leading_only', 'trailing_only' or 'both'
            - length_after_trim: length after stripping
            - leading: placeholder labels for leading chars (if any)
            - trailing: placeholder labels for trailing chars (if any)
        """

        leading_trim = self.text[:len(self.text) - len(self.text.lstrip())]
        trailing_trim = self.text[len(self.text.rstrip()):]

        def _label(chars: str) -> str:
            symbols = {"\t": "[TAB]", "\n": "[LF]", "\r": "[CR]", " ": "[SPACE]"}
            return "".join(symbols.get(c, f"[{repr(c)}]") for c in chars)
        
        if leading_trim and trailing_trim:
            case = "both"
        elif leading_trim:
            case = "leading_only"
        elif trailing_trim:
            case = "trailing_only"
        else:
            case = "none"
        
        result: Dict[str, Union[str, int]] = {
            "case": case,
            "length_after_trim": len(self.text.strip())
            }

        if leading_trim:
            result["leading"] = _label(leading_trim)
        if trailing_trim:
            result["trailing"] = _label(trailing_trim)

        return result

    def visualize_special_chars(self) -> str:
        """Replace tabs, newlines, carraige returns and spaces with placeholders"""
        return (
            self.text
            .replace("\t", "[TAB]")
            .replace("\n", "[LF]")
            .replace("\r", "[CR]") #falls wer den String mit ner Schreibmaschiene geschrieben hat xD
            .replace(" ", "[SPACE]")
            )
    
    def detect_separator(self, candidates: Optional[List[str]]= None) -> Optional[Dict[str, Union[str, int]]]:
        """
        Detects the most common separator in a string.
    
        Parameters:
            candidates: list of str, optional
                Characters to consider as separators.
        Returns:
            dict with 'separator' and 'count' or None if no strong candidate.
        """

        if candidates is None:
            candidates = [",", ";", "|", "\t", " ", ":", "-", "~", "#"]

        counts = {sep: self.text.count(sep) for sep in candidates}
        counts = {sep: count for sep, count in counts.items() if count > 0}

        if not counts:
            return None

        #sort by most common
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        best_sep, best_count = sorted_counts[0]

        #dynamic threshold, maybe not useful(?)
        threshold = max(5, len(self.text) // 20)
        if best_count >= threshold:
            if best_sep == " ":
                return {"count" : best_count, "separator" : "[SPACE]"}
            else: 
                return {"count" : best_count, "separator" : best_sep}
        else:
            return None
    
    def tokenize(self) -> List[str]:
        """Tokenizes a string into words."""
        #default separator is [SPACE]
        sep_info = self.detect_separator()

        if sep_info is None or sep_info["separator"] == "[SPACE]":
            return self.text.split()
        else:
            return self.text.split(sep = sep_info["separator"])

    def find_numerical(self) -> dict[str, str | list[str]]:
        """
        Finds standalone numerical values in a string, 
        including floats and scientific notation.
        """

        if not self.text:
            return {"numbers": [], "joined_string": ""}

        
        tokens = re.split(r"[^\d\.\-+eE]+", self.text) # Split on non-number characters

        def is_valid_num(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False

        numerical_values = [token for token in tokens if token and is_valid_num(token)]

        return {
            "numbers": numerical_values,
            "joined_string": ", ".join(numerical_values)
        }
    

    def summarize_numbers(self) -> Dict[str, float]:
        """Compute summary statistics for extracted numerical values."""
        nums = self.find_numerical()["numbers"]
        arr = np.array([float(n) for n in nums], dtype=np.float64)
        if arr.size == 0:
            return {}

        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "q1": q1,
            "median": float(np.median(arr)),
            "q3": q3,
        }

    # Add these methods to your StringInspector class:

    def line_count(self) -> int:
        """Counts the number of lines in the string."""
        return len(self.text.splitlines()) + 1

    def whitespace_count(self) -> int:
        """Counts the number of spaces in the string."""
        return self.text.count(" ")

    def tab_count(self) -> int:
        """Counts the number of tabs in the string."""
        return self.text.count("\t")


    def html_escape(self) -> str:
        """
        Escape HTML special characters in a string.
        """
        return html.escape(self.text)