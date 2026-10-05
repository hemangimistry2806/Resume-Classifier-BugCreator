"""Resume text preprocessing.

ONE function, used for training AND for prediction, so both always see identical text.
Keeps + # . so c++, c#, .net, node.js survive. Does not remove stopwords.
"""
import re

import numpy as np


def clean_text(text) -> str:
    """Raw resume text -> cleaned text."""
    t = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    t = t.lower()

    t = re.sub(r"<[^>]+>", " ", t)                          # 1. HTML tags
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)                  # 2. HTML entities like &amp;
    t = re.sub(r"(https?://|www\.)\S+", " urltoken ", t)    # 3. URLs
    t = re.sub(r"\S+@\S+\.\S+", " emailtoken ", t)          # 4. emails

    # 5. phone numbers (strict patterns, so year ranges like "2015 - 2018" are NOT touched)
    t = re.sub(r"\+\d{1,3}[\s-]?\d{5}[\s-]?\d{5}", " phonetoken ", t)        # +91 98765 43210
    t = re.sub(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b", " phonetoken ", t)  # 123-456-7890
    t = re.sub(r"(?<!\d)\d{10}(?!\d)", " phonetoken ", t)                    # 9876543210

    t = re.sub(r"[^a-z0-9+#.\s]", " ", t)                   # 6. drop symbols, keep + # .
    t = re.sub(r"\.(?![a-z0-9])", " ", t)                   # 7. drop dots not followed by a letter/digit
    t = re.sub(r"\s+", " ", t).strip()                      # 8. whitespace and line breaks
    return t