# %% [markdown]
# ## Steps 1-3 (rewritten): Load, data quality check, cleaning, text preprocessing
# Replace the old Step 1, Step 2 and Step 3 cells with this whole file.
# Everything from "Step 4: EDA" onward stays the same.

# %%
# ---------- IMPORTS AND SETTINGS ----------
import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathlib import Path
try:
    os.chdir(Path(__file__).resolve().parent.parent)
except NameError:
    pass
print("Working folder:", os.getcwd())

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH = "data/Resume.csv"      # change if your file has another name
MIN_WORDS = 20                     # resumes shorter than this are dropped (explained below)

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# %% [markdown]
# ### Step 1: Load the data

# %%
raw = pd.read_csv(DATA_PATH)
print("Shape:", raw.shape)
print("Columns:", list(raw.columns))
raw.head(3)

# %%
# Find the label column and the resume-text column.
# If this picks the wrong ones, set them by hand:  LABEL_COL = "Category"
label_options = ["Category", "category", "Label", "label", "Class", "class"]
text_options = ["Resume_str", "Resume", "resume", "Text", "text", "Resume_text"]

LABEL_COL = next((c for c in label_options if c in raw.columns), None)
TEXT_COL = next((c for c in text_options if c in raw.columns), None)
print("Label column:", LABEL_COL, "| Text column:", TEXT_COL)
assert LABEL_COL and TEXT_COL, "Set LABEL_COL and TEXT_COL by hand (see Columns above)"

# Work on a copy with simple names. `raw` stays untouched for comparison.
df = raw[[TEXT_COL, LABEL_COL]].rename(columns={TEXT_COL: "text", LABEL_COL: "label"}).copy()

# %% [markdown]
# ### Step 2: Data quality check
# We only MEASURE here. Nothing is deleted until the next cell, so you can
# write the real numbers in your report.

# %%
df["text"] = df["text"].astype("string")
df["label"] = df["label"].astype("string").str.strip()

n_missing_text = df["text"].isna().sum()
n_missing_label = df["label"].isna().sum()

text_filled = df["text"].fillna("")
n_empty = (text_filled.str.strip() == "").sum()                 # includes missing text
word_counts = text_filled.str.split().str.len()
n_short = ((word_counts > 0) & (word_counts < MIN_WORDS)).sum()  # short but not empty
n_dup_text = df.duplicated(subset="text").sum()
n_dup_rows = df.duplicated().sum()

# Same text under different labels = label inconsistency
label_per_text = df.dropna().groupby("text")["label"].nunique()
n_conflicts = (label_per_text > 1).sum()

report = pd.DataFrame({
    "check": ["Total rows", "Missing text", "Missing label", "Empty text (incl. missing)",
              f"Short text (< {MIN_WORDS} words)", "Duplicate texts", "Fully duplicate rows",
              "Texts with conflicting labels", "Number of classes"],
    "count": [len(df), n_missing_text, n_missing_label, n_empty, n_short,
              n_dup_text, n_dup_rows, n_conflicts, df["label"].nunique()],
})
report["percent"] = (report["count"] / len(df) * 100).round(2)
report.loc[report["check"] == "Number of classes", "percent"] = np.nan
report

# %%
# Class names: look for spelling or case variants of the same class
print(sorted(df["label"].dropna().unique()))

# %% [markdown]
# **Fill in from the table above:** missing ___ , empty ___ , duplicates ___ , conflicts ___ .

# %% [markdown]
# ### Step 2b: Apply the cleaning rules
# Each rule has a reason you can say out loud:
# 1. Drop rows with missing label: we cannot train or test without the answer.
# 2. Drop empty text: nothing to learn from.
# 3. Drop text under MIN_WORDS words: almost certainly a failed extraction, too little signal.
# 4. Drop duplicate texts: the same resume in train and test would inflate the score (leakage).
# 5. Drop texts that appear under more than one label: the correct answer is unknowable.

# %%
log = []   # keeps a record of rows removed by every rule

def apply_rule(frame, mask_keep, rule_name):
    removed = (~mask_keep).sum()
    log.append({"rule": rule_name, "rows_removed": int(removed)})
    return frame[mask_keep]

df = apply_rule(df, df["label"].notna(), "missing label")
df = apply_rule(df, df["text"].fillna("").str.strip() != "", "empty text")
df = apply_rule(df, df["text"].str.split().str.len() >= MIN_WORDS, f"under {MIN_WORDS} words")

# conflicting labels: remove EVERY copy of a text that has more than one label
conflict_texts = df.groupby("text")["label"].nunique()
conflict_texts = conflict_texts[conflict_texts > 1].index
df = apply_rule(df, ~df["text"].isin(conflict_texts), "conflicting labels")

df = apply_rule(df, ~df.duplicated(subset="text", keep="first"), "duplicate text")

df = df.reset_index(drop=True)
print(pd.DataFrame(log))
print(f"\nRows kept: {len(df)} of {len(raw)}")

# %% [markdown]
# ### Step 3: Text preprocessing
# One function, used for training AND for prediction later. Rules:
# - lowercase
# - remove HTML tags
# - URLs, emails and phone numbers become placeholder words (the exact value is
#   useless, but "this resume has an email" is harmless information)
# - KEEP the characters  + # .  so c++, c#, .net, node.js stay whole
# - collapse extra spaces and line breaks
# - stopwords are NOT removed here (we test their effect in Part 2)

# %%
def clean_text(text) -> str:
    """Raw resume text -> cleaned text. Must be identical in training and prediction."""
    t = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    t = t.lower()

    t = re.sub(r"<[^>]+>", " ", t)                          # 1. HTML tags
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)                  # 2. HTML entities like &amp;
    t = re.sub(r"(https?://|www\.)\S+", " urltoken ", t)    # 3. URLs
    t = re.sub(r"\S+@\S+\.\S+", " emailtoken ", t)          # 4. emails

    # 5. phone numbers (strict patterns, so year ranges like "2015 - 2018" are NOT touched)
    t = re.sub(r"\+\d{1,3}[\s-]?\d{5}[\s-]?\d{5}", " phonetoken ", t)       # +91 98765 43210
    t = re.sub(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b", " phonetoken ", t)  # 123-456-7890
    t = re.sub(r"(?<!\d)\d{10}(?!\d)", " phonetoken ", t)                    # 9876543210

    t = re.sub(r"[^a-z0-9+#.\s]", " ", t)                   # 6. drop symbols, keep + # .
    t = re.sub(r"\.(?![a-z0-9])", " ", t)                   # 7. drop dots not followed by a letter/digit, keep ".net", "node.js", "3.5"
    t = re.sub(r"\s+", " ", t).strip()                      # 8. whitespace and line breaks
    return t

# %%
# TEST: every line should print True. If one prints False, the function has a bug.
tests = {
    "Skills: C++, C#, .NET, Python, SQL":        ["c++", "c#", ".net", "python", "sql"],
    "Built apps with Node.js and ASP.NET":       ["node.js", "asp.net"],
    "Email me: john.doe@mail.com":               ["emailtoken"],
    "Call +91 98765 43210 or 9876543210":        ["phonetoken"],
    "Worked 2015 - 2018 at ABC Corp.":           ["2015", "2018"],   # years must survive
    "<p>Java&nbsp;Developer</p>":                ["java", "developer"],
    "GPA 3.5 and 5+ years experience":           ["3.5", "5+"],
}
for sample, must_have in tests.items():
    out = clean_text(sample)
    print(all(w in out.split() for w in must_have), "|", out)

# %%
df["clean_text"] = df["text"].apply(clean_text)

# Cleaning may leave a few resumes empty or tiny: check and drop them
before = len(df)
df = df[df["clean_text"].str.split().str.len() >= MIN_WORDS].reset_index(drop=True)
print(f"Dropped {before - len(df)} resumes that became too short after cleaning. Final rows: {len(df)}")

# Before / after of one resume (good screenshot for your report)
print("RAW  :", df["text"].iloc[0][:300])
print("CLEAN:", df["clean_text"].iloc[0][:300])

# %%
# Save the cleaned data so Part 2 can start from here
df[["text", "clean_text", "label"]].to_csv("data/processed/resume_clean.csv", index=False)
print("Saved data/processed/resume_clean.csv")
df.head(3)