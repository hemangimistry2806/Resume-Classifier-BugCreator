# %% [markdown]
# # Part 1b: Exploratory Data Analysis (EDA)
# Input : data/processed/resume_clean.csv   (made by the cleaning file)
# Output: charts in reports/figures/
# Run the cleaning file first. Then run this one cell by cell.

# %%
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from pathlib import Path
try:
    os.chdir(Path(__file__).resolve().parent.parent)
except NameError:
    pass
print("Working folder:", os.getcwd())

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("data/processed/resume_clean.csv")
df["clean_text"] = df["clean_text"].fillna("")
print("Rows:", len(df), "| Classes:", df["label"].nunique())
df.head(3)

# %% [markdown]
# ## 4a. Class distribution
# Question: is the dataset balanced? This decides if we need class_weight='balanced'
# and why we report macro-F1 and not only accuracy.

# %%
counts = df["label"].value_counts()
print(counts)
print("\nLargest class:", counts.index[0], counts.iloc[0])
print("Smallest class:", counts.index[-1], counts.iloc[-1])
print("Imbalance ratio (largest / smallest):", round(counts.max() / counts.min(), 1))

plt.figure(figsize=(10, 8))
sns.barplot(x=counts.values, y=counts.index, color="steelblue")
plt.title("Class distribution of resume categories")
plt.xlabel("Number of resumes")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("reports/figures/01_class_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# **Write 1-2 sentences:** Which classes are big, which are small, how big is the ratio,
# and what does that mean for the model?

# %% [markdown]
# ## 4b. Resume length
# Question: how long is a typical resume? This also decides the maximum sequence
# length for the LSTM/GRU later.

# %%
df["char_count"] = df["clean_text"].str.len()
df["word_count"] = df["clean_text"].str.split().str.len()
print(df[["char_count", "word_count"]].describe().round(1))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df["word_count"], bins=40, kde=True, ax=axes[0], color="teal")
axes[0].set_title("Words per resume")
sns.histplot(df["char_count"], bins=40, kde=True, ax=axes[1], color="darkorange")
axes[1].set_title("Characters per resume")
plt.tight_layout()
plt.savefig("reports/figures/02_resume_length.png", dpi=150)
plt.show()

# %%
# Length by class: do some categories write much longer resumes?
order = df.groupby("label")["word_count"].median().sort_values().index
plt.figure(figsize=(10, 8))
sns.boxplot(data=df, x="word_count", y="label", order=order)
plt.title("Resume length (words) by category")
plt.tight_layout()
plt.savefig("reports/figures/03_length_by_class.png", dpi=150)
plt.show()

# 95th percentile: a good max length for the LSTM/GRU (covers most resumes)
print("95th percentile of words:", int(df["word_count"].quantile(0.95)))

# %% [markdown]
# **Write 1-2 sentences:** typical length, any very short or very long outliers,
# whether some classes differ in length.

# %% [markdown]
# ## 4c. Most frequent words, bigrams and trigrams
# For these charts ONLY we also hide:
# - normal English stopwords (the, and, with ...)
# - template words that appear in every resume and say nothing about the job:
#   company, name, city, state (the dataset hides real names with these words),
#   plus placeholder tokens from our cleaning.
# Finding worth writing in your report: the dataset is anonymised with "company name",
# "city", "state". Our saved clean_text keeps them; we only hide them in the charts.

# %%
TEMPLATE_WORDS = {"company", "name", "city", "state", "emailtoken", "phonetoken", "urltoken"}
STOPS = list(ENGLISH_STOP_WORDS.union(TEMPLATE_WORDS))
TOKEN_PATTERN = r"[a-z0-9.+#]{2,}"      # keeps c++, c#, .net as single tokens


def top_ngrams(corpus, n_gram, k=25):
    vec = CountVectorizer(ngram_range=(n_gram, n_gram), stop_words=STOPS,
                          token_pattern=TOKEN_PATTERN)
    matrix = vec.fit_transform(corpus)
    freq = np.asarray(matrix.sum(axis=0)).ravel()
    return pd.Series(freq, index=vec.get_feature_names_out()).sort_values(ascending=False).head(k)


def plot_top(series, title, filename, color):
    plt.figure(figsize=(9, 7))
    sns.barplot(x=series.values, y=series.index, color=color)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{filename}", dpi=150)
    plt.show()


uni = top_ngrams(df["clean_text"], 1)
plot_top(uni, "Top 25 unigrams", "04_top_unigrams.png", "slateblue")

# %%
bi = top_ngrams(df["clean_text"], 2)
plot_top(bi, "Top 25 bigrams", "05_top_bigrams.png", "seagreen")

tri = top_ngrams(df["clean_text"], 3)
plot_top(tri, "Top 25 trigrams", "06_top_trigrams.png", "indianred")

# %% [markdown]
# **Write 1-2 sentences:** which words dominate, what the bigrams/trigrams show that
# single words do not (for example "customer service", "project management"), and what
# generic noise you noticed. This is the reason Part 2 compares TF-IDF (1,1) with (1,2).

# %% [markdown]
# ## 4d. Word clouds (overall and class-wise)

# %%
try:
    from wordcloud import WordCloud

    def show_cloud(text, title, filename):
        wc = WordCloud(width=900, height=450, background_color="white",
                       stopwords=set(STOPS), collocations=False).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"reports/figures/{filename}", dpi=150)
        plt.show()

    show_cloud(" ".join(df["clean_text"]), "Word cloud: all resumes", "07_wordcloud_all.png")

    # Pick 3 contrasting classes: change these names if you like
    for cls in ["INFORMATION-TECHNOLOGY", "CHEF", "HEALTHCARE"]:
        if cls in df["label"].values:
            show_cloud(" ".join(df.loc[df["label"] == cls, "clean_text"]),
                       f"Word cloud: {cls}", f"08_wordcloud_{cls}.png")
except ImportError:
    print("wordcloud is not installed: py -3.9 -m pip install wordcloud")

# %% [markdown]
# ## 4e. Class-wise vocabulary (top TF-IDF terms per class)
# Question: do the classes use different words? If two classes share almost the same
# top terms, expect the model to confuse them. (Exploration only. In Part 2 the
# vectorizer is fitted on the TRAIN split only.)

# %%
tfidf = TfidfVectorizer(stop_words=STOPS, token_pattern=TOKEN_PATTERN, min_df=3, max_df=0.8)
X = tfidf.fit_transform(df["clean_text"])
terms = np.array(tfidf.get_feature_names_out())

rows = []
for cls in sorted(df["label"].unique()):
    idx = np.where(df["label"].values == cls)[0]
    scores = np.asarray(X[idx].mean(axis=0)).ravel()
    rows.append({"class": cls, "top_terms": ", ".join(terms[scores.argsort()[::-1][:10]])})

class_terms = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 150)
pd.set_option("display.width", 200)
print(class_terms.to_string(index=False))
class_terms.to_csv("reports/class_top_terms.csv", index=False)

# %% [markdown]
# **Write 1-2 sentences:** which classes have very clear vocabulary, and which pairs look
# similar (for example ENGINEERING vs INFORMATION-TECHNOLOGY, or BANKING vs FINANCE vs
# ACCOUNTANT). Those pairs are your predictions for the confusion matrix.

# %% [markdown]
# ## 4f. Noise check

# %%
print("Resumes with an email token:", df["clean_text"].str.contains("emailtoken").sum())
print("Resumes with a phone token :", df["clean_text"].str.contains("phonetoken").sum())
print("Resumes containing 'company name':", df["clean_text"].str.contains("company name").sum())

print("\nShortest 5 resumes:")
print(df.sort_values("word_count").head(5)[["label", "word_count"]])

# %% [markdown]
# ## Checklist before you push to GitHub
# - [ ] 8+ charts saved in reports/figures/
# - [ ] 1-2 sentences written under every chart (in the markdown cells)
# - [ ] Numbers from the cleaning step noted: 2484 rows -> 2481 (1 empty, 2 duplicates)
# - [ ] reports/class_top_terms.csv saved