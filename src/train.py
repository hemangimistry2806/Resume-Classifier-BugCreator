"""Part 2: split, TF-IDF features, ML models, evaluation, saved pipeline.

Run from anywhere:   py -3.9 src\\train.py
Needs: data/processed/resume_clean.csv   (made by notebooks/01_data_cleaning_eda.py)

What it does, in order
  1. Load the cleaned data (RAW text column + labels)
  2. Stratified split 70 / 15 / 15  (train / validation / test)  BEFORE any vectorizer is fitted
  3. Compare models on the VALIDATION set only:
       Naive Bayes, Logistic Regression, Linear SVM  x  TF-IDF (1,1) vs (1,2)
       + a stopword on/off test
  4. Pick the best by validation macro-F1
  5. Evaluate that ONE model on the TEST set (first and only time the test set is used)
  6. Save: comparison table, per-class report, confusion matrix, wrong predictions,
     top terms per class, and the full pipeline (cleaning + TF-IDF + model)
"""
import os
import sys
import time
import warnings
from pathlib import Path

# --- always work from the repo root, so paths work from any folder ---
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("Working folder:", os.getcwd())

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.clean_text import clean_text

warnings.filterwarnings("ignore")
if "--show" not in sys.argv:
    matplotlib.use("Agg")          # save charts without popping windows (use --show to see them)

RANDOM_STATE = 42
TOKEN_PATTERN = r"[a-z0-9.+#]{2,}"   # keeps c++, c#, .net as single tokens
TEMPLATE_WORDS = {"company", "name", "city", "state", "emailtoken", "phonetoken", "urltoken"}
STOPS = list(ENGLISH_STOP_WORDS.union(TEMPLATE_WORDS))

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =====================================================================
# 1. LOAD
# =====================================================================
df = pd.read_csv("data/processed/resume_clean.csv")
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
X_raw = df["text"]            # RAW text: the pipeline cleans it itself (same as at prediction time)
y = df["label"]
print(f"Rows: {len(df)} | Classes: {y.nunique()}")

# =====================================================================
# 2. SPLIT (before fitting anything). stratify keeps class proportions equal in every part
# =====================================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")


# =====================================================================
# 3. MODEL COMPARISON ON VALIDATION
# =====================================================================
def make_pipeline(model, ngram=(1, 1), stop_words=None):
    """Raw resume text -> clean_text -> TF-IDF -> classifier.  One object, used everywhere."""
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,         # our cleaning function (also lowercases)
        token_pattern=TOKEN_PATTERN,
        ngram_range=ngram,
        min_df=2,                        # ignore words seen in only 1 resume
        max_df=0.9,                      # ignore words in > 90% of resumes
        sublinear_tf=True,               # log-scaled term frequency, usually helps text
        stop_words=stop_words,
    )
    return Pipeline([("tfidf", tfidf), ("clf", model)])


def models():
    return {
        "NaiveBayes": MultinomialNB(alpha=0.1),
        "LogReg": LogisticRegression(max_iter=1000, C=10, class_weight="balanced"),
        "LinearSVM": LinearSVC(C=1.0, class_weight="balanced"),
    }


experiments = []
for ngram in [(1, 1), (1, 2)]:
    for name, model in models().items():
        experiments.append((name, ngram, None, model))
# stopword test: does removing stopwords help? (the guide says test it, don't assume)
experiments.append(("LinearSVM", (1, 2), "stopwords removed", LinearSVC(C=1.0, class_weight="balanced")))

rows = []
fitted = {}
for name, ngram, stop_label, model in experiments:
    label = f"{name} | ngram={ngram} | {stop_label or 'stopwords kept'}"
    t0 = time.time()
    pipe = make_pipeline(model, ngram, STOPS if stop_label else None)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    rows.append({
        "model": name, "ngram_range": str(ngram), "stopwords": stop_label or "kept",
        "val_accuracy": accuracy_score(y_val, pred),
        "val_macro_f1": f1_score(y_val, pred, average="macro"),
        "val_weighted_f1": f1_score(y_val, pred, average="weighted"),
        "seconds": round(time.time() - t0, 1),
    })
    fitted[label] = pipe
    print(f"done: {label}  macro-F1={rows[-1]['val_macro_f1']:.3f}")

results = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
results.to_csv("reports/model_comparison.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== VALIDATION RESULTS (sorted by macro-F1) ===")
print(results.round(4).to_string(index=False))

# =====================================================================
# 4. PICK THE BEST (by validation macro-F1, not by accuracy alone)
# =====================================================================
best_row = results.iloc[0]
best_key = (f"{best_row['model']} | ngram={best_row['ngram_range']} | "
            f"{'stopwords removed' if best_row['stopwords'] != 'kept' else 'stopwords kept'}")
best_pipe = fitted[best_key]
print("\nBest model on validation:", best_key)

# =====================================================================
# 5. FINAL TEST EVALUATION (test set used ONCE, only for the chosen model)
# =====================================================================
test_pred = best_pipe.predict(X_test)
print("\n=== TEST RESULTS ===")
print("Accuracy   :", round(accuracy_score(y_test, test_pred), 4))
print("Macro-F1   :", round(f1_score(y_test, test_pred, average="macro"), 4))
print("Weighted-F1:", round(f1_score(y_test, test_pred, average="weighted"), 4))

report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
report_df.to_csv("reports/test_classification_report.csv")
print("\n" + classification_report(y_test, test_pred, zero_division=0))

# confusion matrix (rows = actual class, each row sums to 1 -> shows recall per class)
labels_sorted = sorted(y.unique())
fig, ax = plt.subplots(figsize=(15, 13))
ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred, labels=labels_sorted, normalize="true",
    xticks_rotation=90, cmap="Blues", values_format=".1f", colorbar=False, ax=ax)
ax.set_title(f"Confusion matrix on test set (row-normalised)\n{best_key}")
plt.tight_layout()
plt.savefig("reports/figures/09_confusion_matrix.png", dpi=130)
if "--show" in sys.argv:
    plt.show()
plt.close()

# =====================================================================
# 6. ERROR ANALYSIS DATA, TOP TERMS, SAVE PIPELINE
# =====================================================================
# 6a. wrong predictions with a confidence score and a short preview
if hasattr(best_pipe, "predict_proba"):
    scores = best_pipe.predict_proba(X_test).max(axis=1)
    score_name = "confidence"
else:
    scores = best_pipe.decision_function(X_test).max(axis=1)
    score_name = "decision_score"

errors = pd.DataFrame({
    "actual": y_test.values,
    "predicted": test_pred,
    score_name: np.round(scores, 3),
    "word_count": X_test.str.split().str.len().values,
    "preview": X_test.str.replace(r"\s+", " ", regex=True).str[:200].values,
})
errors = errors[errors["actual"] != errors["predicted"]].sort_values(score_name, ascending=False)
errors.to_csv("reports/test_errors.csv", index=False)
print(f"Wrong predictions: {len(errors)} of {len(y_test)} -> reports/test_errors.csv")

pair_counts = errors.groupby(["actual", "predicted"]).size().sort_values(ascending=False).head(10)
print("\nMost common confusions (actual -> predicted):")
print(pair_counts.to_string())

# 6b. top weighted terms per class (what did the model learn?)
tfidf = best_pipe.named_steps["tfidf"]
clf = best_pipe.named_steps["clf"]
terms = np.array(tfidf.get_feature_names_out())
weights = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
top_rows = []
for i, cls in enumerate(clf.classes_):
    top_idx = weights[i].argsort()[::-1][:12]
    top_rows.append({"class": cls, "top_terms": ", ".join(terms[top_idx])})
pd.DataFrame(top_rows).to_csv("reports/top_features_per_class.csv", index=False)

# 6c. save the full pipeline: raw text in -> category out
joblib.dump(best_pipe, "models/best_tfidf_model.pkl")
print("\nSaved: models/best_tfidf_model.pkl, reports/model_comparison.csv,")
print("       reports/test_classification_report.csv, reports/test_errors.csv,")
print("       reports/top_features_per_class.csv, reports/figures/09_confusion_matrix.png")

# quick sanity check: the saved pipeline must predict from RAW text
loaded = joblib.load("models/best_tfidf_model.pkl")
demo = "Experienced chef with 10 years in restaurant kitchens, menu planning and food safety."
print("\nDemo prediction:", loaded.predict([demo])[0])