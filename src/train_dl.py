"""Deep-learning model: Word2Vec (trained on TRAIN only) + Dense neural network.

Run from anywhere:   py -3.9 src\\train_dl.py
Needs: data/processed/resume_clean.csv and models/best_tfidf_model.pkl (from src/train.py)

Uses the SAME stratified 70/15/15 split as src/train.py, so SVM and the neural
network are compared on exactly the same test resumes.
"""
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.dl_model import W2VDenseClassifier, embed_texts, tokenize

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---- 1. load + split: identical to src/train.py ----
df = pd.read_csv("data/processed/resume_clean.csv")
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
X_raw, y = df["text"], df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# ---- 2. Word2Vec trained on the TRAINING resumes only (no leakage) ----
print("Training Word2Vec on training resumes only...")
train_tokens = [tokenize(t) for t in X_train]
w2v = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=2,
               sg=0, workers=4, epochs=10, seed=RANDOM_STATE)
print("Word2Vec vocabulary size:", len(w2v.wv.key_to_index))

# ---- 3. one vector per resume (mean pooling), scaler fitted on train only ----
Xtr = embed_texts(w2v, X_train)
scaler = StandardScaler().fit(Xtr)
Xtr = scaler.transform(Xtr)

# ---- 4. Dense neural network ----
print("Training Dense neural network...")
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-3,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                    max_iter=300, random_state=RANDOM_STATE)
mlp.fit(Xtr, y_train)
dl = W2VDenseClassifier(w2v, scaler, mlp)

# ---- 5. evaluate DL and SVM on the same val / test resumes ----
svm = joblib.load("models/best_tfidf_model.pkl")
svm_name = "TF-IDF + " + type(svm.named_steps["clf"]).__name__


def metrics(model, X, y_true):
    p = model.predict(X)
    return (accuracy_score(y_true, p), f1_score(y_true, p, average="macro"),
            f1_score(y_true, p, average="weighted"))


rows = []
for name, model in [(svm_name, svm), ("Word2Vec + Dense NN", dl)]:
    for split, X_, y_ in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        a, m, w = metrics(model, X_, y_)
        rows.append({"model": name, "split": split, "accuracy": a,
                     "macro_f1": m, "weighted_f1": w})
comp = pd.DataFrame(rows)
os.makedirs("reports", exist_ok=True)
comp.to_csv("reports/model_compare_svm_vs_dl.csv", index=False)
print("\n=== SVM vs DEEP LEARNING (same split) ===")
print(comp.round(4).to_string(index=False))

# ---- 6. DL per-class report + wrong predictions ----
dl_test_pred = dl.predict(X_test)
print("\n=== DL TEST REPORT ===")
print(classification_report(y_test, dl_test_pred, zero_division=0))
pd.DataFrame(classification_report(y_test, dl_test_pred, output_dict=True,
                                   zero_division=0)).T.to_csv("reports/dl_test_classification_report.csv")

conf = dl.predict_proba(X_test).max(axis=1)
errs = pd.DataFrame({
    "actual": y_test.values, "predicted": dl_test_pred,
    "confidence": np.round(conf, 3),
    "word_count": X_test.str.split().str.len().values,
    "preview": X_test.str.replace(r"\s+", " ", regex=True).str[:200].values,
})
errs = errs[errs["actual"] != errs["predicted"]].sort_values("confidence", ascending=False)
errs.to_csv("reports/dl_test_errors.csv", index=False)
print(f"DL wrong predictions: {len(errs)} of {len(y_test)} -> reports/dl_test_errors.csv")

# ---- 7. save ----
joblib.dump(dl, "models/w2v_dense_model.pkl")
print("\nSaved: models/w2v_dense_model.pkl, reports/model_compare_svm_vs_dl.csv, "
      "reports/dl_test_classification_report.csv, reports/dl_test_errors.csv")