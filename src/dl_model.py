"""Deep-learning model helpers: Word2Vec (mean-pooled) + Dense neural network.

Word2Vec is only the REPRESENTATION. The classifier is the Dense network (MLP).
Kept in its own module so the saved model can be loaded by the app.
"""
import re
from pathlib import Path

import joblib
import numpy as np

from src.clean_text import clean_text

ROOT = Path(__file__).resolve().parent.parent
DL_MODEL_PATH = ROOT / "models" / "w2v_dense_model.pkl"
TOKEN_PATTERN = r"[a-z0-9.+#]{2,}"      # same token rule as the TF-IDF model (keeps c++, c#, .net)


def tokenize(text):
    """Raw resume -> same clean_text as the SVM pipeline -> list of tokens."""
    return re.findall(TOKEN_PATTERN, clean_text(text))


def embed_texts(w2v, texts):
    """Each resume -> ONE vector = mean of its word vectors (unknown words are skipped)."""
    dim = w2v.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        words = [w for w in tokenize(t) if w in w2v.wv.key_to_index]
        if words:
            out[i] = w2v.wv[words].mean(axis=0)
    return out


class W2VDenseClassifier:
    """Raw text in -> Word2Vec mean vector -> scaler -> Dense network -> category."""

    def __init__(self, w2v, scaler, mlp):
        self.w2v, self.scaler, self.mlp = w2v, scaler, mlp

    @property
    def classes_(self):
        return self.mlp.classes_

    def predict_proba(self, texts):
        return self.mlp.predict_proba(self.scaler.transform(embed_texts(self.w2v, texts)))

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


_dl_model = None


def load_dl_model():
    global _dl_model
    if _dl_model is None:
        _dl_model = joblib.load(DL_MODEL_PATH)
    return _dl_model


def predict_category_dl(text: str, top_k: int = 3) -> dict:
    """Same output format as predict_category() in src/predict.py."""
    if text is None or not str(text).strip():
        raise ValueError("Resume text is empty.")
    model = load_dl_model()
    probs = model.predict_proba([text])[0]
    order = np.argsort(probs)[::-1][:top_k]
    return {
        "category": str(model.classes_[order[0]]),
        "score_type": "probability",
        "top": [(str(model.classes_[i]), float(probs[i])) for i in order],
    }