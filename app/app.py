"""FRONTEND: Streamlit app for the Resume Classifier.

Run from the repo root:   streamlit run app/app.py
(If `streamlit` is not found:  py -3.9 -m streamlit run app/app.py)
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import load_model, predict_category  # noqa: E402  (backend)

st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


def text_from_upload(uploaded):
    """Read text from an uploaded .txt or .pdf file."""
    if uploaded.name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            st.error("PDF support needs pypdf:  py -3.9 -m pip install pypdf")
            return ""
        reader = PdfReader(uploaded)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded.read().decode("utf-8", errors="ignore")


st.title("📄 Resume Classifier")
st.write("Paste a resume or upload a .txt / .pdf file, and the model predicts its job category.")

get_model()   # load once

# Sidebar: real numbers from the saved test report (if available)
with st.sidebar:
    st.header("About the model")
    st.write("TF-IDF features + Linear SVM, trained on 24 resume categories.")
    try:
        report = pd.read_csv(ROOT / "reports" / "test_classification_report.csv", index_col=0)
        st.metric("Test accuracy", f"{report.loc['accuracy', 'f1-score']:.1%}")
        st.metric("Test macro-F1", f"{report.loc['macro avg', 'f1-score']:.3f}")
    except Exception:
        st.caption("Run src/train.py to generate test metrics.")
    st.caption("Resumes that overlap in vocabulary (e.g. FINANCE and ACCOUNTANT) can be confused.")

uploaded = st.file_uploader("Upload a resume (.txt or .pdf)", type=["txt", "pdf"])
default_text = text_from_upload(uploaded) if uploaded else ""
resume_text = st.text_area("Or paste the resume text here", value=default_text, height=300)

if st.button("Predict category", type="primary"):
    if not resume_text.strip():
        st.warning("Please paste or upload a resume first.")
    elif len(resume_text.split()) < 20:
        st.warning("This text is very short (under 20 words). The prediction may be unreliable.")
        result = predict_category(resume_text)
        st.success(f"Predicted category: **{result['category']}**")
    else:
        result = predict_category(resume_text, top_k=3)
        st.success(f"Predicted category: **{result['category']}**")

        table = pd.DataFrame(result["top"], columns=["Category", result["score_type"].title()])
        st.subheader("Top 3 closest categories")
        st.dataframe(table, hide_index=True)
        st.bar_chart(table.set_index("Category"))
        if result["score_type"] == "decision score":
            st.caption("Scores are Linear SVM decision scores: higher means more likely. "
                       "They are not percentages.")