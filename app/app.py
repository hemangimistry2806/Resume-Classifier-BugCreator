"""FRONTEND: Streamlit app for the Resume Classifier.

Run from the repo root:   py -3.9 -m streamlit run app/app.py

Tabs
  1. Predict           - paste / upload a resume, get category + department
  2. Compare models    - SVM vs Word2Vec + Dense NN: accuracy comparison and side-by-side prediction
  3. Batch             - upload many resumes, get a table + CSV download
  4. Model evaluation  - test accuracy, macro-F1, per-class + per-department scores,
                         confusion matrix, errors (read from reports/ saved by src/train.py)
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.departments import get_department  # noqa: E402
from src.dl_model import predict_category_dl  # noqa: E402
from src.predict import SAMPLES, load_model, predict_category  # noqa: E402  (backend)

st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="wide")

# ---------- bigger, clearer text (works in light and dark theme) ----------
st.markdown(
    """
    <style>
    html, body, .stApp { font-size: 18px; }
    h1 { font-size: 2.6rem !important; }
    h2, h3 { font-size: 1.7rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 1.25rem; padding: 0.7rem 1.4rem; }
    .stTextArea textarea { font-size: 1.05rem !important; line-height: 1.55; }
    .stButton > button { font-size: 1.25rem; padding: 0.7rem 1.6rem; width: 100%; font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 2.4rem; }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem; }
    .result-card { border-left: 8px solid #4f8bf9; background: rgba(79,139,249,0.14);
                   padding: 1.1rem 1.4rem; border-radius: 10px; margin: 0.5rem 0 1rem 0; }
    .result-label { font-size: 1.05rem; opacity: 0.8; }
    .result-value { font-size: 2.6rem; font-weight: 800; line-height: 1.2; }
    .dept-badge { display: inline-block; margin-top: 0.5rem; padding: 0.25rem 0.9rem;
                  border-radius: 999px; background: rgba(79,139,249,0.35);
                  font-size: 1.1rem; font-weight: 600; }
    .hint-card { border: 2px dashed rgba(128,128,128,0.5); padding: 1.6rem; border-radius: 10px;
                 text-align: center; font-size: 1.15rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- helpers ----------
def pretty(name: str) -> str:
    """INFORMATION-TECHNOLOGY -> Information Technology (short names like HR stay as they are)."""
    return name if len(name) <= 3 else name.replace("-", " ").title()


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def read_report(filename, **kwargs):
    path = ROOT / "reports" / filename
    return pd.read_csv(path, **kwargs) if path.exists() else None


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


def run_prediction(text):
    """Ask for all 24 categories; fall back to 5 if the backend does not allow it."""
    try:
        return predict_category(text, top_k=24)
    except Exception:
        return predict_category(text, top_k=5)


NO_SAMPLE = "-- choose a sample --"
SAMPLE_OPTIONS = {
    "Chef resume (sample)": SAMPLES["chef"],
    "IT resume (sample)": SAMPLES["it"],
    "Teacher resume (sample)": SAMPLES["teacher"],
    "HR resume (sample)": SAMPLES["hr"],
}


def on_sample_change():
    choice = st.session_state.get("sample_choice")
    if choice in SAMPLE_OPTIONS:
        st.session_state["resume_text"] = SAMPLE_OPTIONS[choice]
        st.session_state.pop("result", None)   # clear the old result


get_model()   # load the model once

# ---------- sidebar ----------
with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload a resume (.txt / .pdf) **or** paste its text\n"
        "2. Click **Predict category**\n"
        "3. Use the slider to see more categories\n"
        "4. **Compare models** shows SVM vs the neural network\n"
        "5. Try **Batch** for many resumes, and **Model evaluation** to see how reliable "
        "the model is"
    )
    st.divider()
    st.header("About the model")
    st.write("Final model: TF-IDF + Linear SVM (24 categories). A Word2Vec + Dense neural "
             "network is also trained for comparison. Each category maps to a department.")
    st.caption("Categories with similar vocabulary (for example Finance and Accountant) "
               "can be confused.")

# ---------- page ----------
st.title("📄 Resume Classifier")
st.write("Find out which job category and department a resume belongs to.")

tab_predict, tab_compare, tab_batch, tab_eval = st.tabs(
    ["🔮 Predict", "⚖️ Compare models", "📚 Batch", "📊 Model evaluation"])

# =====================================================================
# TAB 1: PREDICT
# =====================================================================
with tab_predict:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.subheader("1. Add a resume")
        uploaded = st.file_uploader("Upload a resume (.txt or .pdf)", type=["txt", "pdf"])
        if uploaded is not None:
            signature = (uploaded.name, uploaded.size)
            if st.session_state.get("upload_signature") != signature:
                st.session_state["resume_text"] = text_from_upload(uploaded)
                st.session_state["upload_signature"] = signature
                st.session_state.pop("result", None)

        st.selectbox("Or try a sample resume", [NO_SAMPLE, *SAMPLE_OPTIONS],
                     key="sample_choice", on_change=on_sample_change)
        st.text_area("Resume text", key="resume_text", height=330,
                     placeholder="Paste the full resume text here...")
        clicked = st.button("🔍 Predict category", type="primary")

    # Run the model only when the button is clicked, and remember the result so the
    # slider below does not make it disappear.
    if clicked:
        text_now = st.session_state.get("resume_text", "")
        if not text_now.strip():
            st.session_state["result"] = None
            st.session_state["empty_warning"] = True
        else:
            st.session_state["empty_warning"] = False
            st.session_state["result"] = run_prediction(text_now)
            st.session_state["result_words"] = len(text_now.split())

    with right:
        st.subheader("2. Result")
        result = st.session_state.get("result")

        if st.session_state.get("empty_warning"):
            st.warning("Please paste or upload a resume first.")
        elif result is None:
            st.markdown('<div class="hint-card">Your prediction will appear here.<br>'
                        'Add a resume on the left and click <b>Predict category</b>.</div>',
                        unsafe_allow_html=True)
        else:
            words = st.session_state.get("result_words", 0)
            top_all = result["top"]
            margin = top_all[0][1] - top_all[1][1]
            dept = get_department(result["category"])

            st.markdown(
                f'<div class="result-card"><div class="result-label">Predicted category</div>'
                f'<div class="result-value">{pretty(result["category"])}</div>'
                f'<div class="dept-badge">🏢 Department: {dept}</div></div>',
                unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Lead over 2nd choice", f"{margin:+.2f}",
                      help="Gap between the best and the second-best category. "
                           "A bigger gap means a clearer prediction.")
            c2.metric("Words in resume", f"{words:,}")
            if words < 50:
                st.warning("This resume is very short, so the prediction may be unreliable.")

            view = st.radio("Show scores by", ["Category", "Department"],
                            horizontal=True, key="score_view")
            best = pretty(result["category"])

            if view == "Category":
                max_k = len(top_all)
                if max_k > 3:
                    k = st.slider("How many categories to show", 3, max_k,
                                  min(5, max_k), key="top_k")
                else:
                    k = max_k
                top = top_all[:k]
                chart_df = pd.DataFrame({
                    "Category": [pretty(c) for c, _ in top],
                    "Department": [get_department(c) for c, _ in top],
                    "Score": [s for _, s in top]})
                chart = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Score:Q", title=result["score_type"].title()),
                        y=alt.Y("Category:N", title=None,
                                sort=alt.EncodingSortField(field="Score", order="descending"),
                                axis=alt.Axis(labelFontSize=15, labelLimit=260)),
                        color=alt.condition(alt.datum.Category == best,
                                            alt.value("#4f8bf9"), alt.value("#9aa5b1")),
                        tooltip=["Category", "Department", alt.Tooltip("Score:Q", format=".3f")],
                    )
                    .properties(height=max(240, 34 * k))
                )
                st.altair_chart(chart)
            else:
                dept_scores = {}
                for c, s in top_all:
                    d = get_department(c)
                    dept_scores[d] = max(s, dept_scores.get(d, float("-inf")))
                dept_df = pd.DataFrame({"Department": list(dept_scores),
                                        "Score": list(dept_scores.values())})
                chart = (
                    alt.Chart(dept_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Score:Q", title="Best category score in department"),
                        y=alt.Y("Department:N", title=None,
                                sort=alt.EncodingSortField(field="Score", order="descending"),
                                axis=alt.Axis(labelFontSize=15, labelLimit=260)),
                        color=alt.condition(alt.datum.Department == dept,
                                            alt.value("#4f8bf9"), alt.value("#9aa5b1")),
                        tooltip=["Department", alt.Tooltip("Score:Q", format=".3f")],
                    )
                    .properties(height=max(240, 40 * len(dept_df)))
                )
                st.altair_chart(chart)
                st.caption("Each bar is the best-scoring category inside that department.")

            if result["score_type"] == "decision score":
                st.caption("Bars further to the right mean a better match. These are Linear SVM "
                           "scores, not percentages.")

# =====================================================================
# TAB 2: COMPARE MODELS (SVM vs Word2Vec + Dense NN)
# =====================================================================
with tab_compare:
    st.subheader("Accuracy comparison on the same test resumes")
    comp = read_report("model_compare_svm_vs_dl.csv")
    if comp is None:
        st.info("Run  `py -3.9 src\\train_dl.py`  first to create the comparison.")
    else:
        test_df = comp[comp["split"] == "test"].copy()
        long_df = test_df.melt(id_vars=["model"],
                               value_vars=["accuracy", "macro_f1", "weighted_f1"],
                               var_name="Metric", value_name="Score")
        long_df["Metric"] = long_df["Metric"].map(
            {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1"})
        long_df = long_df.rename(columns={"model": "Model"})
        cmp_chart = (
            alt.Chart(long_df).mark_bar()
            .encode(x=alt.X("Model:N", title=None, axis=alt.Axis(labels=False)),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                    color="Model:N", column=alt.Column("Metric:N", title=None),
                    tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")])
            .properties(width=160, height=300)
        )
        st.altair_chart(cmp_chart)
        shown = comp.rename(columns={"model": "Model", "split": "Split", "accuracy": "Accuracy",
                                     "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1"})
        st.dataframe(shown.round(3), hide_index=True)
        st.caption("Both models use the same 70/15/15 stratified split. Word2Vec was trained on "
                   "the training resumes only. The SVM is the final model because it has the "
                   "higher macro-F1.")

    st.subheader("Try both models on one resume")
    text_cmp = st.session_state.get("resume_text", "")
    if not text_cmp.strip():
        st.info("Add a resume in the Predict tab first (upload, paste or sample), then come back.")
    elif st.button("⚖️ Compare predictions", key="cmp_btn"):
        svm_res = predict_category(text_cmp, top_k=3)
        try:
            dl_res = predict_category_dl(text_cmp, top_k=3)
        except Exception as err:
            dl_res = None
            st.error(f"Deep-learning model could not run: {err}")
        if dl_res is not None:
            a, b = st.columns(2)
            for col, title, res, note in [
                (a, "TF-IDF + Linear SVM", svm_res, "decision scores"),
                (b, "Word2Vec + Dense NN", dl_res, "probabilities")]:
                with col:
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">{title}</div>'
                        f'<div class="result-value">{pretty(res["category"])}</div>'
                        f'<div class="dept-badge">🏢 {get_department(res["category"])}</div></div>',
                        unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame({
                        "Closest categories": [pretty(c) for c, _ in res["top"]],
                        note.title(): [round(s, 3) for _, s in res["top"]]}), hide_index=True)
            if svm_res["category"] == dl_res["category"]:
                st.success("Both models agree on the category.")
            else:
                st.warning("The models disagree. The SVM is the more accurate model overall.")

# =====================================================================
# TAB 3: BATCH
# =====================================================================
with tab_batch:
    st.subheader("Classify many resumes at once")
    files = st.file_uploader("Upload several resumes (.txt or .pdf)", type=["txt", "pdf"],
                             accept_multiple_files=True, key="batch_files")
    if files:
        rows = []
        for f in files:
            t = text_from_upload(f)
            if not t.strip():
                rows.append({"File": f.name, "Category": "(no text found)", "Department": "-",
                             "Lead over 2nd choice": None, "Words": 0})
                continue
            r = run_prediction(t)
            rows.append({"File": f.name,
                         "Category": pretty(r["category"]),
                         "Department": get_department(r["category"]),
                         "Lead over 2nd choice": round(r["top"][0][1] - r["top"][1][1], 2),
                         "Words": len(t.split())})
        batch_df = pd.DataFrame(rows)
        st.dataframe(batch_df, hide_index=True)

        counts = batch_df[batch_df["Department"] != "-"]["Department"].value_counts()
        if len(counts):
            st.markdown("**Resumes per department**")
            st.bar_chart(counts)

        st.download_button("⬇️ Download results (CSV)", batch_df.to_csv(index=False),
                           file_name="resume_predictions.csv", mime="text/csv")
    else:
        st.markdown('<div class="hint-card">Upload two or more resumes to see a table '
                    'of categories and departments.</div>', unsafe_allow_html=True)

# =====================================================================
# TAB 4: MODEL EVALUATION
# =====================================================================
with tab_eval:
    report = read_report("test_classification_report.csv", index_col=0)
    comparison = read_report("model_comparison.csv")
    errors = read_report("test_errors.csv")

    if report is None:
        st.info("No evaluation files found yet. Run  `py -3.9 src\\train.py`  first.")
    else:
        accuracy = report.loc["accuracy", "f1-score"]
        macro_f1 = report.loc["macro avg", "f1-score"]
        weighted_f1 = report.loc["weighted avg", "f1-score"]
        n_test = int(report.loc["macro avg", "support"])

        st.subheader("Final results on unseen test resumes")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy:.1%}")
        m2.metric("Macro-F1", f"{macro_f1:.3f}")
        m3.metric("Weighted-F1", f"{weighted_f1:.3f}")
        m4.metric("Test resumes", n_test)

        st.info(
            "**How to read these numbers**\n\n"
            "- **Accuracy**: share of test resumes given the right category. It can hide poor "
            "results on small categories.\n"
            "- **Macro-F1**: average F1 across all 24 categories, so small and large categories "
            "count equally. This is the main metric here.\n"
            "- **Weighted-F1**: like macro-F1 but larger categories count more."
        )

        # per-class F1
        st.subheader("Score for each category (F1)")
        classes = report.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        class_df = pd.DataFrame({
            "Category": [pretty(c) for c in classes.index],
            "Department": [get_department(c) for c in classes.index],
            "F1": classes["f1-score"].values,
            "Precision": classes["precision"].values,
            "Recall": classes["recall"].values,
            "Test resumes": classes["support"].astype(int).values,
        }).sort_values("F1", ascending=False)

        f1_chart = (
            alt.Chart(class_df)
            .mark_bar()
            .encode(
                x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Category:N", sort=alt.EncodingSortField(field="F1", order="descending"),
                        title=None, axis=alt.Axis(labelFontSize=13, labelLimit=260)),
                color=alt.condition(alt.datum.F1 < 0.6, alt.value("#f08c2e"), alt.value("#4f8bf9")),
                tooltip=["Category", "Department", alt.Tooltip("F1:Q", format=".2f"),
                         alt.Tooltip("Precision:Q", format=".2f"),
                         alt.Tooltip("Recall:Q", format=".2f"), "Test resumes"],
            )
            .properties(height=max(300, 26 * len(class_df)))
        )
        st.altair_chart(f1_chart)
        st.caption("Orange bars have F1 below 0.6. Categories with very few test resumes "
                   "(for example BPO) give unreliable scores.")
        with st.expander("Show the full per-category table"):
            st.dataframe(class_df.round(3), hide_index=True)

        # department-wise results (derived from the same test predictions)
        if errors is not None:
            st.subheader("Department-wise results")
            e = errors.copy()
            e["actual_dept"] = e["actual"].map(get_department)
            e["pred_dept"] = e["predicted"].map(get_department)
            cross = e[e["actual_dept"] != e["pred_dept"]]   # mistakes that cross departments

            totals = {}
            for c, n in classes["support"].items():
                d = get_department(c)
                totals[d] = totals.get(d, 0) + int(n)
            out_ = cross.groupby("actual_dept").size()
            in_ = cross.groupby("pred_dept").size()

            rows = []
            for d, total in totals.items():
                tp = total - int(out_.get(d, 0))
                fp = int(in_.get(d, 0))
                recall = tp / total if total else 0
                precision = tp / (tp + fp) if (tp + fp) else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
                rows.append({"Department": d, "Precision": precision, "Recall": recall,
                             "F1": f1, "Test resumes": total})
            dept_df = pd.DataFrame(rows).sort_values("F1", ascending=False)

            dept_acc = 1 - len(cross) / n_test
            d1, d2 = st.columns(2)
            d1.metric("Accuracy (24 categories)", f"{accuracy:.1%}")
            d2.metric("Accuracy (departments)", f"{dept_acc:.1%}",
                      help="A prediction counts as right if the department is right, even if "
                           "the exact category is a neighbour (e.g. Consultant vs Sales).")
            st.dataframe(dept_df.round(3), hide_index=True)
            st.caption("Department accuracy is higher because mix-ups inside one department "
                       "no longer count as mistakes. The 24-category result is still the main "
                       "result.")

        # model comparison
        if comparison is not None:
            st.subheader("How the models compared (validation set)")
            shown = comparison.rename(columns={
                "model": "Model", "ngram_range": "N-grams", "stopwords": "Stopwords",
                "val_accuracy": "Accuracy", "val_macro_f1": "Macro-F1",
                "val_weighted_f1": "Weighted-F1"})[
                ["Model", "N-grams", "Stopwords", "Accuracy", "Macro-F1", "Weighted-F1"]]
            st.dataframe(shown.round(3), hide_index=True)
            st.caption("The best model by validation macro-F1 was then tested once on the "
                       "test set (results above).")

        # confusion matrix
        cm_path = ROOT / "reports" / "figures" / "09_confusion_matrix.png"
        if cm_path.exists():
            st.subheader("Confusion matrix")
            st.image(str(cm_path))
            st.caption("Each row is the true category. The dark diagonal shows correct "
                       "predictions; dark cells off the diagonal show which categories get mixed up.")

        # mistakes
        if errors is not None and len(errors):
            st.subheader("Most common mistakes")
            pairs = (errors.groupby(["actual", "predicted"]).size().reset_index(name="Times")
                     .sort_values("Times", ascending=False).head(8))
            pairs["actual"] = pairs["actual"].map(pretty)
            pairs["predicted"] = pairs["predicted"].map(pretty)
            st.dataframe(pairs.rename(columns={"actual": "True category",
                                               "predicted": "Predicted as"}), hide_index=True)
            with st.expander(f"Show all {len(errors)} wrong predictions"):
                st.dataframe(errors, hide_index=True)