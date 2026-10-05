"""BACKEND: load the saved pipeline and predict a resume category from RAW text.

The saved pipeline already contains clean_text + TF-IDF + the model, so the
preprocessing at prediction time is identical to training.

Use from code:     from src.predict import predict_category
Quick command-line test:
    py -3.9 src\\predict.py                  (runs a few built-in sample resumes)
    py -3.9 src\\predict.py my_resume.txt    (predicts for a text file)
"""
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))      # the saved pipeline needs `src.clean_text` to be importable

MODEL_PATH = ROOT / "models" / "best_tfidf_model.pkl"
_model = None


def load_model():
    """Load the pipeline once and reuse it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_category(text: str, top_k: int = 3) -> dict:
    """Raw resume text -> predicted category plus the top_k closest categories.

    Returns {"category": str, "score_type": str, "top": [(category, score), ...]}
    score_type is "probability" for models that give probabilities, otherwise
    "decision score" (Linear SVM): higher = more likely, but it is NOT a percentage.
    """
    if text is None or not str(text).strip():
        raise ValueError("Resume text is empty.")

    model = load_model()
    classes = model.named_steps["clf"].classes_

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba([text])[0]
        score_type = "probability"
    else:
        scores = model.decision_function([text])[0]
        score_type = "decision score"

    order = np.argsort(scores)[::-1][:top_k]
    return {
        "category": str(classes[order[0]]),
        "score_type": score_type,
        "top": [(str(classes[i]), float(scores[i])) for i in order],
    }


# Short invented resumes, only for a quick smoke test. For the real "unseen resume"
# test, also try resumes from the test split and a few written by hand.
SAMPLES = {
    "chef": "Executive chef with 12 years in restaurant kitchens. Menu planning, food cost control, "
            "catering events, culinary team training, kitchen safety and inventory management.",
    "it": "Network administrator. Windows server, Active Directory, hardware troubleshooting, "
          "software deployment, network security, help desk support, SQL, Python scripting.",
    "teacher": "Elementary school teacher. Lesson planning, classroom management, student assessment, "
               "parent communication, differentiated instruction, curriculum development.",
    "hr": "Human resources generalist. Recruitment, employee relations, payroll, benefits "
          "administration, onboarding, HRIS, compensation and policy compliance.",
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        content = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
        print(predict_category(content))
    else:
        for name, resume in SAMPLES.items():
            result = predict_category(resume)
            print(f"{name:8s} -> {result['category']:25s} top3: {result['top']}")