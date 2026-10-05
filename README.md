# 📄 Resume Classifier — BugCreator

An end-to-end **Resume Classification System** using Natural Language Processing (NLP), Machine Learning and Deep Learning.

The system reads the text of a resume and predicts its **job category** (for example *HR*, *Chef*, *Information Technology*) and the broader **department** that category belongs to.

---

## 🚀 Project Overview

Reviewing a large number of resumes by hand is slow. This project takes resume text (pasted, or uploaded as .txt / .pdf) and predicts its category automatically.

Two models are built and compared on exactly the same data split:

1. **TF-IDF + Linear SVM** (classical machine learning, the final model)
2. **Word2Vec + Dense Neural Network** (deep learning)

### Overall Workflow

```text
Resume Dataset
      ↓
Data Quality Check & Cleaning
      ↓
Exploratory Data Analysis
      ↓
Text Preprocessing (clean_text)
      ↓
Stratified Train / Validation / Test Split (70 / 15 / 15)
      ↓
Feature Engineering (TF-IDF  |  Word2Vec)
      ↓
Models (Naive Bayes, Logistic Regression, Linear SVM  |  Dense NN)
      ↓
Evaluation (Accuracy, Macro-F1, Weighted-F1, Confusion Matrix)
      ↓
Error Analysis
      ↓
Final Pipeline + Streamlit Demo
```

---

# 🎯 Problem Statement

Given the text of a resume, predict its category.

### Input

Resume text containing information such as education, skills, experience, projects, certifications and job titles.

### Output

* A predicted **category** (one of 24)
* The **department** of that category (one of 7)

Example:

```text
Input:
Executive chef with 12 years in restaurant kitchens. Menu planning,
food cost control, catering events, kitchen safety...

Output:
Category   : CHEF
Department : Hospitality & Agriculture
```

---

# 🧠 Objectives

* Understand the dataset before modelling.
* Find missing values, duplicates and empty resumes.
* Analyse class distribution and imbalance.
* Clean text carefully without deleting useful skills (C++, C#, .NET, SQL).
* Split the data **before** fitting any vectorizer, to avoid data leakage.
* Build TF-IDF baselines and one deep-learning model.
* Compare models with macro-F1, per-class scores and a confusion matrix, not accuracy alone.
* Analyse wrong predictions.
* Build one reproducible pipeline: raw resume in, category out.
* Provide a Streamlit demo.

---

# 📊 Dataset

The project uses a resume dataset with **2,484 resumes** in **24 categories**.

### Dataset Columns

```text
ID
Resume_str
Resume_html
Category
```

### Cleaning Result

```text
Original resumes : 2,484
Empty resume     : 1  (removed)
Duplicates       : 2  (removed)
Clean dataset    : 2,481
```

Duplicates are removed because the same resume appearing in both train and test would make the score look better than it really is.

### The 24 Categories

ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER

---

# ⚖️ Class Distribution and Imbalance

Most categories have roughly **96–120** resumes, but a few are much smaller:

| Category    | Approximate Resumes |
| ----------- | ------------------: |
| BPO         |                  22 |
| AUTOMOBILE  |                  36 |
| AGRICULTURE |                  63 |

A model can get a good overall accuracy while failing on these small classes. That is why the project reports **macro-F1, weighted-F1, precision, recall and per-class scores**, and uses `class_weight="balanced"` for the classical models.

---

# 📂 Project Structure

```text
Resume-Classifier-BugCreator/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── Resume.csv
│   └── processed/
│       └── resume_clean.csv
│
├── notebooks/
│   └── 01_data_cleaning_eda.py      # quality check, cleaning, EDA charts
│
├── src/
│   ├── clean_text.py                # the ONE cleaning function (training + prediction)
│   ├── train.py                     # split, TF-IDF models, evaluation, saves best pipeline
│   ├── predict.py                   # loads SVM pipeline, predicts from raw text
│   ├── train_dl.py                  # Word2Vec + Dense NN, SVM vs DL comparison
│   ├── dl_model.py                  # deep-learning model class and predict function
│   └── departments.py               # category -> department mapping
│
├── models/
│   ├── best_tfidf_model.pkl         # full pipeline: clean_text + TF-IDF + SVM
│   └── w2v_dense_model.pkl          # Word2Vec + scaler + Dense NN
│
├── reports/
│   ├── model_comparison.csv                 # validation results of all TF-IDF experiments
│   ├── model_compare_svm_vs_dl.csv          # SVM vs deep learning (validation + test)
│   ├── test_classification_report.csv       # SVM per-class test report
│   ├── dl_test_classification_report.csv    # deep-learning per-class test report
│   ├── test_errors.csv                      # SVM wrong predictions
│   ├── dl_test_errors.csv                   # deep-learning wrong predictions
│   ├── top_features_per_class.csv           # top weighted terms per class
│   └── figures/                             # EDA charts and confusion matrix
│
└── app/
    └── app.py                       # Streamlit demo
```

---

# 🔍 Data Understanding and Quality Check

Before any modelling, the following were checked:

* Dataset shape, columns and data types
* Missing values
* Empty and very short resumes
* Exact duplicate resumes
* Same resume text under two different labels
* Class distribution and imbalance
* Resume length (characters and words)

---

# 📈 Exploratory Data Analysis

EDA shows what a resume looks like in this dataset and which words separate the categories.

1. **Class distribution** – bar chart, shows the imbalance.
2. **Resume length** – characters and words, per class.
3. **Word frequency** – most frequent words after cleaning.
4. **Word clouds** – overall and for the largest classes (supporting chart only).
5. **N-grams** – top unigrams, bigrams and trigrams.
6. **Class-wise vocabulary** – top TF-IDF terms per class.

Charts are saved in `reports/figures/`.

---

# 🧹 Text Preprocessing

One function, `clean_text()` in `src/clean_text.py`, is used for training **and** prediction, so both always match.

```text
Raw Resume
     ↓
Lowercase
     ↓
Remove HTML
     ↓
Replace URL / Email / Phone with tokens
     ↓
Keep useful characters  +  #  .
     ↓
Fix whitespace
     ↓
Clean Resume
```

### Why these choices (easy to explain)

* **Keep `+ # .`** – otherwise `C++` becomes `c` and `.NET` becomes `net`, and real skills are lost.
* **Replace URLs, emails and phone numbers with tokens** – we keep the fact that a contact detail exists, but not the exact value.
* **Stopwords are not removed by default** – the guide says to test their effect first. The comparison includes a stopword-removed run, and the result is in `reports/model_comparison.csv`.
* **Encoding artifacts** such as the stray `Â` from PDFs are removed, because only letters, digits and `+ # .` are kept.

---

# 🧪 Train / Validation / Test Split

A **stratified** split (class proportions kept equal in every part), done **before** anything is fitted:

```text
70% → Training     (1,736 resumes)
15% → Validation   (  372 resumes)
15% → Test         (  373 resumes)
```

* The **validation** set is used to compare models and choose the best one.
* The **test** set is used **once**, only for the chosen model.

### Data Leakage Prevention

```text
Training data only → fit TF-IDF / Word2Vec / scaler
Validation, Test   → only transformed, never fitted on
```

Both models use the **same split** (`random_state=42`), so the comparison is fair.

---

# 🧩 Feature Engineering

## TF-IDF (classical models)

TF-IDF gives a word a high weight when it is frequent in one resume but rare across all resumes.

Settings: `min_df=2`, `max_df=0.9`, `sublinear_tf=True`, and a token pattern that keeps `c++`, `c#` and `.net`. Unigrams `(1,1)` are compared with unigrams + bigrams `(1,2)`.

## Word2Vec (deep-learning model)

Word2Vec turns each word into a dense vector from its context. It is trained **only on the training resumes**. Each resume becomes one vector by **averaging** its word vectors (mean pooling).

> Word2Vec is only the *representation*. The classifier is the Dense neural network.

---

# 🤖 Machine Learning Models

All use TF-IDF features and are compared on the validation set.

| Model | Role |
| ----- | ---- |
| Multinomial Naive Bayes | fast baseline |
| Logistic Regression | interpretable baseline |
| **Linear SVM** | strong model for sparse text, **selected as final** |

Experiments: each model with `(1,1)` and `(1,2)` n-grams, plus a stopword-removal test on the SVM. The best one by **validation macro-F1** is picked.

---

# 🧠 Deep Learning

```text
Resume Text
     ↓
clean_text  (same function as the SVM pipeline)
     ↓
Tokens
     ↓
Word2Vec (trained on training split only)
     ↓
Mean-pooled resume vector (100 dimensions)
     ↓
StandardScaler (fitted on training split only)
     ↓
Dense network (256 → 128 neurons, ReLU, early stopping)
     ↓
Softmax → Resume Category
```

Implemented with `gensim` (Word2Vec) and `scikit-learn` (`MLPClassifier`). No TensorFlow is needed.

---

# 📊 Model Evaluation

Metrics used:

* **Accuracy** – share of resumes given the right category
* **Precision / Recall / F1** – per category
* **Macro-F1** – average F1 over all 24 categories; small classes count equally (main metric)
* **Weighted-F1** – like macro-F1 but larger classes count more
* **Confusion matrix** – shows which categories get mixed up

---

# 📊 Results

### Final comparison on the TEST set (373 unseen resumes)

| Model | Representation | Accuracy | Macro-F1 | Weighted-F1 |
| ----- | -------------- | -------: | -------: | ----------: |
| **Linear SVM (final)** | TF-IDF | **0.686** | **0.666** | **0.676** |
| Dense Neural Network | Word2Vec (mean-pooled) | 0.555 | 0.492 | 0.533 |

### Comparison on the VALIDATION set

| Model | Representation | Accuracy | Macro-F1 | Weighted-F1 |
| ----- | -------------- | -------: | -------: | ----------: |
| Multinomial Naive Bayes | TF-IDF | FILL | FILL | FILL |
| Logistic Regression | TF-IDF | FILL | FILL | FILL |
| Linear SVM | TF-IDF | 0.653 | 0.629 | 0.638 |
| Dense Neural Network | Word2Vec | 0.487 | 0.436 | 0.470 |

All TF-IDF experiments (every model, n-gram setting and the stopword test) are in `reports/model_comparison.csv`.

### Why the SVM wins (easy to explain)

* Resumes are decided by **specific words** (SQL, payroll, menu planning). TF-IDF keeps them; averaging Word2Vec vectors blurs them.
* Word2Vec was trained from scratch on only about 1,700 resumes, so the embeddings are weak. Pretrained vectors or more data would help.
* The neural network has no class weighting, so the smallest classes (BPO, Automobile) score F1 = 0.
* The SVM was chosen because of its higher **macro-F1**, not only accuracy.

### Weak categories

The SVM finds it hardest to separate categories with generic, overlapping vocabulary: **Consultant, Sales, Arts**.

---

# 🏢 Department-wise Prediction

Each of the 24 categories is mapped to one of 7 departments in `src/departments.py`. The model is **not retrained**; the department is looked up from the predicted category.

| Department | Categories |
| ---------- | ---------- |
| Technology & Engineering | Information Technology, Engineering, Automobile, Aviation, Construction |
| Business & Management | Business Development, Consultant, Sales, HR, BPO, Public Relations |
| Finance & Legal | Finance, Accountant, Banking, Advocate |
| Healthcare & Wellness | Healthcare, Fitness |
| Creative & Media | Designer, Digital Media, Arts, Apparel |
| Education | Teacher |
| Hospitality & Agriculture | Chef, Agriculture |

The app shows the department next to the category, a department-level chart, and a department-wise results table in the evaluation tab.

> Department accuracy is higher than 24-category accuracy because mix-ups inside one department no longer count as mistakes. It is an extra view; the 24-category result stays the main result. The grouping is our own choice.

---

# 🔎 Error Analysis

Wrong predictions are saved with the actual class, predicted class, score, word count and a text preview:

* `reports/test_errors.csv` (SVM)
* `reports/dl_test_errors.csv` (deep learning)

Possible causes being studied:

* Overlapping categories (for example Finance vs Accountant)
* Generic or very short resumes
* Few examples for small classes
* Noisy text from PDF extraction
* Loss of detail from mean pooling (deep-learning model)

The aim is to explain *why* the model fails and use that to choose the next improvement, not to keep switching models.

---

# 🖥️ Streamlit Demo

The app has four tabs:

| Tab | What it does |
| --- | ------------ |
| 🔮 Predict | Upload a .txt / .pdf or paste text, get category + department, slider for more categories, switch between category and department scores |
| ⚖️ Compare models | SVM vs deep-learning accuracy chart, and both models' predictions side by side |
| 📚 Batch | Upload many resumes, get a table and a CSV download |
| 📊 Model evaluation | Accuracy, macro-F1, per-class and per-department scores, confusion matrix, common mistakes |

### Prediction Flow

```text
Paste / Upload Resume
       ↓
clean_text (same as training)
       ↓
Saved TF-IDF + Linear SVM
       ↓
Category  →  Department
```

The SVM gives **decision scores**, not percentages. "Lead over 2nd choice" is the gap between the best and second-best score; a bigger gap means a clearer prediction.

---

# 🛠️ Technologies Used

* **Language:** Python
* **Data & plots:** Pandas, NumPy, Matplotlib, Seaborn, WordCloud, Altair
* **Classical ML:** scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, Linear SVM)
* **Deep learning:** gensim (Word2Vec), scikit-learn `MLPClassifier`
* **Demo:** Streamlit, pypdf (PDF reading)
* **Tools:** VS Code, Git, GitHub

---

# 📦 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
wordcloud
joblib
streamlit
altair
pypdf
gensim
```

```bash
pip install -r requirements.txt
```

---

# ⚙️ Installation

```bash
git clone https://github.com/hemangimistry2806/Resume-Classifier-BugCreator.git
cd Resume-Classifier-BugCreator
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run all commands from the **repository root folder**.

### 1. Data cleaning and EDA

```bash
py -3.9 notebooks/01_data_cleaning_eda.py
```

Creates `data/processed/resume_clean.csv` and the charts in `reports/figures/`.

### 2. Train the TF-IDF models

```bash
py -3.9 src/train.py
```

Compares the models on validation, tests the best one once, and saves `models/best_tfidf_model.pkl` and the reports.

### 3. Train the deep-learning model and compare

```bash
py -3.9 src/train_dl.py
```

Trains Word2Vec + Dense NN, saves `models/w2v_dense_model.pkl` and `reports/model_compare_svm_vs_dl.csv`.

### 4. Run the app

```bash
py -3.9 -m streamlit run app/app.py
```

---

# 📁 Data Availability

The original dataset is expected at `data/Resume.csv`. If it is not in the repository, place the competition-provided file there. The cleaned file is created at `data/processed/resume_clean.csv`.

---

# 🔄 Reproducibility

```text
TRAINING

Raw Resume → clean_text → TF-IDF → Model → Saved pipeline


PREDICTION

New Resume → the same saved pipeline → Predicted Category
```

The cleaning function and the vectorizer live inside the saved pipeline, so training and prediction cannot drift apart. The split uses `random_state=42`.

---

# ✅ Development Status

## Phase 1 — Data Understanding & Quality

* [x] Dataset inspection
* [x] Missing-value, empty-resume and duplicate analysis
* [x] Class distribution and imbalance
* [x] Resume length analysis

## Phase 2 — EDA

* [x] Class distribution, resume length, word frequency
* [x] WordCloud
* [x] Unigram, bigram and trigram analysis
* [x] Class-wise top terms

## Phase 3 — Preprocessing

* [x] Reproducible `clean_text()` function
* [x] Technical tokens kept (C++, C#, .NET)
* [x] URL / email / phone tokens

## Phase 4 — Split & Features

* [x] Stratified 70 / 15 / 15 split before fitting
* [x] TF-IDF with (1,1) vs (1,2) n-grams
* [x] Stopword on/off test
* [x] Word2Vec trained on training split only

## Phase 5 — Machine Learning

* [x] Naive Bayes, Logistic Regression, Linear SVM
* [x] Model comparison on validation macro-F1
* [x] Final model selected (Linear SVM)

## Phase 6 — Deep Learning

* [x] Word2Vec + Dense neural network
* [x] Validation and test evaluation
* [x] SVM vs deep-learning comparison

## Phase 7 — Evaluation & Analysis

* [x] Accuracy, precision, recall, macro-F1, weighted-F1
* [x] Per-class results and confusion matrix
* [x] Department-wise results
* [ ] Written error analysis (data saved in `reports/`, write-up in progress)

## Phase 8 — Final Demo

* [x] Prediction pipeline (raw text → category → department)
* [x] Streamlit app (predict, compare, batch, evaluation)
* [ ] Test with 3–5 unseen resumes and record results
* [ ] Final documentation

---

# 👥 Team

### Team Name

**BugCreator**

### Project

**Resume Classifier — SAMATRIX ResumeForge 2026**

### Team Members

* **Hemangi Mistry**
* **Diya Patel**
* **Jeel Patel**

---

# 📜 License

This project is developed for educational and competition purposes as part of **SAMATRIX ResumeForge 2026**.
