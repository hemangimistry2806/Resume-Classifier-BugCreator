# 📄 Resume Classifier — BugCreator

An end-to-end **Resume Classification System** using Natural Language Processing (NLP), Machine Learning, and Deep Learning.

The objective of this project is to automatically classify a resume into its appropriate **career/job category** based on the information contained in the resume text.

---

## 🚀 Project Overview

Resume screening and classification can become time-consuming when a large number of resumes need to be reviewed manually.

This project builds an NLP-based classification pipeline that takes resume text as input and predicts its corresponding category.

### Overall Workflow

```text
Resume Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Text Preprocessing
      ↓
Train / Validation / Test Split
      ↓
Feature Engineering
      ↓
Machine Learning / Deep Learning
      ↓
Model Evaluation
      ↓
Error Analysis
      ↓
Final Prediction Pipeline
      ↓
Streamlit Demo
```

---

# 🎯 Problem Statement

Given the text of a resume, predict the appropriate resume category.

### Input

Resume text containing information such as:

* Education
* Skills
* Experience
* Projects
* Certifications
* Job titles
* Technical technologies

### Output

A predicted resume category.

Example:

```text
Input:
Python, SQL, Java, TensorFlow, AWS,
machine learning, software development...

Output:
INFORMATION-TECHNOLOGY
```

---

# 🧠 Objectives

The main objectives of this project are:

* Understand the resume dataset before modelling.
* Identify missing values, duplicates and empty resumes.
* Analyze class distribution and class imbalance.
* Perform meaningful NLP-based EDA.
* Clean and preprocess resume text carefully.
* Build a strong TF-IDF based machine learning baseline.
* Experiment with multiple machine-learning models.
* Build a deep-learning based approach.
* Compare models using appropriate evaluation metrics.
* Perform error analysis on incorrect predictions.
* Build a reproducible prediction pipeline.
* Provide a simple Streamlit interface for resume classification.

---

# 📊 Dataset

The project uses a resume classification dataset containing **2,484 resumes** across **24 resume categories**.

### Dataset Columns

```text
ID
Resume_str
Resume_html
Category
```

### Dataset Size

```text
Original resumes: 2,484
After cleaning:   2,481
```

During data-quality analysis:

```text
1 empty resume
2 duplicate resumes
```

were identified and removed.

Therefore:

```text
2,484 → 2,481 resumes
```

are used for the cleaned dataset.

---

# ⚖️ Class Distribution and Imbalance

The dataset contains **24 categories**.

Most categories contain approximately **96–120 resumes**, while a few categories have substantially fewer examples.

Examples of smaller classes include:

| Category    | Approximate Resumes |
| ----------- | ------------------: |
| BPO         |                  22 |
| AUTOMOBILE  |                  36 |
| AGRICULTURE |                  63 |

This class imbalance is important because a model can achieve high overall accuracy while performing poorly on smaller categories.

Therefore, model evaluation includes **Macro-F1, Weighted-F1, precision, recall, and per-class performance**, rather than relying only on accuracy.

---

# 📂 Project Structure

```text
Resume-Classifier-BugCreator/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── Resume.csv
│   │
│   └── processed/
│       └── resume_clean.csv
│
├── notebooks/
│   ├── 01_data_cleaning_eda.py
│   ├── 02_eda.py
│   ├── 03_tfidf_ml.py
│   ├── 04_lstm.py
│   └── 05_error_analysis.py
│
├── reports/
│   ├── class_top_terms.csv
│   └── figures/
│       ├── class_distribution.png
│       ├── resume_length.png
│       ├── word_frequency.png
│       ├── wordcloud.png
│       ├── bigram.png
│       ├── trigram.png
│       └── confusion_matrix.png
│
├── src/
│   ├── __init__.py
│   ├── clean_text.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── svm_model.pkl
│   ├── label_encoder.pkl
│   ├── tokenizer.pkl
│   └── lstm_model.keras
│
└── app/
    ├── app.py
    └── utils.py
```

---

# 🔍 Data Understanding and Quality Check

The first stage of the project focuses on understanding the dataset before model training.

The following checks were performed:

* Dataset shape
* Column inspection
* Data types
* Missing values
* Empty resume text
* Duplicate records
* Class distribution
* Class imbalance
* Resume text length

### Cleaning Result

```text
Original Dataset
       ↓
2,484 resumes
       ↓
Remove 1 empty resume
       ↓
Remove 2 duplicates
       ↓
Clean Dataset
       ↓
2,481 resumes
```

---

# 📈 Exploratory Data Analysis

EDA is an important part of the project because resume text contains category-specific skills, technologies, job titles, education terms and experience-related vocabulary.

The following EDA analyses are performed.

## 1. Class Distribution

A bar chart is used to understand the number of resumes in each category.

```text
Category
   ↓
Number of Resumes
```

This also helps identify class imbalance.

---

## 2. Resume Length

Resume length is analyzed using:

* Character count
* Word count

The analysis helps identify short resumes, long resumes and possible outliers.

---

## 3. Word Frequency

The most frequently occurring words are identified after basic text cleaning.

This provides an overview of the vocabulary used throughout the dataset.

---

## 4. WordCloud

A WordCloud is generated to visually inspect common resume vocabulary.

The WordCloud is used as a supporting visualization and is not the only EDA technique.

---

## 5. N-Gram Analysis

The project analyzes:

### Unigrams

```text
python
software
management
engineering
```

### Bigrams

```text
machine learning
software development
project management
```

### Trigrams

```text
natural language processing
machine learning model
software development life
```

N-grams help identify meaningful phrases rather than considering individual words only.

---

## 6. Class-Wise Vocabulary

Important category-specific terms are analyzed to understand which words and phrases distinguish different resume classes.

The results are stored in:

```text
reports/class_top_terms.csv
```

---

# 🧹 Text Preprocessing

The preprocessing pipeline is designed to remove unnecessary noise while preserving meaningful resume information.

The current preprocessing steps are:

```text
Raw Resume
     ↓
Lowercase
     ↓
Remove HTML
     ↓
Replace URL / Email / Phone with tokens
     ↓
Keep useful characters such as + # .
     ↓
Fix whitespace
     ↓
Clean Resume
```

### Current Preprocessing Details

#### 1. Lowercase

Text is converted to a consistent lowercase representation.

Example:

```text
Python → python
Machine Learning → machine learning
```

#### 2. Remove HTML

HTML markup from `Resume_html` or resume text is removed where applicable.

#### 3. URL / Email / Phone Handling

URLs, email addresses and phone numbers are normalized using tokens rather than blindly deleting all information.

Example:

```text
https://example.com → URL_TOKEN
example@gmail.com → EMAIL_TOKEN
9876543210 → PHONE_TOKEN
```

#### 4. Preserve Useful Technical Characters

Characters such as:

```text
+
#
.
```

are preserved because they can be meaningful in technical resume terms.

Examples:

```text
C++
C#
.NET
```

#### 5. Whitespace Normalization

Repeated spaces and unnecessary line breaks are normalized.

### Important Note

**Stopword removal is not currently applied.**

This allows the project to first evaluate the value of common words before deciding whether stopword removal improves classification performance.

---

# 🧪 Train / Validation / Test Split

The cleaned dataset is split into training, validation and test sets using a **stratified split** so that the category distribution is preserved as much as possible across the subsets.

### Final Split

```text
70% → Training
15% → Validation
15% → Testing
```

The test set is kept untouched until final model evaluation.

### Data Leakage Prevention

TF-IDF vocabulary and IDF statistics are fitted only on the training data.

```text
Training Data
      ↓
Fit TF-IDF
      ↓
Transform Validation
      ↓
Transform Test
```

The validation set is used for model selection and tuning, while the test set is reserved for final comparison.

---

# 🧩 Feature Engineering

## TF-IDF

TF-IDF is used as the primary classical NLP representation.

The project experiments with word-level features and n-grams.

Example:

```text
Unigram:
machine

Bigram:
machine learning
```

Important TF-IDF parameters include:

* `ngram_range`
* `min_df`
* `max_df`
* `max_features`

---

# 🤖 Machine Learning Models

The classical machine-learning stage includes:

## Logistic Regression

Used as a strong and interpretable baseline for multiclass text classification.

## Linear SVM

Used as a strong baseline for high-dimensional TF-IDF features.

## Multinomial Naive Bayes

Used as a fast probabilistic text-classification baseline.

The models are compared using the same evaluation strategy.

---

# 🧠 Deep Learning

A neural-network approach is also explored using text sequences and embeddings.

The planned architecture is:

```text
Resume Text
     ↓
Tokenizer
     ↓
Sequence
     ↓
Embedding
     ↓
LSTM / GRU
     ↓
Dropout
     ↓
Dense Layer
     ↓
Softmax
     ↓
Resume Category
```

Word2Vec or embeddings are treated as **representation techniques**, not as classifiers by themselves.

---

# 📊 Model Evaluation

Model performance is evaluated using multiple metrics.

The evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* Macro-F1
* Weighted-F1
* Confusion Matrix
* Per-class performance

## Why Macro-F1?

The dataset is imbalanced, with some classes containing significantly fewer resumes than the majority classes.

Macro-F1 gives equal importance to every category and therefore provides a more balanced view of multiclass performance.

---

# 🔎 Error Analysis

Incorrect predictions are inspected to understand where and why the model fails.

For misclassified resumes, the analysis considers:

```text
Actual Category
Predicted Category
Resume Text Preview
Prediction Score
Possible Reason
```

Possible causes include:

* Overlapping categories
* Similar technical skills
* Generic resumes
* Very short resumes
* Noisy text
* Limited examples for minority classes
* Similar job titles
* Feature representation limitations

The purpose of error analysis is to identify meaningful improvements rather than randomly changing models.

---

# 📊 Results

Final model results will be recorded after completing all experiments.

| Model                   | Representation | Accuracy | Macro-F1 | Weighted-F1 |
| ----------------------- | -------------- | -------: | -------: | ----------: |
| Multinomial Naive Bayes | TF-IDF         |      TBD |      TBD |         TBD |
| Logistic Regression     | TF-IDF         |      TBD |      TBD |         TBD |
| Linear SVM              | TF-IDF         |      TBD |      TBD |         TBD |
| LSTM / GRU              | Embedding      |      TBD |      TBD |         TBD |

> Results will be updated after final model training and evaluation.

---

# 🖥️ Streamlit Demo

The final system provides a simple Streamlit interface where a user can **paste resume text** and receive a predicted category.

### Prediction Flow

```text
Paste Resume Text
       ↓
Preprocessing
       ↓
Feature Extraction
       ↓
Trained Model
       ↓
Predicted Category
```

### Example Interface

```text
┌─────────────────────────────────────┐
│       Resume Classifier AI          │
├─────────────────────────────────────┤
│                                     │
│  Paste Resume Text                  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ Python, SQL, Java, AWS...     │  │
│  │ Machine learning...           │  │
│  └───────────────────────────────┘  │
│                                     │
│       [ Predict Category ]           │
│                                     │
│  Predicted Category:                │
│  INFORMATION-TECHNOLOGY             │
│                                     │
└─────────────────────────────────────┘
```

> The current demo uses pasted resume text. PDF parsing is not part of the current implementation.

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis & Visualization

* Pandas
* NumPy
* Matplotlib
* Seaborn

### NLP & Machine Learning

* Scikit-learn
* TF-IDF
* Word2Vec / Embeddings
* Logistic Regression
* Linear SVM
* Multinomial Naive Bayes

### Deep Learning

* TensorFlow / Keras
* LSTM / GRU

### Visualization

* WordCloud

### Deployment / Demo

* Streamlit

### Development Tools

* VS Code
* Git
* GitHub

---

# 📦 Requirements

The main Python dependencies are:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
wordcloud
joblib
streamlit
```

TensorFlow will also be required for the deep-learning component.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project:

```bash
cd Resume-Classifier-BugCreator
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## 1. Data Cleaning

Run:

```bash
python notebooks/01_data_cleaning_eda.py
```

This processes the original dataset and creates:

```text
data/processed/resume_clean.csv
```

---

## 2. Exploratory Data Analysis

Run:

```bash
python notebooks/02_eda.py
```

The generated charts are stored in:

```text
reports/figures/
```

The class-wise top terms are stored in:

```text
reports/class_top_terms.csv
```

---

## 3. TF-IDF Machine Learning

Run:

```bash
python notebooks/03_tfidf_ml.py
```

This trains and evaluates the classical machine-learning models.

---

## 4. Deep Learning

Run:

```bash
python notebooks/04_lstm.py
```

This trains the LSTM/GRU-based neural model.

---

## 5. Error Analysis

Run:

```bash
python notebooks/05_error_analysis.py
```

This analyzes incorrectly classified resumes.

---

## 6. Streamlit Demo

Run:

```bash
streamlit run app/app.py
```

The local Streamlit application will open in the browser.

---

# 📁 Data Availability

The original dataset is expected at:

```text
data/Resume.csv
```

If the dataset is not included in the GitHub repository because of dataset distribution or size restrictions, obtain the original dataset from the competition-provided source and place it at:

```text
data/Resume.csv
```

The cleaned file generated by the project is:

```text
data/processed/resume_clean.csv
```

---

# 💾 Model Files

Trained model artifacts are stored consistently inside:

```text
models/
```

Expected files include:

```text
models/
├── tfidf_vectorizer.pkl
├── svm_model.pkl
├── label_encoder.pkl
├── tokenizer.pkl
└── lstm_model.keras
```

The project does not use a separate `models/saved_models/` directory.

---

# 🔄 Reproducibility

The final pipeline ensures that preprocessing and feature extraction used during prediction are consistent with the training pipeline.

```text
TRAINING

Raw Resume
    ↓
Preprocessing
    ↓
TF-IDF / Embedding
    ↓
Model Training
    ↓
Saved Model


PREDICTION

New Resume Text
    ↓
Same Preprocessing
    ↓
Saved TF-IDF / Embedding
    ↓
Saved Model
    ↓
Predicted Category
```

This prevents differences between the training and inference pipelines.

---

# ✅ Development Status

## Phase 1 — Data Understanding & Quality

* [x] Dataset inspection
* [x] Column inspection
* [x] Missing-value analysis
* [x] Empty resume analysis
* [x] Duplicate analysis
* [x] Class distribution
* [x] Class imbalance analysis
* [x] Resume length analysis

## Phase 2 — EDA

* [x] Class distribution visualization
* [x] Resume character length
* [x] Resume word length
* [x] Word frequency analysis
* [x] WordCloud
* [x] Unigram analysis
* [x] Bigram analysis
* [x] Trigram analysis
* [x] Class-wise vocabulary / top terms

## Phase 3 — NLP Preprocessing

* [x] Lowercase normalization
* [x] HTML removal
* [x] URL handling
* [x] Email handling
* [x] Phone handling
* [x] Preservation of useful technical characters
* [x] Whitespace normalization
* [x] Reproducible preprocessing function

## Phase 4 — Feature Engineering

* [x] Train/validation/test split
* [x] TF-IDF representation
* [ ] TF-IDF hyperparameter comparison
* [ ] Word2Vec / embedding experiments

## Phase 5 — Machine Learning

* [ ] Logistic Regression
* [ ] Linear SVM
* [ ] Multinomial Naive Bayes
* [ ] Model comparison
* [ ] Final classical model selection

## Phase 6 — Deep Learning

* [ ] Tokenization
* [ ] Embedding
* [ ] LSTM / GRU
* [ ] Validation
* [ ] Final evaluation

## Phase 7 — Evaluation & Analysis

* [ ] Accuracy comparison
* [ ] Precision / Recall
* [ ] Macro-F1
* [ ] Weighted-F1
* [ ] Confusion Matrix
* [ ] Error analysis
* [ ] Final model selection

## Phase 8 — Final Demo

* [ ] Prediction pipeline
* [ ] Streamlit interface
* [ ] Test with unseen resume examples
* [ ] Final documentation

---

# 🎯 Final Goal

The final system aims to provide a complete, reproducible resume classification workflow:

```text
Raw Resume Dataset
        ↓
Data Quality Analysis
        ↓
EDA
        ↓
Text Preprocessing
        ↓
Feature Engineering
        ↓
Classical ML
        ↓
Deep Learning
        ↓
Model Evaluation
        ↓
Error Analysis
        ↓
Final Model
        ↓
Streamlit Demo
        ↓
Resume Category Prediction
```

---

# 👥 Team

### Team Name

**BugCreator**

### Project

**Resume Classifier — SAMATRIX ResumeForge 2026**

### Team Members

* **[Hemangi Mistry]**
* **[Diya Patel]**
* **[Jeel Patel]**

---

# 📜 License

This project is developed for educational and competition purposes as part of **SAMATRIX ResumeForge 2026**.
