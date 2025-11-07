
# Automated Essay Scoring Demo (`main.py`)

## Project Overview

This project demonstrates an **Automated Essay Scoring (AES)** pipeline using classical machine learning techniques and optional deep learning (BERT). It covers the full workflow from text cleaning, feature extraction, modeling, evaluation, explainability, and deployment scaffolds.

**Key Features:**

- Simulate a dataset or load a real essay dataset (ASAP)
- Text preprocessing: lowercase, optional stopwords removal, optional lemmatization
- Feature extraction: TF-IDF, lexical features, optional sentiment
- Models: Linear Regression, Random Forest (optional: BERT fine-tuning)
- Evaluation metrics: RMSE, R², Quadratic Weighted Kappa (QWK)
- Explainability: SHAP or permutation importance fallback
- Deployment scaffold: FastAPI API for serving predictions
- Model artifacts saved for reproducibility

---

## Requirements

```txt
numpy>=1.23
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
nltk>=3.8
textblob>=0.17
shap>=0.44
torch>=2.0           # optional, for BERT
transformers>=4.35   # optional, for BERT
fastapi>=0.103       # optional, for API
uvicorn>=0.23        # optional, for API
````

> ⚠️ NLTK resources must be downloaded once:
>
> ```python
> import nltk
> nltk.download('stopwords')
> nltk.download('wordnet')
> nltk.download('omw-1.4')
> ```

---

## File Structure

```
AutomatedEssayScoring/
├── main.py                     # Full AES pipeline
├── essay_model_artifacts/      # Saved model, TF-IDF vectorizer, lexical feature columns
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── lexical_features_columns.csv
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Usage

Run the full pipeline with simulated dataset:

```bash
python main.py
```

Optional: run FastAPI API scaffold (requires FastAPI and uvicorn):

```bash
uvicorn main:fastapi_scaffold --reload
```

---

## Example Outputs

**Dataset Sample (simulated)**

| essay_text                                   | score |
| -------------------------------------------- | ----- |
| "Education school learning strong..."        | 7     |
| "Technology software program clear..."       | 5     |
| "Environment climate pollution excellent..." | 8     |

**Model Evaluation (Test Set)**

| model            | RMSE | R²   | QWK  |
| ---------------- | ---- | ---- | ---- |
| RandomForest     | 1.22 | 0.78 | 0.82 |
| LinearRegression | 1.75 | 0.55 | 0.60 |

**Explainability Output (Top Features from Permutation Importance / SHAP)**

| feature             | importance_mean / mean_abs_shap |
| ------------------- | ------------------------------- |
| "strong"            | 0.082                           |
| "excellent"         | 0.075                           |
| "word_count"        | 0.063                           |
| "unique_word_ratio" | 0.052                           |
| "clear"             | 0.049                           |

---

## Pipeline Details

1. **Text Cleaning**

* Lowercasing, removing non-alphanumeric characters
* Optional stopword removal and lemmatization using NLTK

2. **Feature Extraction**

* TF-IDF (1-2 grams)
* Lexical features: word count, character count, average word length, unique word ratio
* Optional sentiment analysis via TextBlob

3. **Model Training**

* Linear Regression
* Random Forest (Tree-based model suitable for SHAP)
* Optional: BERT fine-tuning scaffold (requires Transformers & GPU)

4. **Evaluation Metrics**

* **RMSE** – root mean squared error
* **R²** – coefficient of determination
* **Quadratic Weighted Kappa (QWK)** – standard for scoring tasks

5. **Explainability**

* SHAP for Random Forest (TreeExplainer)
* Permutation importance fallback if SHAP unavailable or model unsupported

6. **Deployment**

* FastAPI scaffold to serve predictions from saved model
* Input: `essay_text`, Output: predicted score (float & rounded integer)

---

## Fairness & Interpretability Notes

* Lexical features and TF-IDF terms help understand **which words contribute most to scores**
* SHAP/Permutation importance can highlight **biases or unexpected predictors**
* For real datasets, monitor **score distribution across demographic groups** to check potential unfairness

---

## Artifacts Saved

* `best_model.joblib` – trained scikit-learn model
* `tfidf_vectorizer.joblib` – TF-IDF vectorizer
* `lexical_features_columns.csv` – names of lexical features used in training

---

## References

* [ASAP Automated Student Assessment Prize](https://www.kaggle.com/c/asap-aes)
* SHAP: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
* Quadratic Weighted Kappa: [[https://en.wikipedia.org/wiki/Cohen%27s_kappa#Quadratic_weighted_kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa#Quadratic_weighted_kappa))
