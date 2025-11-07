# Automated Essay Scoring (AES) — Demo Project

**Goal:** Build an interpretable machine learning pipeline that predicts essay scores automatically using **Natural Language Processing (NLP)** and **supervised learning**.  
This demo simulates the **Automated Student Assessment Prize (ASAP)** competition setup.

---

## Overview

This project implements a simplified **Automated Essay Scoring (AES)** system capable of:
- Preprocessing and cleaning raw essay text
- Extracting meaningful features (TF-IDF, word count, sentiment)
- Training multiple models (Linear Regression, Random Forest, optional BERT)
- Evaluating model performance using **Quadratic Weighted Kappa (QWK)** and **RMSE**
- Providing interpretability with **feature importance** and **SHAP explainability**
- Optional deployment readiness with **FastAPI endpoint**

---

## File Structure

Automated_Essay_Scoring/
│
├── main.py                     # Main script containing the entire AES pipeline
├── requirements.txt             # Dependencies list
├── README.md                    # Project documentation
├── data/                        # (Optional) Folder for essay datasets
│   └── sample_data.csv          # Example dataset or generated data
├── models/                      # Trained model outputs (saved with joblib)
│   └── best_model.joblib
├── reports/                     # Generated results and visualizations
│   ├── feature_importance.png
│   ├── shap_summary_plot.png
│   └── model_results.txt
└── utils/                       # (Optional) Utility modules for data prep, viz, etc.

````

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Automated_Essay_Scoring.git
   cd Automated_Essay_Scoring
````

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On macOS/Linux
   venv\Scripts\activate         # On Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Option 1 — Simulate Essay Data (default)

```bash
python main.py
```

### Option 2 — Use Your Own Dataset

Modify the following in `main.py`:

```python
results = run_pipeline(simulate=False, data_path="data/your_dataset.csv")
```

### Outputs:

* Model evaluation metrics printed to console
* Visualizations saved to `/reports/`
* Trained model saved under `/models/`

---

## Pipeline Summary

| Stage                   | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| **Data Loading**        | Loads either synthetic essays or your CSV dataset              |
| **Text Preprocessing**  | Lowercasing, stopword removal, lemmatization                   |
| **Feature Engineering** | TF-IDF vectorization, word count, sentiment scores             |
| **Model Training**      | Compares Linear Regression, Random Forest, and optionally BERT |
| **Evaluation**          | Calculates **QWK**, **RMSE**, and cross-validation scores      |
| **Explainability**      | Uses `permutation_importance` and `SHAP` for interpretability  |

---

## Explainability

### 1. Feature Importance

* Computed using sklearn’s `permutation_importance()`
* Automatically converts sparse TF-IDF matrices to dense arrays for compatibility
* Visualizes top contributing features to essay score prediction

### 2. SHAP Summary

* Provides global interpretability (which words contribute most)
* Optional per-essay breakdown using SHAP’s force plots

---

##  Example Output

**Console Output:**

```
Running Automated Essay Scoring Pipeline...
Simulating dataset with 800 essays...
Training RandomForestRegressor...
RMSE: 1.21
QWK: 0.83
Best model: RandomForestRegressor
Feature importance plot saved to reports/feature_importance.png
SHAP analysis completed.
```

**Feature Importance Plot:**
![Feature Importance](reports/feature_importance.png)

**SHAP Summary Plot:**
![SHAP Summary](reports/shap_summary_plot.png)

---

## Example Fairness & Explainability Summary

| Metric            | Description                                                                  | Example Value                      |
| ----------------- | ---------------------------------------------------------------------------- | ---------------------------------- |
| RMSE              | Root Mean Squared Error — measures average prediction deviation              | **1.21**                           |
| QWK               | Quadratic Weighted Kappa — measures agreement between model and human raters | **0.83**                           |
| Feature Influence | Most impactful essay terms influencing scores                                | “coherence”, “argument”, “grammar” |
| Fairness Check    | Correlation between score and essay length / topic                           | No significant bias detected       |

> **Interpretation:**
> The model performs consistently across essay lengths and topics, with higher scores correlated with logical coherence and vocabulary richness. SHAP analysis shows grammatical accuracy and argument quality as dominant predictors.

---

## Tech Stack

* **Language:** Python 3.9+
* **Libraries:**
  `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `textblob`, `shap`, `joblib`
* **ML Techniques:** TF-IDF, sentiment scoring, regression models, explainable AI
* **Optional Extensions:** BERT fine-tuning (`transformers`), API serving (`FastAPI`)

---

## Next Steps (Stretch Goals)

* Add **FastAPI service** for live essay scoring endpoint
* Integrate **BERT-based embeddings** for semantic understanding
* Incorporate **bias & fairness audits** by demographic or topic clusters
* Optimize for **real-time inference** and deployment

---

## Citation

This project structure and evaluation metric (QWK) are inspired by the **Automated Student Assessment Prize (ASAP)** competition hosted on **Kaggle**.

---

## Author

**Shun Le Yi Mon**
