"""
essay_scoring_demo.py

Automated Essay Scoring demo pipeline:
- Simulate dataset OR load ASAP dataset (if available)
- Text cleaning: lowercase, simple tokenization, remove stopwords (optional), lemmatize (optional)
- Feature extraction: TF-IDF + simple lexical features + optional sentiment
- Models: LinearRegression, RandomForest; optional: BERT fine-tuning (requires transformers + datasets)
- Evaluation: RMSE, R^2, and Quadratic Weighted Kappa (QWK)
- Explainability: SHAP (if available) or permutation importance fallback
- FastAPI scaffold to serve the best model (optional)
"""

import os
import re
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, cohen_kappa_score
from sklearn.inspection import permutation_importance

# Optional libraries (guarded)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# Optional BERT (transformers)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Optional FastAPI
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

# Utilities for text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    # Attempt to ensure required NLTK data is present (will be no-op if already installed)
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        try:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        except Exception:
            pass
except Exception:
    NLTK_AVAILABLE = False


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ---------------------------
# Helper functions
# ---------------------------
def simple_clean(text: str, remove_stopwords: bool = True, lemmatize: bool = False) -> str:
    """
    Basic text cleaning:
    - lowercase
    - remove non-alphanumeric characters (retain spaces)
    - optional stopword removal and lemmatization (requires NLTK)
    """
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = re.sub(r"[^a-z0-9\s']", " ", txt)  # preserve apostrophes optionally
    tokens = txt.split()
    if remove_stopwords and NLTK_AVAILABLE:
        stop = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop]
    if lemmatize and NLTK_AVAILABLE:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def lexical_features(texts: List[str]) -> pd.DataFrame:
    """
    Compute simple lexical features per essay:
    - word_count, char_count, avg_word_len, unique_word_ratio
    - sentiment polarity (TextBlob) if available
    """
    rows = []
    for t in texts:
        if not isinstance(t, str) or len(t.strip()) == 0:
            words = []
        else:
            words = t.split()
        word_count = len(words)
        char_count = len(t)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
        unique_word_ratio = len(set(words)) / word_count if word_count > 0 else 0.0
        sentiment = None
        if TEXTBLOB_AVAILABLE:
            try:
                sentiment = TextBlob(t).sentiment.polarity
            except Exception:
                sentiment = None
        rows.append({
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_len': avg_word_len,
            'unique_word_ratio': unique_word_ratio,
            'sentiment': sentiment
        })
    return pd.DataFrame(rows)


def qwk(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = None, max_rating: int = None) -> float:
    """
    Quadratic Weighted Kappa using sklearn's cohen_kappa_score with weights='quadratic'.
    Because cohen_kappa_score requires discrete labels, we round predictions to nearest integer
    and clip to the allowed score range.
    """
    if min_rating is None:
        min_rating = int(min(y_true.min(), np.floor(y_pred.min())))
    if max_rating is None:
        max_rating = int(max(y_true.max(), np.ceil(y_pred.max())))
    # Round preds to integers in range
    y_pred_round = np.rint(y_pred).astype(int)
    y_pred_round = np.clip(y_pred_round, min_rating, max_rating)
    y_true_int = np.rint(y_true).astype(int)
    y_true_int = np.clip(y_true_int, min_rating, max_rating)
    return cohen_kappa_score(y_true_int, y_pred_round, weights='quadratic')


# ---------------------------
# Simulate dataset (small)
# ---------------------------
def simulate_essay_data(n_samples: int = 200, min_score: int = 0, max_score: int = 10) -> pd.DataFrame:
    """
    Simulate a simple essay dataset: 'essay_text' and 'score' (integer 0..10 by default).
    We simulate text by combining topic words and injecting length/quality signals.
    """
    topics = [
        "education school learning teacher student classroom study exam",
        "technology computer internet software hardware program data",
        "environment climate change pollution sustainability green",
        "health medicine fitness diet exercise wellbeing disease",
        "history war revolution empire culture society"
    ]
    samples = []
    for i in range(n_samples):
        topic = random.choice(topics)
        # base quality score correlated with 'quality words' and length
        base = random.gauss((min_score + max_score) / 2, 2)
        # create text: variable length and some "good/bad" words
        length = int(np.clip(np.random.normal(120, 40), 30, 400))  # chars
        # Decide on "quality" to influence score
        quality = np.random.choice(['low', 'med', 'high'], p=[0.2, 0.6, 0.2])
        if quality == 'low':
            quality_words = " bad poor weak unclear "
            base -= 1.8
        elif quality == 'high':
            quality_words = " strong excellent well-structured insightful "
            base += 1.8
        else:
            quality_words = " clear average "
        # Construct text
        words = (topic + " " + quality_words).split()
        # Repeat words to reach length (approx)
        text = " ".join(np.random.choice(words, size=max(5, length // 5)))
        # Add some punctuation and variation
        text = text.capitalize() + ". " + text + "."
        # Map base to discrete score within bounds
        score = int(np.clip(round(base + np.random.normal(0, 1.5)), min_score, max_score))
        samples.append({'essay_text': text, 'score': score})
    return pd.DataFrame(samples)


# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(simulate: bool = True, n_samples: int = 500):
    # 1) Load or simulate data
    if simulate:
        print("Simulating dataset...")
        df = simulate_essay_data(n_samples, min_score=0, max_score=10)
    else:
        # placeholder: user may load ASAP data here
        raise NotImplementedError("Loading a real dataset is not implemented in this demo. Set simulate=True.")

    # Quick view
    print("Dataset sample:")
    print(df.head())

    # 2) Clean text
    print("Cleaning text (lowercase, optional stopwords/lemmatize if NLTK installed)...")
    df['clean_text'] = df['essay_text'].apply(lambda t: simple_clean(t, remove_stopwords=True, lemmatize=False))

    # 3) Feature extraction
    print("Extract lexical features...")
    lex_df = lexical_features(df['clean_text'].tolist())
    # TF-IDF
    print("Compute TF-IDF features (max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(df['clean_text'])
    # Combine features into a single design matrix
    X_lex = lex_df.fillna(0)
    # We'll use a ColumnTransformer-like approach: combine TF-IDF (sparse) with dense lexical features
    from scipy.sparse import hstack
    X = hstack([X_tfidf, X_lex.values]).tocsr()
    y = df['score'].values

    # Train/test split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['score']
    )

    # 4) Models: LinearRegression and RandomForest
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    models = {'LinearRegression': lr, 'RandomForest': rf}
    eval_rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, preds)
        qwk_val = qwk(y_test, preds, min_rating=0, max_rating=10)
        eval_rows.append({'model': name, 'RMSE': rmse, 'R2': r2, 'QWK': qwk_val})
        print(f"\n{name} -> RMSE: {rmse:.3f}, R2: {r2:.3f}, QWK: {qwk_val:.3f}")

    eval_df = pd.DataFrame(eval_rows).sort_values('RMSE')
    best_name = eval_df.iloc[0]['model']
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    # 5) Explainability: SHAP (if available) or Permutation Importance
    if SHAP_AVAILABLE:
        try:
            print("\nRunning SHAP for RandomForest (can be slow)...")
            # For tree model use TreeExplainer for speed
            if isinstance(best_model, RandomForestRegressor):
                explainer = shap.TreeExplainer(best_model)
                shap_vals = explainer.shap_values(X_test)  # dense or sparse ok
                # aggregate top tfidf terms: get feature names
                tfidf_feature_names = tfidf.get_feature_names_out().tolist()
                lex_names = X_lex.columns.tolist()
                feature_names = list(tfidf_feature_names) + lex_names
                mean_abs = np.abs(shap_vals).mean(axis=0)
                shap_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
                print("\nTop SHAP features:")
                print(shap_df.head(15).to_string(index=False))
            else:
                print("SHAP currently used only for tree-based models in this demo.")
        except Exception as e:
            print("SHAP failed or too slow; falling back to permutation importance. Error:", e)
            SHAP_FALLBACK = True
        else:
            SHAP_FALLBACK = False
    else:
        SHAP_FALLBACK = True

    if SHAP_FALLBACK:
        print("\nComputing permutation importance (fallback)...")
        # If X_test is sparse (e.g., TF-IDF output), convert it to dense
        if hasattr(X_test, "toarray"):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = X_test

        res = permutation_importance(best_model, X_test_dense, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)

        # Build feature names similarly
        tfidf_feature_names = tfidf.get_feature_names_out().tolist()
        lex_names = X_lex.columns.tolist()
        feature_names = list(tfidf_feature_names) + lex_names
        imp_df = pd.DataFrame({'feature': feature_names, 'importance_mean': res.importances_mean}).sort_values('importance_mean', ascending=False)
        print("\nTop permutation-importances:")
        print(imp_df.head(20).to_string(index=False))

    # 6) Save best model and vectorizer for deployment
    import joblib
    os.makedirs('essay_model_artifacts', exist_ok=True)
    joblib.dump(best_model, os.path.join('essay_model_artifacts', 'best_model.joblib'))
    joblib.dump(tfidf, os.path.join('essay_model_artifacts', 'tfidf_vectorizer.joblib'))
    # Save lex feature columns
    X_lex.reset_index(drop=True, inplace=True)
    X_lex.to_csv(os.path.join('essay_model_artifacts', 'lexical_features_columns.csv'), index=False)
    print("\nSaved model artifacts under ./essay_model_artifacts")

    # 7) Optional: BERT fine-tuning scaffold (requires transformers & datasets & GPU ideally)
    if TRANSFORMERS_AVAILABLE:
        print("\nTransformers are available. BERT fine-tuning scaffold is below (not executed automatically).")
        print("To fine-tune BERT, run a separate script using the tokenizers and Trainer API, using df_train/df_test.")
        # Provide a minimal scaffold in comments or separate script in real use.
    else:
        print("\nTransformers not available — skipping BERT fine-tuning scaffold.")

    # Return objects for further use
    return {
        'df': df, 'X': X, 'y': y,
        'tfidf': tfidf, 'lex_df': X_lex,
        'models': models, 'best_model': best_model,
        'eval_df': eval_df
    }


# ---------------------------
# FastAPI scaffold (optional)
# ---------------------------
def fastapi_scaffold():
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. To create an API, install fastapi and uvicorn.")
        return
    import joblib
    app = FastAPI()
    model = joblib.load('essay_model_artifacts/best_model.joblib')
    tfidf = joblib.load('essay_model_artifacts/tfidf_vectorizer.joblib')
    lex_cols = pd.read_csv('essay_model_artifacts/lexical_features_columns.csv')

    class EssayIn(BaseModel):
        essay_text: str

    @app.post("/predict")
    def predict(item: EssayIn):
        txt = simple_clean(item.essay_text, remove_stopwords=True, lemmatize=False)
        lex = lexical_features([txt]).fillna(0)
        X_tfidf = tfidf.transform([txt])
        from scipy.sparse import hstack
        X_comb = hstack([X_tfidf, lex.values]).tocsr()
        pred = model.predict(X_comb)[0]
        pred_round = int(np.clip(round(pred), 0, 10))
        return {"pred_score_float": float(pred), "pred_score_int": pred_round}

    print("FastAPI scaffold created. Run with: uvicorn essay_scoring_demo:fastapi_scaffold --reload")
    return app


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    results = run_pipeline(simulate=True, n_samples=800)
    # Optionally run API scaffold if you installed FastAPI
    # api_app = fastapi_scaffold()
