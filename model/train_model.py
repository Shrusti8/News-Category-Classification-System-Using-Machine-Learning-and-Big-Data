import os
import re
import string
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


CATEGORY_MAP = {
   
    'THE WORLDPOST':  'WORLD NEWS',
    'WORLDPOST':      'WORLD NEWS',
  
    'ARTS':           'ARTS & CULTURE',
    'CULTURE & ARTS': 'ARTS & CULTURE',
    
    'STYLE':          'STYLE & BEAUTY',
   
    'HEALTHY LIVING': 'WELLNESS',
   
    'PARENTS':        'PARENTING',
  
    'TASTE':          'FOOD & DRINK',
    
    'FIFTY':          'WOMEN',
   
    'U.S. NEWS':      'POLITICS',
    
    'GOOD NEWS':      'IMPACT',
}

def merge_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Apply category merging to reduce overlapping classes."""
    original = df['category'].nunique()
    df['category'] = df['category'].replace(CATEGORY_MAP)
    merged = df['category'].nunique()
    print(f"📦 Categories: {original} → {merged} after merging similar ones")
    return df



lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Clean and normalize raw news text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset; supports CSV and JSON lines format."""
    if filepath.endswith('.json') or filepath.endswith('.jsonl'):
        df = pd.read_json(filepath, lines=True)
    else:
        df = pd.read_csv(filepath)

    if 'headline' in df.columns and 'short_description' in df.columns:
        df['text'] = df['headline'].fillna('') + ' ' + df['short_description'].fillna('')
    elif 'text' not in df.columns:
        raise ValueError("Dataset must have a 'text' column or 'headline'+'short_description' columns.")

    if 'category' not in df.columns:
        raise ValueError("Dataset must have a 'category' column.")

    df = df.dropna(subset=['text', 'category'])

    # Merge overlapping categories
    df = merge_categories(df)

    print("\n📊 Class Distribution:")
    print(df['category'].value_counts())

    print("\n🧹 Cleaning text...")
    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.strip() != '']

    return df


def build_pipeline() -> Pipeline:
    """
    Pipeline with regularized TF-IDF + Logistic Regression.
    Key anti-overfitting settings:
      - max_features=30000  : larger vocab for 32 clean classes
      - min_df=2            : ignores very rare words (noise)
      - max_df=0.95         : ignores words in almost every doc
      - sublinear_tf=True   : dampens high-frequency word dominance
      - C=1.0               : balanced regularization
      - class_weight=balanced: handles class imbalance
    """
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        analyzer='word'
    )

    classifier = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

    return Pipeline([
        ('tfidf', vectorizer),
        ('clf', classifier)
    ])


def evaluate_with_cv(pipeline: Pipeline, X, y, n_splits: int = 5):
    """Stratified K-Fold cross-validation to detect overfitting."""
    print(f"\nRunning {n_splits}-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_accuracy = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    cv_f1 = cross_val_score(pipeline, X, y, cv=skf, scoring='f1_weighted', n_jobs=-1)

    print(f"  CV Accuracy : {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  CV F1 Score : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    gap_threshold = 0.05
    if cv_accuracy.std() > gap_threshold:
        print("High variance across folds — possible overfitting. Consider more regularization.")
    else:
        print("Low variance — model generalizes well.")

    return cv_accuracy.mean(), cv_f1.mean()

def train_and_evaluate(df: pd.DataFrame, output_dir: str = '.'):
    X = df['text_clean'].values
    y = df['category'].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    pipeline = build_pipeline()

    # Cross-validation on training set
    evaluate_with_cv(pipeline, X_train, y_train)

    # Train on full training set
    print("\nTraining final model...")
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    train_pred = pipeline.predict(X_train)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nTraining Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy     : {test_acc:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")

    overfit_gap = train_acc - test_acc
    print(f"Overfit Gap       : {overfit_gap:.4f}", end=" ")
    if overfit_gap > 0.10:
        print("Significant overfitting detected!")
    elif overfit_gap > 0.05:
        print("Mild overfitting — monitor carefully.")
    else:
        print("Acceptable generalization.")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model artifacts
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'pipeline.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    print(f"\nModel saved to '{output_dir}/pipeline.pkl'")
    print(f"Label encoder saved to '{output_dir}/label_encoder.pkl'")

    return pipeline, le

if __name__ == '__main__':
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else '../dataset/news.json'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    print(f"Loading dataset from: {dataset_path}")
    df = load_data(dataset_path)
    print(f"Loaded {len(df):,} samples with {df['category'].nunique()} categories.\n")

    pipeline, le = train_and_evaluate(df, output_dir=output_dir)
