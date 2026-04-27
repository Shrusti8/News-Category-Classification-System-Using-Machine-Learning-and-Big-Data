import pickle
import sys
import re
import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def load_artifacts(model_dir: str = '.'):
    with open(f'{model_dir}/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return pipeline, le


def predict_single(text: str, pipeline, le) -> dict:
    """Predict category and return confidence for a single news text."""
    cleaned = clean_text(text)
    proba = pipeline.predict_proba([cleaned])[0]
    pred_idx = np.argmax(proba)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = round(float(proba[pred_idx]) * 100, 2)

   
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {"category": le.inverse_transform([i])[0], "confidence": round(float(proba[i]) * 100, 2)}
        for i in top3_idx
    ]
    return {"prediction": pred_label, "confidence": confidence, "top3": top3}


if __name__ == '__main__':
    model_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    pipeline, le = load_artifacts(model_dir)
    print("Model loaded successfully.\n")


    print("Enter news headlines to classify (type 'quit' to exit):\n")
    while True:
        text = input("News text: ").strip()
        if text.lower() == 'quit':
            break
        result = predict_single(text, pipeline, le)
        print(f"  → Predicted: {result['prediction']} ({result['confidence']}% confidence)")
        print(f"  → Top 3: {result['top3']}\n")
