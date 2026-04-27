import os
import re
import string
import pickle
import logging
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app) 


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Identical preprocessing as used during model training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(os.path.dirname(__file__), '..', 'model'))

try:
    with open(os.path.join(MODEL_DIR, 'pipeline.pkl'), 'rb') as f:
        pipeline = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    logger.info("Model and label encoder loaded successfully.")
except FileNotFoundError:
    logger.error("Model files not found. Run model/train_model.py first.")
    pipeline = None
    label_encoder = None




@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "model_loaded": pipeline is not None,
        "categories": label_encoder.classes_.tolist() if label_encoder else []
    })


@app.route('/predict', methods=['POST'])
def predict():
    
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Request body must include a 'text' field."}), 400

    raw_text = data['text']

    
    if not isinstance(raw_text, str) or len(raw_text.strip()) < 5:
        return jsonify({"error": "Text must be a non-empty string with at least 5 characters."}), 400

    if len(raw_text) > 10000:
        raw_text = raw_text[:10000] 

   
    cleaned = clean_text(raw_text)
    if not cleaned:
        return jsonify({"error": "Text could not be processed after cleaning."}), 400

   
    try:
        proba = pipeline.predict_proba([cleaned])[0]
        pred_idx = int(np.argmax(proba))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = round(float(proba[pred_idx]) * 100, 2)

        
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {
                "category": label_encoder.inverse_transform([int(i)])[0],
                "confidence": round(float(proba[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        logger.info(f"Predicted: {pred_label} ({confidence}%) for text: '{raw_text[:60]}...'")

        return jsonify({
            "prediction": pred_label,
            "confidence": confidence,
            "top3": top3,
            "cleaned_text": cleaned  
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed. Check server logs."}), 500


@app.route('/categories', methods=['GET'])
def get_categories():
    
    if label_encoder is None:
        return jsonify({"error": "Model not loaded."}), 503
    return jsonify({"categories": label_encoder.classes_.tolist()})


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    
    if pipeline is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(silent=True)
    if not data or 'texts' not in data or not isinstance(data['texts'], list):
        return jsonify({"error": "Request body must include a 'texts' array."}), 400

    texts = data['texts'][:50]  # Limit batch size
    cleaned_texts = [clean_text(t) for t in texts]
    valid_mask = [bool(c.strip()) for c in cleaned_texts]

    results = []
    for i, (raw, cleaned, valid) in enumerate(zip(texts, cleaned_texts, valid_mask)):
        if not valid:
            results.append({"index": i, "error": "Could not process text."})
            continue
        proba = pipeline.predict_proba([cleaned])[0]
        pred_idx = int(np.argmax(proba))
        results.append({
            "index": i,
            "prediction": label_encoder.inverse_transform([pred_idx])[0],
            "confidence": round(float(proba[pred_idx]) * 100, 2)
        })

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
