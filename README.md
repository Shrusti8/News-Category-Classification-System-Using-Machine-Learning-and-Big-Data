# 📰 News Category Classification System

A full-stack news classification system with anti-overfitting improvements across every layer.

## 🗂️ Project Structure

```
News-Category-Classification-System/
├── backend/
│   ├── app.py              # Flask REST API with input preprocessing
│   └── requirements.txt
├── bigdata/
│   └── spark_preprocess.py # PySpark pipeline for large-scale data
├── dataset/
│   ├── prepare_dataset.py  # Cleaning, augmentation, stratified split
│   └── README.md
├── frontend/
│   └── index.html          # Web UI with single + batch prediction
├── model/
│   ├── train_model.py      # Training with CV and regularization
│   └── evaluate_model.py   # Interactive evaluation
└── README.md
```

---

## ✅ Overfitting Fixes Applied

| Layer    | Fix |
|----------|-----|
| Model    | `C=0.5` L2 regularization in Logistic Regression |
| Model    | `class_weight='balanced'` for imbalanced categories |
| Model    | TF-IDF: `max_features=15000`, `min_df=2`, `max_df=0.95` |
| Model    | `sublinear_tf=True` to dampen high-frequency words |
| Training | 5-Fold Stratified Cross-Validation |
| Training | Overfit gap monitoring (train acc vs test acc) |
| Dataset  | Text augmentation for minority classes |
| Dataset  | Stratified train/val/test split |
| Backend  | Same text cleaning applied at inference time |
| BigData  | Spark `regParam=0.1` L2 + Cross-Validation |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Download dataset
Download the HuffPost News Category Dataset from Kaggle:
https://www.kaggle.com/datasets/rmisra/news-category-dataset

Save it as `dataset/news.json`

### 3. Prepare dataset
```bash
cd dataset
python prepare_dataset.py news.json .
```

### 4. Train model
```bash
cd model
python train_model.py ../dataset/news.json .
```

### 5. Start backend
```bash
cd backend
python app.py
# Server runs at http://localhost:5000
```

### 6. Open frontend
Open `frontend/index.html` in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint        | Description |
|--------|-----------------|-------------|
| GET    | `/`             | Health check + loaded categories |
| POST   | `/predict`      | Classify a single news text |
| POST   | `/batch_predict`| Classify multiple texts at once |
| GET    | `/categories`   | List all available categories |

### Example request
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists find new vaccine for influenza"}'
```

### Example response
```json
{
  "prediction": "WELLNESS",
  "confidence": 87.34,
  "top3": [
    {"category": "WELLNESS", "confidence": 87.34},
    {"category": "SCIENCE", "confidence": 8.21},
    {"category": "HEALTHY LIVING", "confidence": 2.14}
  ]
}
```

---

## 📊 Evaluating Your Model

After training, check for overfitting:
```bash
cd model
python evaluate_model.py .
```

A **train accuracy - test accuracy gap > 10%** means overfitting.
Increase regularization by lowering `C` in `train_model.py`.

---

## 🔧 Tuning Tips

| Issue | Fix |
|-------|-----|
| Still overfitting | Lower `C` (e.g. `C=0.1`) |
| Underfitting / low accuracy | Raise `C` (e.g. `C=1.0`) or increase `max_features` |
| Class imbalance | `class_weight='balanced'` (already applied) |
| Slow training | Use `solver='saga'` with `n_jobs=-1` |

---

## 🛠️ Tech Stack

- **ML**: scikit-learn, NLTK
- **Backend**: Flask, Flask-CORS
- **Big Data**: PySpark
- **Frontend**: HTML, CSS, Vanilla JS
