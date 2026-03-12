import json
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from pipeline.features import get_embedding, get_batch_embeddings

MODEL_PATH = 'saved_models/eta_model.pkl'
ENCODER_PATH = 'saved_models/category_encoder.pkl'

PRIORITY_MAP = {"high": 3, "medium": 2, "low": 1}


def _build_features(embeddings, priorities, categories, cat_encoder, fit_encoder=False):
    """Combine embeddings with encoded priority and category into a single feature matrix."""
    priority_encoded = np.array([PRIORITY_MAP[p] for p in priorities]).reshape(-1, 1)

    if fit_encoder:
        category_encoded = cat_encoder.fit_transform(categories).reshape(-1, 1)
    else:
        category_encoded = cat_encoder.transform(categories).reshape(-1, 1)

    X = np.hstack([np.array(embeddings).astype('float32'), priority_encoded, category_encoded])
    return X


def train():
    """
    Load complaints, build combined features (embedding + priority + category),
    train Random Forest Regressor on eta_days. 80/20 split. Save model + encoder.
    """
    os.makedirs('saved_models', exist_ok=True)

    with open('data/complaints.json', 'r', encoding='utf-8') as f:
        complaints = json.load(f)

    texts = [c['text'] for c in complaints]
    priorities = [c['priority'] for c in complaints]
    categories = [c['category'] for c in complaints]
    y = np.array([c['eta_days'] for c in complaints], dtype='float32')

    print(f"Generating embeddings for {len(texts)} complaints...")
    embeddings = get_batch_embeddings(texts)

    cat_encoder = LabelEncoder()
    X = _build_features(embeddings, priorities, categories, cat_encoder, fit_encoder=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training Random Forest Regressor on {len(X_train)} samples, testing on {len(X_test)}...")
    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nMAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    joblib.dump(reg, MODEL_PATH)
    joblib.dump(cat_encoder, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")


def predict(text, priority, category, reg=None, cat_encoder=None):
    """
    Load saved model + encoder, build feature vector, predict eta_days.
    Returns predicted eta_days as a rounded integer.
    """
    if reg is None:
        reg = joblib.load(MODEL_PATH)
    if cat_encoder is None:
        cat_encoder = joblib.load(ENCODER_PATH)

    embedding = get_embedding(text)
    X = _build_features([embedding], [priority], [category], cat_encoder, fit_encoder=False)

    predicted = reg.predict(X)[0]
    return int(round(predicted))


if __name__ == "__main__":
    import subprocess, sys

    # Run train + predict in subprocess to capture output cleanly
    code = '''
import sys, json
sys.path.insert(0, '.')
from models.eta_regressor import train, predict

train()

test_cases = [
    ("Sewage overflow causing health hazard", "high", "Sanitation"),
    ("Billing query for last month", "low", "Revenue"),
    ("No water supply for 3 days", "high", "Water Supply"),
]

print("\\n--- ETA Predictions ---")
for text, priority, category in test_cases:
    eta = predict(text, priority, category)
    print(f"Text: {text}")
    print(f"  Priority: {priority}, Category: {category}")
    print(f"  -> Predicted ETA: {eta} days\\n")
'''

    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, encoding='utf-8', cwd='.'
    )

    with open('eta_output.txt', 'w', encoding='utf-8') as f:
        f.write(result.stdout)

    print("Done. Exit code:", result.returncode)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:] if result.stderr else "none")
