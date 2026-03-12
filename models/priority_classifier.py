import json
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

from pipeline.features import get_embedding, get_batch_embeddings

MODEL_PATH = 'saved_models/priority_model.pkl'


def train():
    """
    Load complaints, embed texts, train Random Forest on priority labels.
    80/20 split, random_state=42. 5-fold CV. Save model with joblib.
    """
    os.makedirs('saved_models', exist_ok=True)

    with open('data/complaints.json', 'r', encoding='utf-8') as f:
        complaints = json.load(f)

    texts = [c['text'] for c in complaints]
    labels = [c['priority'] for c in complaints]

    print(f"Generating embeddings for {len(texts)} complaints...")
    embeddings = get_batch_embeddings(texts)
    X = np.array(embeddings).astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    # 5-fold cross validation
    print("Running 5-fold cross validation...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training set
    print(f"\nTraining Random Forest on {len(X_train)} samples, testing on {len(X_test)}...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def predict(text, clf=None):
    """
    Load saved model, embed input text, predict priority with confidence.
    """
    if clf is None:
        clf = joblib.load(MODEL_PATH)
    embedding = get_embedding(text)
    embedding = np.array([embedding]).astype('float32')

    predicted = clf.predict(embedding)[0]
    probas = clf.predict_proba(embedding)[0]
    confidence = float(max(probas))

    return {"priority": predicted, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    # Train
    train()

    # Test predictions
    test_texts = [
        "Sewage overflow on main road causing health hazard",
        "Billing query for last month invoice",
        "No water supply for 3 days in entire colony"
    ]

    print("\n--- Predictions ---")
    for text in test_texts:
        result = predict(text)
        print(f"Text: {text}")
        print(f"  -> Priority: {result['priority']} (confidence: {result['confidence']})\n")
