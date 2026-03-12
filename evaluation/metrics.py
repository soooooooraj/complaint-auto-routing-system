import json
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from models.priority_classifier import predict as predict_priority
from models.eta_regressor import predict as predict_eta

# Constants
REPORTS_DIR = "evaluation/reports"
COMPLAINTS_FILE = "data/complaints.json"

def run_evaluation():
    # 0. Setup
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Load data
    with open(COMPLAINTS_FILE, 'r', encoding='utf-8') as f:
        complaints = json.load(f)
    
    # Split using same parameters as training (80/20, random_state=42)
    # We only care about the test set
    _, test_complaints = train_test_split(complaints, test_size=0.2, random_state=42)
    
    total = len(test_complaints)
    print(f"Beginning evaluation on {total} test complaints...")
    
    texts = [c['text'] for c in test_complaints]
    true_priorities = [c['priority'] for c in test_complaints]
    true_etas = [c['eta_days'] for c in test_complaints]
    categories = [c['category'] for c in test_complaints]
    
    pred_priorities = []
    pred_etas = []
    
    # 1. Generate Predictions
    # Note: Running one by one to simulate live inference using the predict functions
    # For large datasets, batching would be better, but following the spec for predict() functionality
    start_time = time.time()
    for i in range(total):
        # Priority Prediction
        p_res = predict_priority(texts[i])
        pred_priorities.append(p_res['priority'])
        
        # ETA Prediction (using true priority and category to evaluate the regressor isolation)
        e_val = predict_eta(texts[i], true_priorities[i], categories[i])
        pred_etas.append(e_val)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total}...")
            
    eval_duration = time.time() - start_time
    print(f"Predictions completed in {eval_duration:.2f} seconds.")

    # 2. Priority Classification Evaluation
    # 2.1 classification_report.txt
    report_text = classification_report(true_priorities, pred_priorities)
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("Priority Classification Report\n")
        f.write("="*30 + "\n")
        f.write(report_text)
    
    # 2.2 confusion_matrix.png
    plt.figure(figsize=(10, 8))
    labels = ["high", "medium", "low"]
    cm = confusion_matrix(true_priorities, pred_priorities, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Priority Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    # 3. ETA Regression Evaluation
    # 3.1 eta_error_distribution.png
    true_etas = np.array(true_etas)
    pred_etas = np.array(pred_etas)
    errors = pred_etas - true_etas
    mae = mean_absolute_error(true_etas, pred_etas)
    rmse = np.sqrt(mean_squared_error(true_etas, pred_etas))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=20, color='purple')
    plt.axvline(0, color='red', linestyle='--')
    plt.title('ETA Prediction Error Distribution')
    plt.xlabel('Error (Predicted - Actual) in Days')
    plt.ylabel('Frequency')
    plt.text(0.7, 0.9, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(os.path.join(REPORTS_DIR, "eta_error_distribution.png"))
    plt.close()

    # 4. Stress Test
    print("\nRunning stress test on ambiguous complaints...")
    stress_test_cases = [
        "The situation has been going on for a while now",
        "Things are not working properly in our area",
        "We need someone to look into this matter urgently",
        "This has been reported before but nothing happened",
        "The problem is getting worse every day",
        "Multiple residents have complained about this",
        "This is affecting our daily routine significantly",
        "We request immediate attention to this matter",
        "The authorities have been unresponsive so far",
        "This needs to be fixed as soon as possible"
    ]
    
    stress_results = []
    print("\n--- Stress Test Predictions ---")
    for text in stress_test_cases:
        res = predict_priority(text)
        print(f"Text: {text}")
        print(f"  -> Predicted: {res['priority']} (Confidence: {res['confidence']})")
        stress_results.append({
            "text": text,
            "prediction": res['priority'],
            "confidence": res['confidence']
        })

    # 5. evaluation_summary.json
    priority_accuracy = accuracy_score(true_priorities, pred_priorities)
    priority_f1 = f1_score(true_priorities, pred_priorities, average='weighted')
    
    summary = {
        "standard_test_accuracy": round(priority_accuracy, 4),
        "priority_f1_weighted": round(priority_f1, 4),
        "eta_mae": round(float(mae), 4),
        "eta_rmse": round(float(rmse), 4),
        "total_test_samples": total,
        "stress_test_predictions": stress_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(REPORTS_DIR, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary

if __name__ == "__main__":
    summary = run_evaluation()
    print("\nEvaluation complete")
    print("-" * 20)
    print(f"1. {os.path.join(REPORTS_DIR, 'classification_report.txt')}")
    print(f"2. {os.path.join(REPORTS_DIR, 'confusion_matrix.png')}")
    print(f"3. {os.path.join(REPORTS_DIR, 'eta_error_distribution.png')}")
    print(f"4. {os.path.join(REPORTS_DIR, 'evaluation_summary.json')}")
