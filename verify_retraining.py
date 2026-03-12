import joblib
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from models.priority_classifier import predict as predict_priority, MODEL_PATH as PRIO_PATH

# Load the retrained priority model
clf = joblib.load(PRIO_PATH)

test_cases = [
    ("Delay in issuing income certificate", "low"),
    ("No water supply for 3 days", "high"),
    ("Sewage overflow causing health hazard", "high"),
    ("Billing query for electricity bill", "low"),
    ("Large pothole on main road causing accidents", "medium"),
    ("Garbage not collected for a week", "medium"),
    ("Live wire fallen near school", "high"),
    ("Request for speed breaker on residential road", "low"),
    ("Water contamination health risk in colony", "high"),
    ("Property tax calculation dispute", "medium")
]

print(f"{'Text':<50} | {'Expected':<10} | {'Predicted':<10} | {'Confidence':<10} | {'Result':<10}")
print("-" * 100)

all_pass = True
for text, expected in test_cases:
    res = predict_priority(text, clf=clf)
    prio = res["priority"]
    conf = res["confidence"]
    
    status = "PASS" if prio == expected and conf > 0.45 else "FAIL"
    if status == "FAIL":
        all_pass = False
        
    print(f"{text[:50]:<50} | {expected:<10} | {prio:<10} | {conf:<10.4f} | {status:<10}")

print("-" * 100)
if all_pass:
    print("ALL 10 TEST CASES PASSED!")
else:
    print("SOME TEST CASES FAILED. CHECK DATA.")
