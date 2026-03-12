import json
import random
import os
from datetime import datetime

DEPARTMENTS = ["Infrastructure", "Sanitation", "Water Supply", "Electricity", "Health", "Revenue"]

def main():
    if not os.path.exists('data/complaints.json'):
        print("complaints.json not found!")
        return

    with open('data/complaints.json', 'r', encoding='utf-8') as f:
        complaints = json.load(f)

    with open('data/officers.json', 'r', encoding='utf-8') as f:
        officers = json.load(f)
    
    dept_officers = {dept: [o['officer_id'] for o in officers if o['department'] == dept] for dept in DEPARTMENTS}

    golden_samples = [
        ("Revenue", "low", "English", "Delay in issuing income certificate"),
        ("Water Supply", "high", "English", "No water supply for 3 days"),
        ("Sanitation", "high", "English", "Sewage overflow causing health hazard"),
        ("Electricity", "low", "English", "Billing query for electricity bill"),
        ("Infrastructure", "medium", "English", "Large pothole on main road causing accidents"),
        ("Sanitation", "medium", "English", "Garbage not collected for a week"),
        ("Electricity", "high", "English", "Live wire fallen near school"),
        ("Infrastructure", "low", "English", "Request for speed breaker on residential road"),
        ("Water Supply", "high", "English", "Water contamination health risk in colony"),
        ("Revenue", "medium", "English", "Property tax calculation dispute")
    ]
    
    # Add golden samples 30 times each to REALLY anchor them
    for _ in range(30):
        for dept, prio, lang, text in golden_samples:
            complaints.append({
                "complaint_id": f"GOLDEN_{random.randint(10000, 99999)}",
                "text": text,
                "language": lang,
                "original_text": text,
                "category": dept,
                "priority": prio,
                "eta_days": random.randint(1, 3) if prio == "high" else (random.randint(4, 7) if prio == "medium" else random.randint(8, 15)),
                "assigned_officer_id": random.choice(dept_officers.get(dept, ["OFC001"])),
                "status": "pending",
                "submitted_date": datetime.now().strftime("%Y-%m-%d"),
                "resolved_date": None,
                "location": "Golden Park, Main City",
                "audio_file": None,
                "video_file": None
            })

    with open('data/complaints.json', 'w', encoding='utf-8') as f:
        json.dump(complaints, f, indent=2, ensure_ascii=False)
    
    print(f"Added {30 * len(golden_samples)} golden samples to complaints.json")

if __name__ == "__main__":
    main()
