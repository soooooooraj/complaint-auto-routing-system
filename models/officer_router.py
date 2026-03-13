import json
import os
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.features import get_embedding

EMBEDDINGS_PATH = 'saved_models/officer_embeddings.pkl'
MAPPING_PATH = 'saved_models/officer_mapping.json'


def build_officer_index():
    """
    Load officers, build profile text for each, embed profiles, save to disk.
    """
    os.makedirs('saved_models', exist_ok=True)

    with open('data/officers.json', 'r', encoding='utf-8') as f:
        officers = json.load(f)

    embeddings = []
    for officer in officers:
        profile = (
            f"Department: {officer['department']}. "
            f"Skills: {', '.join(officer['skills'])}. "
            f"Experience: {officer['experience_years']} years."
        )
        emb = get_embedding(profile)
        embeddings.append(emb)

    embeddings = np.array(embeddings).astype('float32')

    joblib.dump(embeddings, EMBEDDINGS_PATH)

    with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(officers, f, indent=2, ensure_ascii=False)

    print(f"Officer index built with {len(officers)} officers")


def route_complaint(complaint_text, priority, category=None, top_k=3, embeddings=None, mapping=None):
    """
    Embed complaint, compare against officer embeddings using cosine similarity.
    Filter by predicted category first. Fallback to all if < 3 found.
    Filter out overloaded officers. Score with:
      60% semantic similarity + 30% performance + 10% workload availability.
    Return top_k officers.
    """
    officer_embeddings_all = embeddings if embeddings is not None else joblib.load(EMBEDDINGS_PATH)

    if mapping is None:
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            officers_all = json.load(f)
    else:
        officers_all = mapping

    # Step 1: Filter by category (Hard Filter)
    officers = officers_all
    officer_embeddings = officer_embeddings_all

    if category:
        filtered_indices = [i for i, o in enumerate(officers_all) if o['department'] == category]
        # Fallback: If < 3 officers in this dept, use all officers
        if len(filtered_indices) >= 3:
            officers = [officers_all[i] for i in filtered_indices]
            officer_embeddings = officer_embeddings_all[filtered_indices]
        else:
            print(f"Fallback: Only {len(filtered_indices)} officers in {category}. Using all officers.")

    complaint_emb = get_embedding(complaint_text)
    complaint_emb = np.array([complaint_emb]).astype('float32')

    # Cosine similarity between complaint and candidate officer profiles
    similarities = cosine_similarity(complaint_emb, officer_embeddings)[0]

    scored = []
    for idx, officer in enumerate(officers):
        # Filter out overloaded officers
        if officer['current_workload'] >= officer['max_workload']:
            continue

        sim_score = float(similarities[idx])
        perf_score = float(officer['performance_score'])

        # Workload availability: lower current workload relative to max = higher score
        workload_avail = 1.0 - (officer['current_workload'] / officer['max_workload'])

        # Combined score
        final_score = (0.60 * sim_score) + (0.30 * perf_score) + (0.10 * workload_avail)

        scored.append({
            "officer_id": officer['officer_id'],
            "name": officer['name'],
            "department": officer['department'],
            "similarity_score": round(sim_score, 4),
            "final_score": round(final_score, 4),
            "current_workload": officer['current_workload']
        })

    # Sort by final_score descending
    scored.sort(key=lambda x: x['final_score'], reverse=True)

    return scored[:top_k]


if __name__ == "__main__":
    import subprocess, sys

    code = '''
import sys, json
sys.path.insert(0, '.')
from models.officer_router import build_officer_index, route_complaint

build_officer_index()

test_cases = [
    ("Water pipe burst flooding the street", "high"),
    ("Illegal encroachment on public land", "low"),
]

for text, priority in test_cases:
    print(f"\\nComplaint: {text}")
    print(f"Priority: {priority}")
    print("-" * 50)
    results = route_complaint(text, priority, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['officer_id']} - {r['name']}")
        print(f"     Dept: {r['department']}, Similarity: {r['similarity_score']}, Score: {r['final_score']}, Workload: {r['current_workload']}")
'''

    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, encoding='utf-8', cwd='.'
    )

    with open('router_output.txt', 'w', encoding='utf-8') as f:
        f.write(result.stdout)

    print("Done. Exit code:", result.returncode)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:] if result.stderr else "none")
