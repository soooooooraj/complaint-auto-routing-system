import json
import os
import faiss
import numpy as np

from pipeline.features import get_embedding, get_batch_embeddings

EMBEDDING_DIM = 768
INDEX_PATH = 'saved_models/faiss_index.bin'
MAPPING_PATH = 'saved_models/faiss_mapping.json'


def build_index(complaints_list=None):
    """
    Load complaints from data/complaints.json (or use provided list).
    Get embedding for each complaint's "text" field using get_batch_embeddings.
    Build a FAISS index with dimension 768.
    Save index to saved_models/faiss_index.bin
    Save complaint IDs mapping to saved_models/faiss_mapping.json
    """
    os.makedirs('saved_models', exist_ok=True)

    if complaints_list is None:
        with open('data/complaints.json', 'r', encoding='utf-8') as f:
            complaints_list = json.load(f)

    ids_mapping = {}
    texts = []

    for idx, complaint in enumerate(complaints_list):
        texts.append(complaint['text'])
        ids_mapping[str(idx)] = {
            "complaint_id": complaint['complaint_id'],
            "text": complaint['text'],
            "category": complaint['category'],
            "priority": complaint['priority']
        }

    print(f"Generating embeddings for {len(texts)} complaints...")
    embeddings = get_batch_embeddings(texts)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index (L2 distance)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(ids_mapping, f, indent=2, ensure_ascii=False)

    print(f"Index built with {index.ntotal} complaints")


def load_index():
    """
    Load faiss_index.bin and faiss_mapping.json.
    Return (index, mapping_dict).
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError("FAISS index files not found. Run build_index() first.")

    index = faiss.read_index(INDEX_PATH)

    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    return index, mapping


def find_similar(query_text, top_k=5, index=None, mapping=None):
    """
    Embed query_text using get_embedding, search FAISS index for top_k nearest.
    Return list of {complaint_id, text, category, priority, similarity_score}.
    """
    if index is None or mapping is None:
        loaded_index, loaded_mapping = load_index()
        index = index if index is not None else loaded_index
        mapping = mapping if mapping is not None else loaded_mapping

    query_vector = get_embedding(query_text)
    query_vector = np.array([query_vector]).astype('float32')

    distances, indices = index.search(query_vector, top_k)

    results = []
    for faiss_id, distance in zip(indices[0], distances[0]):
        if faiss_id == -1:
            continue
        mapped = mapping[str(faiss_id)]
        score = float(1 / (1 + distance))
        results.append({
            "complaint_id": mapped["complaint_id"],
            "text": mapped["text"],
            "category": mapped["category"],
            "priority": mapped["priority"],
            "similarity_score": round(score, 4)
        })

    return results


if __name__ == "__main__":
    # Build index
    build_index()

    # Test similarity search
    query = "water pipe burst flooding the street"
    results = find_similar(query, top_k=5)

    # Write results to file (avoids terminal encoding issues on Windows)
    with open('search_output.txt', 'w', encoding='utf-8') as f:
        f.write(f"Query: {query}\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i}. [{r['similarity_score']}] {r['complaint_id']} - {r['category']} ({r['priority']})\n")
            f.write(f"   {r['text']}\n\n")

    print("Done. Results written to search_output.txt")
