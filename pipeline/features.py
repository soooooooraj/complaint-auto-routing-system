from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Lazy load model
_embedding_model = None

def load_model():
    """
    Load sentence-transformers model "paraphrase-multilingual-mpnet-base-v2" locally.
    This model runs fully offline. Cache it so it only loads once.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformer 'paraphrase-multilingual-mpnet-base-v2' (offline caching)...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _embedding_model

def get_embedding(text: str):
    """
    Takes a clean English text string.
    Returns a numpy array embedding using the loaded model.
    """
    model = load_model()
    return model.encode(text)

def get_batch_embeddings(texts_list: list):
    """
    Takes a list of texts.
    Returns a list of numpy arrays (or a 2D numpy array) using batch encoding.
    """
    model = load_model()
    # By default, encode() handles lists and returns a 2D numpy array
    return model.encode(texts_list)

if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Pair 1 (dissimilar)
    text1_pair1 = "Road has a large pothole near the school"
    text2_pair1 = "Water supply is disrupted for 3 days"
    
    # Pair 2 (similar)
    text1_pair2 = "Pothole on the road causing accidents"
    text2_pair2 = "Large crater on street damaging vehicles"
    
    # Test batch embedding
    print("Testing get_batch_embeddings on Pair 1...")
    embeddings_pair1 = get_batch_embeddings([text1_pair1, text2_pair1])
    
    print(f"Embedding shape for text 1: {embeddings_pair1[0].shape}")
    print(f"Embedding shape for text 2: {embeddings_pair1[1].shape}")
    
    sim1 = cosine_similarity([embeddings_pair1[0]], [embeddings_pair1[1]])[0][0]
    print(f"Cosine similarity between Pair 1: {sim1:.4f}\n")
    
    # Test pair 2
    print("Testing get_batch_embeddings on Pair 2...")
    embeddings_pair2 = get_batch_embeddings([text1_pair2, text2_pair2])
    
    sim2 = cosine_similarity([embeddings_pair2[0]], [embeddings_pair2[1]])[0][0]
    print(f"Cosine similarity between Pair 2: {sim2:.4f}")
