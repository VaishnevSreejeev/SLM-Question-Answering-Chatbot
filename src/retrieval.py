from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # Use a better embedding model

def get_embeddings(texts):
    """
    Generates embeddings for a list of texts.
    """
    return model.encode(texts)

def retrieve_relevant_chunk(question, chunks, chunk_embeddings):
    """
    Retrieves the most relevant chunk for a given question based on cosine similarity.
    """
    question_embedding = get_embeddings([question])
    similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()
    
    print("Similarities:", similarities)  # Debugging
    
    # Lower the similarity threshold to allow weaker matches
    most_relevant_idx = similarities.argmax()
    if similarities[most_relevant_idx] < 0.3:  # Adjust threshold as needed
        print("No sufficiently relevant chunk found.")
        return None
    
    return chunks[most_relevant_idx]