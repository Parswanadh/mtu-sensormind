import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_knowledge_base():
    base_dir = os.path.dirname(__file__)
    rules_path = os.path.join(base_dir, 'maintenance_rules.txt')
    index_path = os.path.join(base_dir, 'faiss_index.bin')
    corpus_path = os.path.join(base_dir, 'rules_corpus.pkl')

    if not os.path.exists(rules_path):
        print(f"Error: {rules_path} not found.")
        return

    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    # Using a small, fast model ideal for this CPU/local task
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Reading rules...")
    with open(rules_path, 'r', encoding='utf-8') as f:
        # Filter out empty lines
        rules = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(rules)} rules. Generating embeddings...")
    embeddings = model.encode(rules, convert_to_numpy=True)

    print("Building FAISS index...")
    # L2 distance index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("Saving FAISS index and corpus...")
    faiss.write_index(index, index_path)
    
    with open(corpus_path, 'wb') as f:
        pickle.dump(rules, f)

    print("Knowledge base successfully built and saved!")

if __name__ == "__main__":
    build_knowledge_base()
