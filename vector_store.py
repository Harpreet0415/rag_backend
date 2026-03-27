import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []  # list of chunk dicts
        self.dimension = 384  # MiniLM embedding dimension

    def build_index(self, chunks: List[Dict]) -> None:
        """Build FAISS index from text chunks."""
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product = cosine after normalization
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for most relevant chunks given a query."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)
        
        return results

    def save(self, path: str) -> None:
        """Persist FAISS index and chunks to disk."""
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str) -> bool:
        """Load FAISS index and chunks from disk."""
        index_path = os.path.join(path, "faiss.index")
        chunks_path = os.path.join(path, "chunks.pkl")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            return True
        return False

    def clear(self) -> None:
        self.index = None
        self.chunks = []
