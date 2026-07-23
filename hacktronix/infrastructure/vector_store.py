"""
FAISS Vector Store Implementation for Semantic Search.

Implements `IVectorStore` interface using FAISS and SentenceTransformers (or fallback embedding engine).
Enables rapid semantic slicing of the World Model for the Query Layer.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np

from hacktronix.domain.interfaces import IVectorStore
from hacktronix.domain.entities import Entity


class FAISSVectorStore(IVectorStore):
    """
    FAISS-backed vector store for semantic entity indexing and top-k retrieval.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self.entity_ids: List[str] = []
        self.entity_texts: List[str] = []
        self.index = None
        self.model = None

        self._init_model_and_index()

    def _init_model_and_index(self) -> None:
        """Loads sentence-transformers model and initializes FAISS index."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            self.model = None  # Fallback to simple TF-IDF/character vector encoder

        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        except Exception:
            self.index = None  # Fallback to numpy Cosine similarity

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text string."""
        if self.model is not None:
            emb = self.model.encode([text], convert_to_numpy=True)[0]
            return emb.astype("float32")
        else:
            # Fallback deterministic pseudo-embedding (hashed bag-of-words)
            vec = np.zeros(self.embedding_dim, dtype="float32")
            words = text.lower().split()
            for w in words:
                idx = hash(w) % self.embedding_dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec

    def index_entity(self, entity: Entity) -> None:
        """Indexes or re-indexes an entity by generating semantic text string."""
        states_str = " ".join([f"{k} is {v.value}" for k, v in entity.states.items()])
        text_representation = f"{entity.name} category: {entity.category.value} room: {entity.room_id or ''} states: {states_str}"

        if entity.id in self.entity_ids:
            idx = self.entity_ids.index(entity.id)
            self.entity_texts[idx] = text_representation
            # Rebuild index for simplicity and consistency
            self._rebuild_index()
            return

        emb = self._get_embedding(text_representation)
        self.entity_ids.append(entity.id)
        self.entity_texts.append(text_representation)

        if self.index is not None:
            import faiss
            self.index.add(np.expand_dims(emb, axis=0))

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from stored entity embeddings."""
        if not self.entity_texts:
            if self.index is not None:
                self.index.reset()
            return

        embeddings = [self._get_embedding(t) for t in self.entity_texts]
        matrix = np.array(embeddings, dtype="float32")

        if self.index is not None:
            self.index.reset()
            self.index.add(matrix)

    def search_relevant_entities(self, query: str, top_k: int = 5) -> List[str]:
        """
        Performs vector similarity search against query string.
        Returns ordered list of top_k entity IDs.
        """
        if not self.entity_ids:
            return []

        query_emb = self._get_embedding(query)
        top_k = min(top_k, len(self.entity_ids))

        if self.index is not None and self.index.ntotal > 0:
            distances, indices = self.index.search(np.expand_dims(query_emb, axis=0), top_k)
            matched_ids = []
            for i in indices[0]:
                if 0 <= i < len(self.entity_ids):
                    matched_ids.append(self.entity_ids[i])
            return matched_ids
        else:
            # Fallback numpy inner product similarity
            embeddings = np.array([self._get_embedding(t) for t in self.entity_texts], dtype="float32")
            scores = np.dot(embeddings, query_emb)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [self.entity_ids[i] for i in top_indices]

    def clear(self) -> None:
        """Reset index and entity mappings."""
        self.entity_ids.clear()
        self.entity_texts.clear()
        if self.index is not None:
            self.index.reset()
