import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from app.services.embeddings import embedding_service
from app.utils.logging import logger

STORAGE_FILE = Path("./data/vector_store.json")


class HybridSearchEngine:
    """
    Hybrid Search Engine combining:
    1. Dense Vector Embeddings (Ollama nomic-embed-text)
    2. BM25 Keyword Search
    3. Reciprocal Rank Fusion (RRF)
    """

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: List[np.ndarray] = []
        self.bm25_corpus: List[List[str]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self._load_persisted_store()

    def _load_persisted_store(self):
        if STORAGE_FILE.exists():
            try:
                data = json.loads(STORAGE_FILE.read_text(encoding="utf-8"))
                self.chunks = data.get("chunks", [])
                raw_vectors = data.get("vectors", [])
                self.vectors = [np.array(v, dtype=np.float32) for v in raw_vectors]
                if self.chunks:
                    self._rebuild_bm25()
                logger.info(f"Loaded {len(self.chunks)} chunks into Hybrid Vector Store.")
            except Exception as e:
                logger.warning(f"Failed to load persisted vector store: {e}")

    def _save_persisted_store(self):
        STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self.chunks,
            "vectors": [v.tolist() for v in self.vectors]
        }
        STORAGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _rebuild_bm25(self):
        self.bm25_corpus = [c["content"].lower().split() for c in self.chunks]
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25_index = None

    async def add_documents(self, chunks: List[Dict[str, Any]]):
        """Embed and index new chunks into the hybrid store."""
        for chunk in chunks:
            text = chunk["content"]
            try:
                emb = await embedding_service.get_embedding(text)
                vec = np.array(emb, dtype=np.float32)
                # Normalize vector for fast cosine similarity
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self.vectors.append(vec)
            except Exception as e:
                logger.warning(f"Embedding failed, using random normalized vector: {e}")
                rnd = np.random.randn(768).astype(np.float32)
                self.vectors.append(rnd / np.linalg.norm(rnd))

            self.chunks.append(chunk)

        self._rebuild_bm25()
        self._save_persisted_store()
        logger.info(f"Successfully indexed {len(chunks)} chunks in Hybrid RAG store.")

    def delete_document(self, filename: str) -> int:
        """Remove all chunks and vectors associated with a filename."""
        initial_count = len(self.chunks)
        keep_indices = [i for i, c in enumerate(self.chunks) if c.get("filename") != filename]
        
        self.chunks = [self.chunks[i] for i in keep_indices]
        self.vectors = [self.vectors[i] for i in keep_indices]
        
        self._rebuild_bm25()
        self._save_persisted_store()
        
        deleted_count = initial_count - len(self.chunks)
        logger.info(f"Deleted {deleted_count} chunks for document '{filename}'.")
        return deleted_count

    def get_document_chunk_count(self, filename: str) -> int:
        return sum(1 for c in self.chunks if c.get("filename") == filename)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        min_cosine_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Run Hybrid Vector Search + BM25 + Reciprocal Rank Fusion.
        Applies a strict cosine similarity relevance filter so irrelevant queries
        do not match unrelated documents.
        """
        if not self.chunks:
            return []

        # 1. Vector Search
        vector_scores = []
        max_cosine = 0.0
        try:
            q_emb = await embedding_service.get_embedding(query)
            q_vec = np.array(q_emb, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

            for idx, doc_vec in enumerate(self.vectors):
                sim = float(np.dot(q_vec, doc_vec))
                if sim > max_cosine:
                    max_cosine = sim
                vector_scores.append((idx, sim))
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            for idx in range(len(self.chunks)):
                vector_scores.append((idx, 0.0))

        vector_ranked = sorted(vector_scores, key=lambda x: x[1], reverse=True)

        # 2. BM25 Search with Stopword Filtering
        STOP_WORDS = {
            "a", "an", "the", "and", "or", "of", "in", "for", "on", "with",
            "at", "by", "to", "is", "are", "was", "were", "what", "which",
            "how", "who", "whom", "this", "that", "it", "from", "as", "be"
        }
        raw_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower()).split()
        tokenized_q = [w for w in raw_tokens if w not in STOP_WORDS and len(w) > 2]
        bm25_ranked = []
        max_bm25 = 0.0
        if self.bm25_index and tokenized_q:
            scores = self.bm25_index.get_scores(tokenized_q)
            max_bm25 = max(scores) if len(scores) > 0 else 0.0
            bm25_ranked = sorted([(idx, score) for idx, score in enumerate(scores)], key=lambda x: x[1], reverse=True)
        else:
            bm25_ranked = []

        # If cosine similarity is low and no meaningful BM25 keyword matches, reject
        if max_cosine < min_cosine_threshold and max_bm25 <= 0.0:
            logger.info(
                f"Hybrid search rejected candidates: max_cosine={max_cosine:.3f} < {min_cosine_threshold}, "
                f"max_bm25={max_bm25:.2f}. No relevant documents found."
            )
            return []

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, (idx, sim) in enumerate(vector_ranked):
            # Only fuse if vector similarity meets strict threshold
            if sim >= min_cosine_threshold:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, (idx, score) in enumerate(bm25_ranked):
            if score > 0.0:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        if not rrf_scores:
            return []

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        # Map index to vector similarity for quick lookup
        vec_sim_map = dict(vector_scores)

        for idx, score in sorted_indices:
            chunk = dict(self.chunks[idx])
            chunk["rrf_score"] = round(float(score * 50), 3)
            chunk["cosine_similarity"] = round(float(vec_sim_map.get(idx, 0.0)), 4)
            chunk["score"] = chunk["cosine_similarity"]
            results.append(chunk)

        return results


hybrid_search = HybridSearchEngine()
