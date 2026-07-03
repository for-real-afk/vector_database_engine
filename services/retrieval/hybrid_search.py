import os
import uuid
import logging
import numpy as np
from sqlalchemy.orm import Session

from models.database_models import Collection, Embedding, Chunk, Document
from services.retrieval.filter_resolver import MetadataFilterResolver
from index.serialization.persistence import HNSWIndexManager
from storage.segments.writer import BinarySegmentSerializer
from storage.cache.segment_cache import SegmentCacheManager
from core.config import settings

logger = logging.getLogger(__name__)

class HybridSearchCoordinator:
    """
    Coordinates hybrid search retrieval.
    Merges in-graph HNSW semantic results with database keyword match results
    while strictly enforcing resolved metadata pre-filters.
    """
    def __init__(self, db: Session, embedding_provider, cache_manager=None):
        self.db = db
        self.embedding_provider = embedding_provider
        self.cache_manager = cache_manager or SegmentCacheManager()

    def search(
        self,
        collection_id: uuid.UUID,
        query_text: str,
        query_vector: list[float] = None,
        filters: dict = None,
        k: int = 5,
        alpha: float = 0.5
    ) -> list[dict]:
        """
        Execute hybrid search.
        Score = alpha * SemanticScore + (1 - alpha) * KeywordScore
        """
        # 1. Fetch collection details
        collection = self.db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            raise ValueError(f"Collection {collection_id} not found.")

        # Resolve query vector
        if query_vector is None:
            query_vector = self.embedding_provider.embed_text(query_text)

        if len(query_vector) != collection.dimension:
            raise ValueError("Query vector dimension mismatch.")

        # 2. Resolve metadata pre-filters to allowed embedding IDs
        allowed_ids = None
        if filters:
            resolver = MetadataFilterResolver(self.db)
            allowed_ids = resolver.resolve_filters(collection_id, filters)
            if not allowed_ids:
                # No records match metadata filters, return empty results early
                logger.info("Metadata filtering returned empty allowed set. Halting search.")
                return []

        # 3. Retrieve all segments linked to the collection
        segment_groups = self.db.query(Embedding.segment_id)\
            .join(Chunk)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)\
            .filter(Embedding.segment_id != None)\
            .distinct()\
            .all()
        segment_ids = [sg[0] for sg in segment_groups]

        semantic_candidates = {}

        # Cache-backed vector resolver for HNSW graph restoration
        def cache_resolver(n_uuid, s_uuid, v_idx):
            return self.cache_manager.get_vector(s_uuid, v_idx, collection.dimension).tolist()

        # 4. Search segments semantically
        for seg_id in segment_ids:
            snapshot_dir = os.path.join(settings.STORAGE_ROOT, "snapshots", str(seg_id))
            
            # Scenario A: Segment is sealed and has HNSW snapshot
            if os.path.exists(os.path.join(snapshot_dir, "graph.bin")):
                try:
                    hnsw_idx, _ = HNSWIndexManager.restore(snapshot_dir, cache_resolver)
                    results = hnsw_idx.search(query_vector, k=k*2, ef=64, allowed_ids=allowed_ids)
                    for score, node_id in results:
                        # Convert L2/Manhattan distances to similarity-like scores
                        # Score: Cosine distance -> Cosine similarity
                        if collection.metric.upper() == "COSINE":
                            sim = 1.0 - score # since search returns distance
                        elif collection.metric.upper() in ("L2", "EUCLIDEAN", "MANHATTAN"):
                            sim = 1.0 / (1.0 + score) # inverse normalization
                        else:
                            sim = score
                        semantic_candidates[node_id] = sim
                except Exception as e:
                    logger.error(f"HNSW restore search failed for segment {seg_id}: {e}. Falling back to exact scan.")
                    self._exact_scan_segment(seg_id, query_vector, collection, allowed_ids, semantic_candidates)
            
            # Scenario B: Growing segment, fallback to exact scan
            else:
                self._exact_scan_segment(seg_id, query_vector, collection, allowed_ids, semantic_candidates)

        if not semantic_candidates:
            return []

        # 5. Retrieve text and calculate keyword match scores
        # Query matching chunks from DB
        chunk_query = self.db.query(Chunk.id, Chunk.text_content, Embedding.id)\
            .join(Embedding)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)
            
        if allowed_ids is not None:
            chunk_query = chunk_query.filter(Embedding.id.in_(list(allowed_ids)))
            
        chunks = chunk_query.all()
        
        # Calculate TF-IDF-like keyword overlap score
        query_words = [w.lower() for w in query_text.split() if len(w) > 2]
        keyword_scores = {}
        
        for chunk_id, text, emb_id in chunks:
            if not query_words:
                keyword_scores[emb_id] = 0.0
                continue
                
            text_lower = text.lower()
            matches = sum(1 for w in query_words if w in text_lower)
            # Normalised keyword overlap score
            keyword_scores[emb_id] = float(matches / len(query_words))

        # 6. Normalize and combine scores
        hybrid_results = []
        
        # Hydrate candidate info
        for emb_id, sem_score in semantic_candidates.items():
            key_score = keyword_scores.get(emb_id, 0.0)
            combined_score = (alpha * sem_score) + ((1.0 - alpha) * key_score)
            
            # Retrieve chunk details
            emb_rec = self.db.query(Embedding).filter(Embedding.id == emb_id).first()
            if not emb_rec:
                continue
                
            chunk_rec = emb_rec.chunk
            doc_rec = chunk_rec.document
            
            hybrid_results.append({
                'id': emb_id,
                'score': combined_score,
                'payload': {
                    'chunk_id': str(chunk_rec.id),
                    'document_id': str(doc_rec.id),
                    'title': doc_rec.title,
                    'text': chunk_rec.text_content
                }
            })

        # Sort descending
        hybrid_results = sorted(hybrid_results, key=lambda x: x['score'], reverse=True)
        return hybrid_results[:k]

    def _exact_scan_segment(self, seg_id, query_vector, collection, allowed_ids, candidates):
        """Fallback exact scan for segments without graphs."""
        try:
            segment_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{seg_id}.bin")
            if not os.path.exists(segment_path):
                return
            with open(segment_path, "rb") as f:
                data = f.read()
            _, records = BinarySegmentSerializer.deserialize(data, collection.dimension)
            
            q_vec = np.array(query_vector, dtype=np.float32)
            for r in records:
                if r.get('status') != 1:
                    continue
                if allowed_ids is not None and r['id'] not in allowed_ids:
                    continue
                    
                r_vec = np.array(r['vector'], dtype=np.float32)
                
                # Distance calculations
                norm_q = np.linalg.norm(q_vec)
                norm_r = np.linalg.norm(r_vec)
                if collection.metric.upper() == "COSINE":
                    sim = float(np.dot(q_vec, r_vec) / (norm_q * norm_r)) if norm_q != 0 and norm_r != 0 else 0.0
                elif collection.metric.upper() in ("L2", "EUCLIDEAN", "MANHATTAN"):
                    dist = float(np.linalg.norm(q_vec - r_vec))
                    sim = 1.0 / (1.0 + dist)
                else:
                    sim = float(np.dot(q_vec, r_vec))
                    
                candidates[r['id']] = sim
        except Exception as e:
            logger.error(f"Fallback exact scan failed for segment {seg_id}: {e}")
