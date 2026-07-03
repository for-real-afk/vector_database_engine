import os
import uuid
import logging
import numpy as np
from sqlalchemy.orm import Session

from models.database_models import Collection, Embedding, Chunk, Document
from storage.segments.writer import BinarySegmentSerializer
from index.distance.metrics import DistanceMetric
from core.config import settings

logger = logging.getLogger(__name__)

class ExactRetriever:
    """
    Retrieval Engine Version 1 (Exact Search).
    Scans all binary segments on disk, computes exact vector similarity 
    via NumPy, sorts candidate listings, and returns the Top-K elements.
    """
    def __init__(self, db: Session):
        self.db = db

    def search(self, collection_id: uuid.UUID, query_vector: list[float], k: int = 5) -> list[dict]:
        """
        Execute O(N) brute force search over all segments in a collection.
        Returns a sorted list of candidate dicts: {'id': UUID, 'score': float, 'payload': dict}
        """
        # 1. Fetch collection details to verify dimensions and metric configuration
        collection = self.db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found.")

        if len(query_vector) != collection.dimension:
            raise ValueError(
                f"Query vector dimension ({len(query_vector)}) does not match "
                f"collection dimension ({collection.dimension})."
            )

        # 2. Query DB to identify all segment files linked to this collection
        db_segments = self.db.query(Embedding.segment_id)\
            .join(Chunk)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)\
            .filter(Embedding.segment_id != None)\
            .distinct()\
            .all()

        segment_ids = [s[0] for s in db_segments]
        if not segment_ids:
            logger.info("No storage segments found for collection search.")
            return []

        candidates = []
        q_vec = np.array(query_vector, dtype=np.float32)

        # 3. Read segment files and compute distances
        for seg_id in segment_ids:
            segment_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{seg_id}.bin")
            if not os.path.exists(segment_path):
                logger.warning(f"Segment file {seg_id}.bin not found on disk. Skipping.")
                continue

            try:
                with open(segment_path, "rb") as f:
                    segment_bytes = f.read()
                
                # Deserialization pulls the full vector array block
                _, records = BinarySegmentSerializer.deserialize(segment_bytes, collection.dimension)
                
                for record in records:
                    # Filter out deleted records (Tombstones)
                    if record.get('status') != 1:
                        continue

                    r_vec = np.array(record['vector'], dtype=np.float32)
                    
                    # Compute distance based on collection metric configuration
                    score = self._compute_distance(q_vec, r_vec, collection.metric)
                    
                    candidates.append({
                        'id': record['id'],
                        'score': score,
                        'payload': record.get('payload', {})
                    })
            except Exception as e:
                logger.error(f"Failed to read segment {seg_id} during retrieval: {e}")
                continue

        # 4. Sort results depending on the metric rule
        sorted_candidates = self._sort_candidates(candidates, collection.metric)

        # 5. Return Top-K
        return sorted_candidates[:k]

    def _compute_distance(self, q_vec: np.ndarray, r_vec: np.ndarray, metric: str) -> float:
        """Helper to calculate distance matching the metric name."""
        m_name = metric.upper()
        if m_name == "COSINE":
            # For cosine, return Cosine Similarity (closer to 1.0 is better)
            return DistanceMetric.cosine_similarity(q_vec, r_vec)
        elif m_name in ("L2", "EUCLIDEAN"):
            # For L2, return Euclidean distance (closer to 0.0 is better)
            return DistanceMetric.l2_distance(q_vec, r_vec)
        elif m_name == "DOTPRODUCT":
            # For Dot Product, larger dot product is better
            return DistanceMetric.dot_product(q_vec, r_vec)
        elif m_name == "MANHATTAN":
            # For Manhattan, lower score is better
            return DistanceMetric.manhattan_distance(q_vec, r_vec)
        else:
            # Fallback to Cosine Similarity
            return DistanceMetric.cosine_similarity(q_vec, r_vec)

    def _sort_candidates(self, candidates: list[dict], metric: str) -> list[dict]:
        """
        Sort candidates:
        - For similarity-based (Cosine, DotProduct): Descending order (higher is better).
        - For distance-based (L2, Manhattan): Ascending order (lower is better).
        """
        m_name = metric.upper()
        if m_name in ("COSINE", "DOTPRODUCT"):
            return sorted(candidates, key=lambda x: x['score'], reverse=True)
        else:
            # L2, Manhattan: Ascending order
            return sorted(candidates, key=lambda x: x['score'], reverse=False)
