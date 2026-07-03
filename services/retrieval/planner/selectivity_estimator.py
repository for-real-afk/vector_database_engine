import uuid
import logging
from typing import Dict, Any, Optional, Set
from sqlalchemy.orm import Session

from models.database_models import Metadata, Chunk, Document

logger = logging.getLogger(__name__)

class MetadataSelectivityEstimator:
    """
    Subsystem responsible for cardinality and selectivity estimation of metadata filters.
    Computes matching chunk intersection ratios to inform optimizer cost models.
    """
    def __init__(self, db: Session):
        self.db = db

    def estimate_selectivity(self, collection_id: uuid.UUID, filters: Optional[Dict[str, Any]]) -> float:
        """
        Estimate the selectivity ratio (percentage of matching records) for filters.
        Returns a float between 0.001 and 1.0.
        """
        if not filters:
            return 1.0

        # Query total active chunk count in the collection
        total_chunks = self.db.query(Chunk.id)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)\
            .count()

        if total_chunks == 0:
            return 1.0

        matching_chunks_per_filter = []

        for key, val in filters.items():
            # Retrieve all metadata values for the key within this collection
            records = self.db.query(Metadata.chunk_id, Metadata.value)\
                .join(Chunk)\
                .join(Document)\
                .filter(Document.collection_id == collection_id)\
                .filter(Metadata.key == key)\
                .all()

            chunk_matches: Set[uuid.UUID] = set()
            for chunk_id, value_data in records:
                # Compare raw python types (handled by SQL Custom JSONB deserializer)
                if value_data == val:
                    chunk_matches.add(chunk_id)

            matching_chunks_per_filter.append(chunk_matches)

        if not matching_chunks_per_filter:
            return 1.0

        # Enforce AND intersection across filters
        final_chunk_ids = matching_chunks_per_filter[0]
        for s in matching_chunks_per_filter[1:]:
            final_chunk_ids = final_chunk_ids.intersection(s)

        selectivity = len(final_chunk_ids) / total_chunks
        
        # Apply a floor selectivity of 0.001 to prevent planning division-by-zero errors
        return max(0.001, selectivity)
