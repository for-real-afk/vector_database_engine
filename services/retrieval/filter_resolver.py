import uuid
import logging
from typing import Set
from sqlalchemy.orm import Session
from models.database_models import Metadata, Chunk, Document, Embedding

logger = logging.getLogger(__name__)

class MetadataFilterResolver:
    """
    Resolves metadata filter queries against the database schema.
    Retrieves the set of embedding UUIDs matching all constraints.
    """
    def __init__(self, db: Session):
        self.db = db

    def resolve_filters(self, collection_id: uuid.UUID, filters: dict) -> Set[uuid.UUID]:
        """
        Resolve filters (e.g. {"category": "database", "priority": 1}) into a set of Embedding UUIDs.
        """
        if not filters:
            return set()

        # Step 1: Resolve matching chunk IDs for each key-value filter
        matching_chunks_per_filter = []

        for key, val in filters.items():
            # Query metadata records for this key inside the target collection
            records = self.db.query(Metadata.chunk_id, Metadata.value)\
                .join(Chunk)\
                .join(Document)\
                .filter(Document.collection_id == collection_id)\
                .filter(Metadata.key == key)\
                .all()

            chunk_matches = set()
            for chunk_id, value_data in records:
                # Compare in-memory (JSONBType decorator returns raw Python values)
                if value_data == val:
                    chunk_matches.add(chunk_id)
            
            matching_chunks_per_filter.append(chunk_matches)

        if not matching_chunks_per_filter:
            return set()

        # Intersect chunk IDs across all filters to enforce AND logic
        final_chunk_ids = matching_chunks_per_filter[0]
        for s in matching_chunks_per_filter[1:]:
            final_chunk_ids = final_chunk_ids.intersection(s)

        if not final_chunk_ids:
            return set()

        # Step 2: Map matching chunk IDs to embedding IDs (HNSW Node IDs)
        embeddings = self.db.query(Embedding.id)\
            .filter(Embedding.chunk_id.in_(list(final_chunk_ids)))\
            .all()

        allowed_ids = {e[0] for e in embeddings}
        return allowed_ids
