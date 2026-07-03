import uuid
import numpy as np

class CachedSegment:
    """
    In-memory cache representing a loaded segment.
    Holds contiguous vector arrays and raw JSON payloads.
    """
    def __init__(
        self,
        segment_id: uuid.UUID,
        vectors: np.ndarray,
        record_ids: list[uuid.UUID],
        payloads: list[dict],
        deleted_offsets: set[int]
    ):
        self.segment_id = segment_id
        self.vectors = vectors # 2D NumPy array of shape (N, dimension)
        self.record_ids = record_ids
        self.payloads = payloads
        self.deleted_offsets = deleted_offsets
        
        # Calculate memory footprint in bytes
        vector_bytes = self.vectors.nbytes
        # rough string byte length estimation for payloads
        payload_bytes = sum(len(str(p)) for p in payloads)
        self.memory_bytes = vector_bytes + payload_bytes

    def get_vector(self, vector_idx: int) -> np.ndarray:
        """Retrieve vector at target index."""
        return self.vectors[vector_idx]

    def get_payload(self, vector_idx: int) -> dict:
        """Retrieve payload dictionary at target index."""
        return self.payloads[vector_idx]

    def is_deleted(self, vector_idx: int) -> bool:
        """Check if vector index corresponds to a deleted tombstone."""
        return vector_idx in self.deleted_offsets

    @property
    def record_count(self) -> int:
        return len(self.record_ids)
