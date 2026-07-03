import os
import uuid
import logging
import numpy as np
from typing import Tuple
from storage.cache.cached_segment import CachedSegment
from storage.segments.writer import BinarySegmentSerializer
from core.config import settings

logger = logging.getLogger(__name__)

class SegmentCacheManager:
    """
    Manages in-memory cache of binary segment files using an LRU eviction policy.
    Prevents database queries and disk reads on hot search pathways.
    """
    def __init__(self, max_memory_bytes: int = 512 * 1024 * 1024):
        self.max_memory_bytes = max_memory_bytes
        self.current_memory_bytes = 0
        self.cache: dict[uuid.UUID, CachedSegment] = {}
        self.lru_order: list[uuid.UUID] = []
        
        # Diagnostics
        self.hit_count = 0
        self.miss_count = 0

    def get_vector(self, segment_id: uuid.UUID, vector_idx: int, dimension: int) -> np.ndarray:
        """
        Retrieve a vector from the segment cache.
        If cache miss, lazily loads segment from disk.
        """
        cached_seg = self._get_or_load_segment(segment_id, dimension)
        if vector_idx >= cached_seg.record_count:
            raise IndexError(f"Vector index {vector_idx} out of range for segment {segment_id}")
        return cached_seg.get_vector(vector_idx)

    def get_payload(self, segment_id: uuid.UUID, vector_idx: int, dimension: int) -> dict:
        """
        Retrieve payload data from the segment cache.
        """
        cached_seg = self._get_or_load_segment(segment_id, dimension)
        if vector_idx >= cached_seg.record_count:
            raise IndexError(f"Vector index {vector_idx} out of range for segment {segment_id}")
        return cached_seg.get_payload(vector_idx)

    def is_deleted(self, segment_id: uuid.UUID, vector_idx: int, dimension: int) -> bool:
        """
        Verify if vector index corresponds to a deleted tombstone.
        """
        cached_seg = self._get_or_load_segment(segment_id, dimension)
        return cached_seg.is_deleted(vector_idx)

    def clear(self):
        """Reset the cache manager state."""
        self.cache.clear()
        self.lru_order.clear()
        self.current_memory_bytes = 0
        self.hit_count = 0
        self.miss_count = 0

    def _get_or_load_segment(self, segment_id: uuid.UUID, dimension: int) -> CachedSegment:
        # Cache Hit
        if segment_id in self.cache:
            # Update LRU ordering (move to end)
            self.lru_order.remove(segment_id)
            self.lru_order.append(segment_id)
            self.hit_count += 1
            return self.cache[segment_id]

        # Cache Miss
        self.miss_count += 1
        segment_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{segment_id}.bin")
        if not os.path.exists(segment_path):
            raise FileNotFoundError(f"Segment file {segment_path} not found on disk.")

        logger.info(f"Cache miss. Lazily loading segment {segment_id} from disk.")
        with open(segment_path, "rb") as f:
            data = f.read()

        # Unpack binary data
        _, records = BinarySegmentSerializer.deserialize(data, dimension)
        
        # Construct raw contiguous vector arrays
        vectors_list = [r['vector'] for r in records]
        vectors_matrix = np.array(vectors_list, dtype=np.float32)
        
        record_ids = [r['id'] for r in records]
        payloads = [r['payload'] for r in records]
        deleted_offsets = {i for i, r in enumerate(records) if r.get('status') == 2}

        cached_seg = CachedSegment(
            segment_id=segment_id,
            vectors=vectors_matrix,
            record_ids=record_ids,
            payloads=payloads,
            deleted_offsets=deleted_offsets
        )

        # Enforce memory limits before inserting
        self._ensure_capacity(cached_seg.memory_bytes)

        # Insert to cache
        self.cache[segment_id] = cached_seg
        self.lru_order.append(segment_id)
        self.current_memory_bytes += cached_seg.memory_bytes

        return cached_seg

    def _ensure_capacity(self, incoming_bytes: int):
        """Evict oldest segment entries if incoming segment exceeds cache limits."""
        while (self.current_memory_bytes + incoming_bytes > self.max_memory_bytes) and self.lru_order:
            evict_id = self.lru_order.pop(0)
            evicted_seg = self.cache.pop(evict_id)
            self.current_memory_bytes -= evicted_seg.memory_bytes
            logger.info(f"LRU Cache evicted segment {evict_id} (Freed {evicted_seg.memory_bytes} bytes).")

    @property
    def hit_ratio(self) -> float:
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return float(self.hit_count / total)
