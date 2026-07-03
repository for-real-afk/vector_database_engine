import os
import json
import struct
import uuid
import logging
from typing import Optional, Set
from sqlalchemy import func
from sqlalchemy.orm import Session

from models.database_models import Embedding, Chunk, Document, Collection
from core.config import settings

logger = logging.getLogger(__name__)

class CollectionStatistics:
    """
    Data container holding all gathered statistics for a collection.
    Provides serializable metadata used by the query planner cost models.
    """
    def __init__(self):
        self.collection_id: Optional[uuid.UUID] = None
        self.collection_size = 0
        self.growing_segments = 0
        self.sealed_segments = 0
        self.deleted_vectors = 0
        self.dimension = 0
        
        # Graph-level statistics
        self.graph_nodes = 0
        self.graph_layers = 0
        self.average_degree = 16.0  # default M
        self.max_degree = 32.0      # default M0
        
        # Cache-level statistics
        self.cache_hit_ratio = 0.0
        self.cache_miss_ratio = 0.0
        self.cache_memory_bytes = 0
        
        # Historic heuristics
        self.average_query_latency_ms = 5.0
        
    def to_dict(self) -> dict:
        """Serialize statistics properties to a dictionary."""
        return {
            "collection_id": str(self.collection_id) if self.collection_id else None,
            "collection_size": self.collection_size,
            "growing_segments": self.growing_segments,
            "sealed_segments": self.sealed_segments,
            "deleted_vectors": self.deleted_vectors,
            "dimension": self.dimension,
            "graph_nodes": self.graph_nodes,
            "graph_layers": self.graph_layers,
            "average_degree": self.average_degree,
            "max_degree": self.max_degree,
            "cache_hit_ratio": self.cache_hit_ratio,
            "cache_miss_ratio": self.cache_miss_ratio,
            "cache_memory_bytes": self.cache_memory_bytes,
            "average_query_latency_ms": self.average_query_latency_ms
        }


class StatisticsCatalog:
    """
    Subsystem responsible for collecting collection, cache, and HNSW graph statistics.
    Used by the Query Optimizer to estimate execution cost plans.
    """
    def __init__(self, db: Session, cache_manager=None):
        self.db = db
        self.cache_manager = cache_manager

    def get_statistics(self, collection_id: uuid.UUID) -> CollectionStatistics:
        """
        Gather metrics for the target collection, inspecting database, HNSW graph headers, 
        and segment caches.
        """
        stats = CollectionStatistics()
        stats.collection_id = collection_id

        # 1. Fetch collection metadata
        collection = self.db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            raise ValueError(f"Collection {collection_id} not found.")
        stats.dimension = collection.dimension

        # 2. Count total active embeddings
        stats.collection_size = self.db.query(Embedding.id)\
            .join(Chunk)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)\
            .count()

        # 3. Retrieve all distinct segment IDs for the collection
        segment_groups = self.db.query(Embedding.segment_id)\
            .join(Chunk)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)\
            .filter(Embedding.segment_id != None)\
            .distinct()\
            .all()
        segment_ids = [sg[0] for sg in segment_groups]

        # 4. Scan segment states and inspect graph binary files on disk
        for seg_id in segment_ids:
            snapshot_dir = os.path.join(settings.STORAGE_ROOT, "snapshots", str(seg_id))
            graph_path = os.path.join(snapshot_dir, "graph.bin")
            meta_path = os.path.join(snapshot_dir, "metadata.json")

            if os.path.exists(graph_path):
                stats.sealed_segments += 1
                # Read binary VHNS header to fetch node counts and layers count
                try:
                    with open(graph_path, "rb") as f:
                        header_bytes = f.read(64)
                    if len(header_bytes) == 64:
                        magic, version, max_level, _, node_count = struct.unpack(
                            "<4s H H 16s I 36x",
                            header_bytes
                        )
                        if magic == b'VHNS':
                            stats.graph_nodes += node_count
                            stats.graph_layers = max(stats.graph_layers, int(max_level) + 1)
                except Exception as e:
                    logger.warning(f"Failed to read graph header for segment {seg_id}: {e}")
                
                # Fetch tombstones / deleted count from metadata
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        stats.deleted_vectors += len(meta.get("tombstones", []))
                        stats.average_degree = float(meta.get("M", 16))
                        stats.max_degree = float(meta.get("M0", 32))
                    except Exception as e:
                        logger.warning(f"Failed to read metadata snapshot for segment {seg_id}: {e}")
            else:
                stats.growing_segments += 1

        # 5. Extract cache metrics
        if self.cache_manager:
            stats.cache_memory_bytes = self.cache_manager.current_memory_bytes
            total_cache_requests = self.cache_manager.hit_count + self.cache_manager.miss_count
            if total_cache_requests > 0:
                stats.cache_hit_ratio = float(self.cache_manager.hit_count / total_cache_requests)
                stats.cache_miss_ratio = 1.0 - stats.cache_hit_ratio

        return stats
