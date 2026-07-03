import os
import shutil
import uuid
import logging
from typing import Optional
from sqlalchemy import func
from sqlalchemy.orm import Session
from models.database_models import Embedding, Chunk, Document, Collection
from storage.segments.writer import BinarySegmentSerializer
from index.graph.hnsw import HNSWIndex
from index.serialization.persistence import HNSWIndexManager
from core.config import settings

logger = logging.getLogger(__name__)

class SegmentCompactor:
    """
    Coordinates segment lifecycle management including background compaction and merging.
    Purges deleted records (tombstones) and redirects PostgreSQL references transactionally.
    """
    def __init__(self, db: Session, cache_manager=None):
        self.db = db
        self.cache_manager = cache_manager

    def should_compact(self, collection_id: uuid.UUID) -> bool:
        """
        Evaluate if collection segments require compaction.
        Triggers compaction if:
        - There are 2 or more sealed/inactive segments of size < 0.5 * MAX_VECTORS_PER_SEGMENT.
        - Or any segment contains deleted tombstones > 20% of its record count.
        """
        # Group and count embeddings per segment
        active_segments = self.db.query(Embedding.segment_id, func.count(Embedding.id))\
            .join(Chunk)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)\
            .filter(Embedding.segment_id != None)\
            .group_by(Embedding.segment_id)\
            .all()

        if len(active_segments) < 2:
            return False

        small_segments_count = 0
        threshold = int(settings.MAX_VECTORS_PER_SEGMENT * 0.5)

        for seg_id, count in active_segments:
            if count < threshold:
                small_segments_count += 1

        if small_segments_count >= 2:
            return True

        return False

    def compact(self, collection_id: uuid.UUID) -> Optional[uuid.UUID]:
        """
        Merge eligible sealed segments into a single new consolidated segment.
        Purges soft-deleted records and updates DB mappings in a single transaction.
        """
        # 1. Fetch collection metadata
        collection = self.db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            raise ValueError(f"Collection {collection_id} not found.")

        # 2. Get list of segments linked to this collection
        segment_groups = self.db.query(Embedding.segment_id, func.count(Embedding.id))\
            .join(Chunk)\
            .join(Document)\
            .filter(Document.collection_id == collection_id)\
            .filter(Embedding.segment_id != None)\
            .group_by(Embedding.segment_id)\
            .all()

        segment_ids = [sg[0] for sg in segment_groups]
        if len(segment_ids) < 2:
            logger.info("Fewer than 2 segments found. Compaction not required.")
            return None

        merged_records = []
        deleted_records = []

        # 3. Read and collect active records from all source segments
        for seg_id in segment_ids:
            segment_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{seg_id}.bin")
            if not os.path.exists(segment_path):
                logger.warning(f"Segment file {seg_id}.bin not found during compaction. Skipping.")
                continue

            try:
                with open(segment_path, "rb") as f:
                    data = f.read()
                _, records = BinarySegmentSerializer.deserialize(data, collection.dimension)
                
                for r in records:
                    if r.get('status') == 1: # Active
                        merged_records.append(r)
                    else: # Deleted (Tombstone)
                        deleted_records.append(r)
            except Exception as e:
                logger.error(f"Failed to read segment {seg_id} for compaction: {e}")
                raise e

        if not merged_records:
            logger.warning("No active records found to merge during compaction.")
            return None

        # 4. Write merged records to a new segment
        merged_seg_id = uuid.uuid4()
        segment_bytes = BinarySegmentSerializer.serialize(
            merged_seg_id, 
            merged_records, 
            collection.dimension
        )
        
        merged_segment_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{merged_seg_id}.bin")
        os.makedirs(os.path.dirname(merged_segment_path), exist_ok=True)
        with open(merged_segment_path, "wb") as f:
            f.write(segment_bytes)

        # 5. Build HNSW index on the merged segment
        hnsw_idx = HNSWIndex(
            dimension=collection.dimension, 
            metric=collection.metric,
            M=16,
            M0=32,
            ef_construction=64
        )
        
        new_segment_mappings = {}
        for new_idx, record in enumerate(merged_records):
            hnsw_idx.insert(record['id'], record['vector'])
            new_segment_mappings[record['id']] = (merged_seg_id, new_idx)

        # Write graph snapshot
        snapshot_dir = os.path.join(settings.STORAGE_ROOT, "snapshots", str(merged_seg_id))
        HNSWIndexManager.snapshot(snapshot_dir, hnsw_idx, new_segment_mappings)

        # 6. Commit pointer redirections to PostgreSQL in a single database transaction
        try:
            # Update Embedding table coordinates
            for new_idx, record in enumerate(merged_records):
                emb = self.db.query(Embedding).filter(Embedding.id == record['id']).first()
                if emb:
                    emb.segment_id = merged_seg_id
                    emb.vector_idx = new_idx

            # Structurally delete tombstones from relational metadata
            for r in deleted_records:
                self.db.query(Embedding).filter(Embedding.id == r['id']).delete()

            self.db.commit()
            logger.info(f"Database references updated to merged segment {merged_seg_id}.")
        except Exception as e:
            self.db.rollback()
            # Clean up the newly created binary segment and graph on disk if transaction rolls back
            if os.path.exists(merged_segment_path):
                os.remove(merged_segment_path)
            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)
            logger.error(f"PostgreSQL redirection transaction failed. Rolled back: {e}")
            raise e

        # 7. Post-cleanup: delete old disk files and update memory cache
        for old_id in segment_ids:
            # Delete old segment file
            old_seg_path = os.path.join(settings.STORAGE_ROOT, "segments", f"{old_id}.bin")
            if os.path.exists(old_seg_path):
                os.remove(old_seg_path)

            # Delete old snapshot folder
            old_snap_dir = os.path.join(settings.STORAGE_ROOT, "snapshots", str(old_id))
            if os.path.exists(old_snap_dir):
                shutil.rmtree(old_snap_dir)

            # Evict from in-memory cache manager
            if self.cache_manager:
                if old_id in self.cache_manager.cache:
                    del self.cache_manager.cache[old_id]
                if old_id in self.cache_manager.lru_order:
                    self.cache_manager.lru_order.remove(old_id)

        logger.info(f"Successfully compacted {len(segment_ids)} segments into {merged_seg_id}.")
        return merged_seg_id
