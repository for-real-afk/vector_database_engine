import os
import uuid
import tempfile
import pytest
import numpy as np
from sqlalchemy.orm import Session

from models.database_models import Collection, Embedding, Chunk, Document
from services.ingestion.pipeline import IngestionPipeline
from services.chunking.chunkers import FixedSizeChunker
from embeddings.providers import MockEmbeddingProvider
from storage.segments.writer import BinarySegmentSerializer
from storage.segments.compactor import SegmentCompactor
from storage.cache.segment_cache import SegmentCacheManager
from core.config import settings

def test_segment_sealing_on_capacity(db_session: Session):
    collection_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # Override capacity threshold to a low number for testing
        old_max = settings.MAX_VECTORS_PER_SEGMENT
        settings.MAX_VECTORS_PER_SEGMENT = 3
        
        try:
            # Create collection
            col = Collection(
                id=collection_id,
                name="sealing_col",
                namespace="default",
                dimension=16,
                metric="Cosine"
            )
            db_session.add(col)
            db_session.commit()
            
            provider = MockEmbeddingProvider(dimension=16)
            chunker = FixedSizeChunker(chunk_size=50)
            pipeline = IngestionPipeline(db_session, provider, chunker)
            
            # Ingest 3 separate single-chunk documents (fills segment 1)
            pipeline.ingest_document(collection_id, "doc1", "first text block")
            pipeline.ingest_document(collection_id, "doc2", "second text block")
            doc3 = pipeline.ingest_document(collection_id, "doc3", "third text block")
            
            # Retrieve segment ID of doc3
            emb3 = db_session.query(Embedding).join(Chunk).filter(Chunk.document_id == doc3.id).first()
            seg1_id = emb3.segment_id
            
            # Verify that the HNSW graph snapshot was automatically written to disk
            snapshot_dir = os.path.join(temp_dir, "snapshots", str(seg1_id))
            assert os.path.exists(os.path.join(snapshot_dir, "graph.bin"))
            assert os.path.exists(os.path.join(snapshot_dir, "metadata.json"))
            
            # Ingest a 4th document (should be placed in a new segment)
            doc4 = pipeline.ingest_document(collection_id, "doc4", "fourth text block")
            emb4 = db_session.query(Embedding).join(Chunk).filter(Chunk.document_id == doc4.id).first()
            seg2_id = emb4.segment_id
            
            assert seg1_id != seg2_id
            
            # Verify that the new segment 2 does NOT have a graph snapshot yet (still growing)
            snapshot_dir2 = os.path.join(temp_dir, "snapshots", str(seg2_id))
            assert not os.path.exists(snapshot_dir2)
            
        finally:
            settings.MAX_VECTORS_PER_SEGMENT = old_max

def test_compaction_and_tombstone_purge(db_session: Session):
    collection_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        col = Collection(
            id=collection_id,
            name="compact_col",
            namespace="default",
            dimension=16,
            metric="Cosine"
        )
        db_session.add(col)
        db_session.commit()
        
        provider = MockEmbeddingProvider(dimension=16)
        chunker = FixedSizeChunker(chunk_size=50)
        pipeline = IngestionPipeline(db_session, provider, chunker)
        
        # 1. Ingest 4 documents (MAX_VECTORS_PER_SEGMENT is 10, so they would normally sit in 1 segment)
        # We manually split them into 3 distinct segments to simulate active merging
        # Doc 1 & Doc 2 -> Segment A
        # Doc 3 -> Segment B
        # Doc 4 -> Segment C
        
        # Ingest Doc 1 & 2
        d1 = pipeline.ingest_document(collection_id, "d1", "document one content text")
        d2 = pipeline.ingest_document(collection_id, "d2", "document two content text")
        
        # Force next document to be written to a new segment by temporarily mocking segment search
        old_max = settings.MAX_VECTORS_PER_SEGMENT
        settings.MAX_VECTORS_PER_SEGMENT = 1 # Force creation of new segments
        
        try:
            d3 = pipeline.ingest_document(collection_id, "d3", "document three content text")
            d4 = pipeline.ingest_document(collection_id, "d4", "document four content text")
        finally:
            settings.MAX_VECTORS_PER_SEGMENT = old_max
            
        # Get all active segments
        segs = db_session.query(Embedding.segment_id).distinct().all()
        seg_ids = [s[0] for s in segs]
        assert len(seg_ids) == 3
        
        # 2. Soft-delete Doc 3 (flag its status as 2 in its segment file on disk)
        emb3 = db_session.query(Embedding).join(Chunk).filter(Chunk.document_id == d3.id).first()
        seg3_id = emb3.segment_id
        seg3_path = os.path.join(temp_dir, "segments", f"{seg3_id}.bin")
        
        with open(seg3_path, "rb") as f:
            data = f.read()
        seg_uuid, records = BinarySegmentSerializer.deserialize(data, 16)
        
        for r in records:
            if r['id'] == emb3.id:
                r['status'] = 2 # Tombstone
                
        updated_data = BinarySegmentSerializer.serialize(seg_uuid, records, 16)
        with open(seg3_path, "wb") as f:
            f.write(updated_data)
            
        # 3. Perform Compaction
        cache_manager = SegmentCacheManager()
        compactor = SegmentCompactor(db_session, cache_manager)
        
        assert compactor.should_compact(collection_id) is True
        
        merged_seg_id = compactor.compact(collection_id)
        assert merged_seg_id is not None
        
        # 4. Verify post-conditions
        # Check that old segment files are deleted
        for old_id in seg_ids:
            old_seg_path = os.path.join(temp_dir, "segments", f"{old_id}.bin")
            assert not os.path.exists(old_seg_path)
            
        # Check that merged segment file exists
        merged_path = os.path.join(temp_dir, "segments", f"{merged_seg_id}.bin")
        assert os.path.exists(merged_path)
        
        # Check that HNSW snapshot was written for the merged segment
        merged_snapshot_dir = os.path.join(temp_dir, "snapshots", str(merged_seg_id))
        assert os.path.exists(os.path.join(merged_snapshot_dir, "graph.bin"))
        
        # Check that the database Embedding rows have been updated
        # Chunks from d1, d2, d4 should point to merged_seg_id
        # Chunk from d3 (deleted) should have its Embedding record removed from the DB
        d3_emb = db_session.query(Embedding).filter(Embedding.id == emb3.id).first()
        assert d3_emb is None # Removed!
        
        active_embs = db_session.query(Embedding).join(Chunk).join(Document)\
            .filter(Document.collection_id == collection_id).all()
            
        assert len(active_embs) == 3 # Doc 1, 2, and 4
        for emb in active_embs:
            assert emb.segment_id == merged_seg_id
            assert emb.vector_idx in (0, 1, 2)
