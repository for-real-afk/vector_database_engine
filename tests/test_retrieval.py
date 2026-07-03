import os
import uuid
import tempfile
import pytest
import numpy as np
from sqlalchemy.orm import Session

from models.database_models import Collection, Embedding, Chunk
from index.distance.metrics import DistanceMetric
from services.retrieval.retriever import ExactRetriever
from services.ingestion.pipeline import IngestionPipeline
from services.chunking.chunkers import FixedSizeChunker
from embeddings.providers import MockEmbeddingProvider
from storage.segments.writer import BinarySegmentSerializer
from core.config import settings

# 1. Math metric tests
def test_distance_metrics():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    w = np.array([1.0, 0.0, 0.0])
    
    # Cosine Similarity
    assert DistanceMetric.cosine_similarity(u, v) == 0.0 # orthogonal
    assert DistanceMetric.cosine_similarity(u, w) == 1.0 # identical
    
    # L2 distance
    assert DistanceMetric.l2_distance(u, v) == pytest.approx(np.sqrt(2.0))
    assert DistanceMetric.l2_distance(u, w) == 0.0
    
    # Dot Product
    assert DistanceMetric.dot_product(u, v) == 0.0
    assert DistanceMetric.dot_product(u, w) == 1.0
    
    # Manhattan distance
    assert DistanceMetric.manhattan_distance(u, v) == 2.0
    assert DistanceMetric.manhattan_distance(u, w) == 0.0

# 2. Search Integration tests
def test_exact_retriever_search_flow(db_session: Session):
    collection_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # Create mock collection with dimension=64
        col = Collection(
            id=collection_id,
            name="retrieval_collection",
            namespace="default",
            dimension=64,
            metric="Cosine"
        )
        db_session.add(col)
        db_session.commit()
        
        # Setup ingestion pipeline
        provider = MockEmbeddingProvider(dimension=64)
        chunker = FixedSizeChunker(chunk_size=100)
        pipeline = IngestionPipeline(db_session, provider, chunker)
        
        # Ingest 3 distinct documents
        doc1 = pipeline.ingest_document(
            collection_id=collection_id,
            title="Apples Doc",
            text_content="Apples are fresh and sweet red fruits."
        )
        doc2 = pipeline.ingest_document(
            collection_id=collection_id,
            title="Databases Doc",
            text_content="Databases are engines that write WAL logs and guarantee transactions."
        )
        doc3 = pipeline.ingest_document(
            collection_id=collection_id,
            title="React Doc",
            text_content="Vite and React are used to compile web application dashboards."
        )
        
        retriever = ExactRetriever(db_session)
        
        # Query matching databases (Mock embedding provider generates deterministic vectors from query text)
        q_vec = provider.embed_text("databases write transactions")
        results = retriever.search(collection_id, q_vec, k=1)
        
        assert len(results) == 1
        # It should correspond to doc2 payload
        assert "Databases" in results[0]['payload']['text']
        
        # Query fruits
        q_vec_fruits = provider.embed_text("sweet red fruits")
        results_fruits = retriever.search(collection_id, q_vec_fruits, k=1)
        assert len(results_fruits) == 1
        assert "Apples" in results_fruits[0]['payload']['text']

def test_tombstone_filtering(db_session: Session):
    collection_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        col = Collection(
            id=collection_id,
            name="tombstone_col",
            namespace="default",
            dimension=64,
            metric="Cosine"
        )
        db_session.add(col)
        db_session.commit()
        
        provider = MockEmbeddingProvider(dimension=64)
        chunker = FixedSizeChunker(chunk_size=100)
        pipeline = IngestionPipeline(db_session, provider, chunker)
        
        doc = pipeline.ingest_document(
            collection_id=collection_id,
            title="Temp Doc",
            text_content="Ingesting some data vectors to be marked as deleted soon."
        )
        
        retriever = ExactRetriever(db_session)
        q_vec = provider.embed_text("some data vectors")
        
        # Normal search: should yield candidate
        results_before = retriever.search(collection_id, q_vec, k=5)
        assert len(results_before) > 0
        
        # Locate segment on disk and manually edit record status byte to 2 (Tombstone)
        emb = db_session.query(Embedding).join(Chunk).filter(Chunk.document_id == doc.id).first()
        seg_id = emb.segment_id
        segment_path = os.path.join(temp_dir, "segments", f"{seg_id}.bin")
        
        # Load binary segment, update record status to 2, and write it back
        with open(segment_path, "rb") as f:
            data = f.read()
            
        seg_uuid, records = BinarySegmentSerializer.deserialize(data, 64)
        # Mark all records in this segment as deleted
        for r in records:
            r['status'] = 2
            
        # Re-serialize
        updated_data = BinarySegmentSerializer.serialize(seg_uuid, records, 64)
        with open(segment_path, "wb") as f:
            f.write(updated_data)
            
        # Search again: should yield NO results because of tombstone filtering
        results_after = retriever.search(collection_id, q_vec, k=5)
        assert len(results_after) == 0

def test_l2_metric_sorting(db_session: Session):
    collection_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # Create L2 collection
        col = Collection(
            id=collection_id,
            name="l2_col",
            namespace="default",
            dimension=32,
            metric="L2"
        )
        db_session.add(col)
        db_session.commit()
        
        provider = MockEmbeddingProvider(dimension=32)
        chunker = FixedSizeChunker(chunk_size=100)
        pipeline = IngestionPipeline(db_session, provider, chunker)
        
        pipeline.ingest_document(collection_id, "doc1", "first record text")
        pipeline.ingest_document(collection_id, "doc2", "second record text")
        
        retriever = ExactRetriever(db_session)
        q_vec = provider.embed_text("first record text")
        
        results = retriever.search(collection_id, q_vec, k=5)
        assert len(results) == 2
        # For L2 metric, lower score represents closer vector.
        # Check that results[0] has lower L2 score than results[1]
        assert results[0]['score'] < results[1]['score']
