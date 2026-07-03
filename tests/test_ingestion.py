import os
import uuid
import tempfile
import json
import pytest
import numpy as np
from sqlalchemy.orm import Session

from models.database_models import Collection, Document, Chunk, Embedding, AuditLog
from services.chunking.chunkers import (
    FixedSizeChunker, 
    SlidingWindowChunker, 
    RecursiveCharacterChunker, 
    SemanticChunker
)
from embeddings.providers import MockEmbeddingProvider
from storage.segments.writer import BinarySegmentSerializer
from services.ingestion.pipeline import IngestionPipeline
from core.config import settings

# 1. Chunker tests
def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=10)
    text = "abcdefghijklmnopqrstuvwxyz" # 26 chars
    chunks = chunker.chunk(text)
    assert len(chunks) == 3
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "klmnopqrst"
    assert chunks[2] == "uvwxyz"

def test_sliding_window_chunker():
    chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=4)
    text = "abcdefghijkl" # 12 chars
    chunks = chunker.chunk(text)
    # abcdefghij -> len 10
    # Next starts at 10 - 4 = 6 (g)
    # ghijkl -> len 6
    assert len(chunks) == 2
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "ghijkl"

def test_recursive_character_chunker():
    chunker = RecursiveCharacterChunker(chunk_size=15, chunk_overlap=5, separators=["\n", " "])
    text = "hello world\nthis is a test\nof recursive chunking"
    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    for c in chunks:
        assert len(c) <= 15

def test_semantic_chunker():
    provider = MockEmbeddingProvider(dimension=4)
    chunker = SemanticChunker(embedding_provider=provider, similarity_threshold=0.5)
    
    # Text with varying sentence meanings
    text = "The quick brown fox jumps over the lazy dog. Programming vector engines is fun. Databases require persistent logs."
    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    assert isinstance(chunks, list)

# 2. Embedding provider tests
def test_mock_embedding_provider():
    dimension = 128
    provider = MockEmbeddingProvider(dimension=dimension)
    
    vec1 = provider.embed_text("hello database engine")
    vec2 = provider.embed_text("hello database engine")
    vec3 = provider.embed_text("different text")
    
    assert len(vec1) == dimension
    assert vec1 == vec2 # Determinism check
    assert vec1 != vec3 # Variation check
    
    # Assert unit length normalization
    norm = np.linalg.norm(np.array(vec1))
    assert pytest.approx(norm) == 1.0

# 3. Binary serializer tests
def test_binary_segment_serialization():
    segment_id = uuid.uuid4()
    dimension = 1536
    
    # Create mock records
    records = []
    for _ in range(5):
        records.append({
            'id': uuid.uuid4(),
            'vector': np.random.randn(dimension).tolist(),
            'payload': {'chunk_text': 'This is a sample text for binary segment verification.'},
            'status': 1
        })
        
    # Serialize
    serialized_bytes = BinarySegmentSerializer.serialize(segment_id, records, dimension)
    assert len(serialized_bytes) > 64 # Must be larger than header
    
    # Verify alignment properties
    # The header size is 64. Index table is 5 * 32 = 160.
    # Total metadata size = 224. Next 64-byte boundary = 224 (it is already aligned!).
    # So vector block start must be exactly 224.
    magic, = struct_unpack_magic(serialized_bytes)
    assert magic == b'VSEG'
    
    # Deserialize
    deserialized_id, deserialized_records = BinarySegmentSerializer.deserialize(serialized_bytes, dimension)
    
    assert deserialized_id == segment_id
    assert len(deserialized_records) == 5
    for orig, decomp in zip(records, deserialized_records):
        assert orig['id'] == decomp['id']
        assert orig['payload'] == decomp['payload']
        assert orig['status'] == decomp['status']
        # Assert vectors are matching within float precision
        np.testing.assert_allclose(orig['vector'], decomp['vector'], rtol=1e-5)

def struct_unpack_magic(data: bytes):
    import struct
    return struct.unpack("<4s", data[:4])

# 4. Pipeline Integration Test (uses SQLite memory DB)
def test_ingestion_pipeline_flow(db_session: Session):
    # Setup mock collection
    collection_id = uuid.uuid4()
    # Create user
    user_id = uuid.uuid4()
    
    # Override settings storage root to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # Insert a collection metadata row
        col = Collection(
            id=collection_id,
            name="test_collection",
            namespace="default",
            dimension=128,
            metric="Cosine"
        )
        db_session.add(col)
        db_session.commit()
        
        provider = MockEmbeddingProvider(dimension=128)
        chunker = FixedSizeChunker(chunk_size=50)
        pipeline = IngestionPipeline(db_session, provider, chunker)
        
        # Run ingestion
        doc = pipeline.ingest_document(
            collection_id=collection_id,
            title="Database Engine Walkthrough",
            text_content="A database engine needs custom storage layouts and page buffer pools. High-performance vectors require SIMD instruction cache alignments.",
            metadata_dict={"department": "engineering", "security_level": 2}
        )
        
        assert doc.status == "completed"
        
        # Verify PostgreSQL database entries
        chunks = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) > 0
        
        # Verify active segment file exists on disk
        segment_files = os.listdir(os.path.join(temp_dir, "segments"))
        assert len(segment_files) == 1
        segment_id_str = segment_files[0].replace(".bin", "")
        segment_id = uuid.UUID(segment_id_str)
        
        # Read and check segment contents
        segment_file_path = os.path.join(temp_dir, "segments", segment_files[0])
        with open(segment_file_path, "rb") as f:
            seg_data = f.read()
            
        des_seg_id, des_records = BinarySegmentSerializer.deserialize(seg_data, 128)
        assert des_seg_id == segment_id
        assert len(des_records) == len(chunks)
