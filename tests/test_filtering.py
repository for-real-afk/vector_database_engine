import os
import uuid
import tempfile
import pytest
import numpy as np
from sqlalchemy.orm import Session

from models.database_models import Collection, Embedding, Chunk, Document, Metadata
from services.ingestion.pipeline import IngestionPipeline
from services.chunking.chunkers import FixedSizeChunker
from embeddings.providers import MockEmbeddingProvider
from services.retrieval.filter_resolver import MetadataFilterResolver
from services.retrieval.hybrid_search import HybridSearchCoordinator
from core.config import settings

def test_metadata_filter_resolver(db_session: Session):
    collection_id = uuid.uuid4()
    
    # Setup database layout
    col = Collection(
        id=collection_id,
        name="filter_col",
        namespace="default",
        dimension=8,
        metric="Cosine"
    )
    db_session.add(col)
    
    doc = Document(id=uuid.uuid4(), collection_id=collection_id, status="completed")
    db_session.add(doc)
    db_session.commit()
    
    # Setup chunks and embeddings
    ch1 = Chunk(id=uuid.uuid4(), document_id=doc.id, text_content="t1", chunk_index=0)
    ch2 = Chunk(id=uuid.uuid4(), document_id=doc.id, text_content="t2", chunk_index=1)
    ch3 = Chunk(id=uuid.uuid4(), document_id=doc.id, text_content="t3", chunk_index=2)
    db_session.add_all([ch1, ch2, ch3])
    db_session.commit()
    
    emb1 = Embedding(id=uuid.uuid4(), chunk_id=ch1.id, vector_data=[0.1]*8)
    emb2 = Embedding(id=uuid.uuid4(), chunk_id=ch2.id, vector_data=[0.2]*8)
    emb3 = Embedding(id=uuid.uuid4(), chunk_id=ch3.id, vector_data=[0.3]*8)
    db_session.add_all([emb1, emb2, emb3])
    db_session.commit()
    
    # Setup Metadata
    # ch1 matches {"category": "db"}
    # ch2 matches {"category": "web"}
    # ch3 matches {"category": "db", "priority": 1}
    m1 = Metadata(id=uuid.uuid4(), document_id=doc.id, chunk_id=ch1.id, key="category", value="db")
    m2 = Metadata(id=uuid.uuid4(), document_id=doc.id, chunk_id=ch2.id, key="category", value="web")
    m3 = Metadata(id=uuid.uuid4(), document_id=doc.id, chunk_id=ch3.id, key="category", value="db")
    m4 = Metadata(id=uuid.uuid4(), document_id=doc.id, chunk_id=ch3.id, key="priority", value=1)
    db_session.add_all([m1, m2, m3, m4])
    db_session.commit()
    
    resolver = MetadataFilterResolver(db_session)
    
    # Query matching category=db
    allowed1 = resolver.resolve_filters(collection_id, {"category": "db"})
    assert allowed1 == {emb1.id, emb3.id}
    
    # Query matching category=db AND priority=1
    allowed2 = resolver.resolve_filters(collection_id, {"category": "db", "priority": 1})
    assert allowed2 == {emb3.id}
    
    # Query non-matching
    allowed3 = resolver.resolve_filters(collection_id, {"category": "non_exist"})
    assert allowed3 == set()

def test_hybrid_search_with_pre_filtering_and_scoring(db_session: Session):
    collection_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # Limit segment capacity to build graphs automatically on sealing
        old_max = settings.MAX_VECTORS_PER_SEGMENT
        settings.MAX_VECTORS_PER_SEGMENT = 4
        
        try:
            # Create collection
            col = Collection(
                id=collection_id,
                name="hybrid_col",
                namespace="default",
                dimension=16,
                metric="Cosine"
            )
            db_session.add(col)
            db_session.commit()
            
            provider = MockEmbeddingProvider(dimension=16)
            chunker = FixedSizeChunker(chunk_size=50)
            pipeline = IngestionPipeline(db_session, provider, chunker)
            
            # Ingest documents with metadata tags
            # 4 documents matching "apple"
            # 2 documents matching "banana"
            pipeline.ingest_document(collection_id, "Apple 1", "Apples are red fruits.", {"tag": "apple"})
            pipeline.ingest_document(collection_id, "Apple 2", "Apples grow on trees.", {"tag": "apple"})
            pipeline.ingest_document(collection_id, "Apple 3", "Apples make sweet cider.", {"tag": "apple"})
            pipeline.ingest_document(collection_id, "Apple 4", "Apple pies are delicious.", {"tag": "apple"})
            
            # At this point, the first 4 documents have filled segment 1 (sealed, has graph snapshot)
            
            pipeline.ingest_document(collection_id, "Banana 1", "Bananas are yellow fruits.", {"tag": "banana"})
            pipeline.ingest_document(collection_id, "Banana 2", "Bananas grow in tropical zones.", {"tag": "banana"})
            
            # Banana documents are in segment 2 (growing, no graph snapshot yet)
            
            coordinator = HybridSearchCoordinator(db_session, provider)
            
            # Query "sweet fruits" with tag filter "apple"
            # Expected: only apple-tagged documents should be returned, despite bananas also having "fruits"
            results = coordinator.search(
                collection_id=collection_id,
                query_text="sweet fruits",
                filters={"tag": "apple"},
                k=5,
                alpha=0.5
            )
            
            assert len(results) > 0
            for r in results:
                # Resolve tag from DB to assert correctness
                ch_id = r['payload']['chunk_id']
                meta_tag = db_session.query(Metadata.value).filter(Metadata.chunk_id == uuid.UUID(ch_id)).filter(Metadata.key == "tag").scalar()
                assert meta_tag == "apple"
                assert "Banana" not in r['payload']['title']
                
            # Query "yellow fruits" with no filter
            # Expected: Bananas should be ranked high due to text match and semantic vector match
            results_unfiltered = coordinator.search(
                collection_id=collection_id,
                query_text="yellow fruits",
                k=1,
                alpha=0.5
            )
            assert len(results_unfiltered) == 1
            assert "Banana 1" in results_unfiltered[0]['payload']['title']
            
        finally:
            settings.MAX_VECTORS_PER_SEGMENT = old_max
