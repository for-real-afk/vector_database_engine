import os
import uuid
import struct
import json
import tempfile
import pytest
from sqlalchemy.orm import Session

from models.database_models import Collection, Embedding, Chunk, Document
from services.retrieval.planner.strategy_registry import RetrievalStrategy, StrategyRegistry
from services.retrieval.planner.statistics import StatisticsCatalog
from storage.cache.segment_cache import SegmentCacheManager
from core.config import settings

# 1. Define a dummy strategy for registry testing
class DummyRetrievalStrategy(RetrievalStrategy):
    def execute(self, db, collection_id, query_vector, k, allowed_ids=None):
        return [{"id": "dummy_res"}]

    def estimate_cost(self, stats, k, filters=None):
        return {'cpu': 1.0, 'io': 0.0, 'memory': 0.0, 'total_cost': 1.0}


def test_strategy_registry():
    registry = StrategyRegistry()
    dummy = DummyRetrievalStrategy()
    
    # Assert empty list initially
    assert len(registry.list_strategies()) == 0
    
    # Register strategy
    registry.register("DUMMY", dummy)
    
    assert "DUMMY" in registry.list_strategies()
    assert registry.get_strategy("DUMMY") == dummy
    
    # Check case-insensitivity
    assert registry.get_strategy("dummy") == dummy
    
    with pytest.raises(KeyError):
        registry.get_strategy("NON_EXIST")

def test_statistics_catalog_gathering(db_session: Session):
    collection_id = uuid.uuid4()
    seg_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # Create collection
        col = Collection(
            id=collection_id,
            name="stats_col",
            namespace="default",
            dimension=16,
            metric="Cosine"
        )
        db_session.add(col)
        db_session.commit()
        
        # Insert mock database records
        doc = Document(id=uuid.uuid4(), collection_id=collection_id, status="completed")
        db_session.add(doc)
        db_session.commit()
        
        ch = Chunk(id=uuid.uuid4(), document_id=doc.id, text_content="text block", chunk_index=0)
        db_session.add(ch)
        db_session.commit()
        
        emb = Embedding(id=uuid.uuid4(), chunk_id=ch.id, segment_id=seg_id, vector_idx=0, vector_data=[0.1]*16)
        db_session.add(emb)
        db_session.commit()
        
        # Write mock graph.bin (HEADER_FORMAT = "<4s H H 16s I 36x")
        # magic=b'VHNS', version=1, max_level=2 (means 3 layers), entry_node_uuid, node_count=50
        snapshot_dir = os.path.join(temp_dir, "snapshots", str(seg_id))
        os.makedirs(snapshot_dir, exist_ok=True)
        
        header_bytes = struct.pack(
            "<4s H H 16s I 36x",
            b'VHNS',
            1,
            2,
            uuid.uuid4().bytes,
            50
        )
        with open(os.path.join(snapshot_dir, "graph.bin"), "wb") as f:
            f.write(header_bytes)
            
        # Write mock metadata.json
        meta_dict = {
            "dimension": 16,
            "metric": "Cosine",
            "M": 12,
            "M0": 24,
            "tombstones": [str(uuid.uuid4())] # 1 deleted vector
        }
        with open(os.path.join(snapshot_dir, "metadata.json"), "w") as f:
            json.dump(meta_dict, f)
            
        # Configure Cache Manager and simulate hits/misses
        cache_manager = SegmentCacheManager()
        cache_manager.hit_count = 8
        cache_manager.miss_count = 2
        cache_manager.current_memory_bytes = 4096
        
        # Run statistics catalog
        catalog = StatisticsCatalog(db_session, cache_manager)
        stats = catalog.get_statistics(collection_id)
        
        # Assert database counts
        assert stats.collection_size == 1
        assert stats.dimension == 16
        assert stats.sealed_segments == 1
        assert stats.growing_segments == 0
        
        # Assert binary file headers parse outcomes
        assert stats.graph_nodes == 50
        assert stats.graph_layers == 3
        
        # Assert metadata properties
        assert stats.average_degree == 12.0
        assert stats.max_degree == 24.0
        assert stats.deleted_vectors == 1
        
        # Assert cache telemetry properties
        assert stats.cache_hit_ratio == 0.8
        assert stats.cache_miss_ratio == pytest.approx(0.2)
        assert stats.cache_memory_bytes == 4096
        
        # Assert serializability
        d = stats.to_dict()
        assert d["collection_size"] == 1
        assert d["graph_nodes"] == 50
        assert d["cache_hit_ratio"] == 0.8
