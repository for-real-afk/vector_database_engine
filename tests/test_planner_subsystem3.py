import os
import pytest
import uuid
from unittest.mock import MagicMock
import json
from sqlalchemy.orm import Session

from models.database_models import Collection, Embedding, Chunk, Document
from services.retrieval.planner.strategy_registry import StrategyRegistry
from services.retrieval.planner.statistics import StatisticsCatalog, CollectionStatistics
from services.retrieval.planner.cost.cost_estimator import CostEstimator
from services.retrieval.planner.planner import QueryPlanner
from services.retrieval.planner.strategies.exact_strategy import ExactRetrievalStrategy
from services.retrieval.planner.strategies.hnsw_strategy import HNSWRetrievalStrategy
from services.retrieval.planner.execution_plan import ExecutionPlan
from core.config import settings

# Define a broken strategy to test fallback handlers
class BrokenHNSWStrategy(HNSWRetrievalStrategy):
    def execute(self, db, collection_id, query_vector, k, allowed_ids=None):
        raise RuntimeError("Simulated graph serialization corruptions.")


def test_query_planner_exact_vs_hnsw_decisions():
    # Setup Registry
    registry = StrategyRegistry()
    registry.register("EXACT", ExactRetrievalStrategy())
    registry.register("HNSW", HNSWRetrievalStrategy())
    
    # Setup Estimator
    estimator = CostEstimator()
    
    # 1. Collection stats representing a small segment collection (no built index)
    stats_small = CollectionStatistics()
    stats_small.collection_size = 50
    stats_small.dimension = 64
    stats_small.graph_nodes = 0  # No graph built
    
    catalog_mock = MagicMock()
    catalog_mock.get_statistics.return_value = stats_small
    
    planner = QueryPlanner(registry, catalog_mock, estimator)
    plan_small = planner.plan(collection_id=uuid.uuid4(), k=5)
    
    assert plan_small.strategy_name == "EXACT"
    assert "no sealed graph" in "".join(plan_small.decision_trace).lower()

    # 2. Collection stats representing a huge index collection (graph built)
    stats_large = CollectionStatistics()
    stats_large.collection_size = 10000
    stats_large.graph_nodes = 10000
    stats_large.graph_layers = 4
    stats_large.average_degree = 16
    stats_large.dimension = 64
    
    catalog_mock.get_statistics.return_value = stats_large
    plan_large = planner.plan(collection_id=uuid.uuid4(), k=5)
    
    assert plan_large.strategy_name == "HNSW"
    assert "selected hnsw" in "".join(plan_large.decision_trace).lower()
    
    # Verify outputs
    js = plan_large.to_json()
    assert '"strategy_name": "HNSW"' in js
    
    md = plan_large.to_markdown()
    assert "# 📋 Query Execution Plan Details" in md
    assert "Chosen Strategy" in md
    assert "Decision Trace" in md

def test_query_planner_fault_tolerance_fallback(db_session: Session):
    import tempfile
    from storage.segments.writer import BinarySegmentSerializer
    
    collection_id = uuid.uuid4()
    segment_id = uuid.uuid4()
    emb_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # 1. Create collection
        col = Collection(
            id=collection_id,
            name="fallback_col",
            namespace="default",
            dimension=16,
            metric="Cosine"
        )
        db_session.add(col)
        db_session.commit()
        
        # 2. Insert mock database records so exact fallback can scan it successfully
        doc = Document(id=uuid.uuid4(), collection_id=collection_id, status="completed")
        db_session.add(doc)
        db_session.commit()
        ch = Chunk(id=uuid.uuid4(), document_id=doc.id, text_content="fallback match", chunk_index=0)
        db_session.add(ch)
        db_session.commit()
        emb = Embedding(id=emb_id, chunk_id=ch.id, segment_id=segment_id, vector_idx=0, vector_data=[0.1]*16)
        db_session.add(emb)
        db_session.commit()

        # Write dummy segment file
        segment_path = os.path.join(temp_dir, "segments", f"{segment_id}.bin")
        os.makedirs(os.path.dirname(segment_path), exist_ok=True)
        records = [{
            'id': emb_id,
            'vector': [0.1]*16,
            'status': 1,
            'payload': {}
        }]
        data = BinarySegmentSerializer.serialize(segment_id, records, dimension=16)
        with open(segment_path, "wb") as f:
            f.write(data)

        # Setup Strategy Registry with broken HNSW strategy
        registry = StrategyRegistry()
        registry.register("EXACT", ExactRetrievalStrategy())
        registry.register("HNSW", BrokenHNSWStrategy())
        
        # Setup stats indicating active graph exists
        stats = CollectionStatistics()
        stats.collection_size = 5000
        stats.graph_nodes = 5000
        stats.graph_layers = 3
        stats.average_degree = 16
        stats.dimension = 16
        
        catalog = MagicMock()
        catalog.get_statistics.return_value = stats
        
        planner = QueryPlanner(registry, catalog, CostEstimator())
        plan = planner.plan(collection_id=collection_id, k=5)
        
        # Planner chooses HNSW because graph nodes exist and estimated cost is lower
        assert plan.strategy_name == "HNSW"
        
        # When executing, the BrokenHNSWStrategy raises an exception
        # Fault-tolerance should catch it and execute EXACT strategy fallback successfully
        results = plan.execute(db_session, collection_id=collection_id, query_vector=[0.1]*16, k=5)
        
        assert len(results) == 1
        assert results[0]["id"] == emb_id

