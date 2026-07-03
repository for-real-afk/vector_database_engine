import pytest
import uuid
from unittest.mock import MagicMock
from sqlalchemy.orm import Session

from models.database_models import Collection, Document, Chunk, Metadata
from services.retrieval.planner.selectivity_estimator import MetadataSelectivityEstimator
from services.retrieval.planner.planner import QueryPlanner
from services.retrieval.planner.strategy_registry import StrategyRegistry
from services.retrieval.planner.statistics import StatisticsCatalog, CollectionStatistics
from services.retrieval.planner.cost.cost_estimator import CostEstimator
from services.retrieval.planner.strategies.exact_strategy import ExactRetrievalStrategy

def test_metadata_selectivity_estimation(db_session: Session):
    collection_id = uuid.uuid4()
    
    # 1. Create collection
    col = Collection(
        id=collection_id,
        name="selectivity_col",
        namespace="default",
        dimension=16,
        metric="Cosine"
    )
    db_session.add(col)
    db_session.commit()
    
    # 2. Ingest 10 document chunk records with structured metadata
    # - 4 documents with {"category": "A", "priority": 1}
    # - 2 documents with {"category": "A", "priority": 2}
    # - 4 documents with {"category": "B", "priority": 2}
    for i in range(10):
        doc = Document(id=uuid.uuid4(), collection_id=collection_id, status="completed")
        db_session.add(doc)
        db_session.commit()
        
        ch = Chunk(id=uuid.uuid4(), document_id=doc.id, text_content=f"chunk {i}", chunk_index=0)
        db_session.add(ch)
        db_session.commit()
        
        # Add metadata properties
        if i < 4:
            m1 = Metadata(id=uuid.uuid4(), chunk_id=ch.id, key="category", value="A")
            m2 = Metadata(id=uuid.uuid4(), chunk_id=ch.id, key="priority", value=1)
            db_session.add(m1)
            db_session.add(m2)
        elif i < 6:
            m1 = Metadata(id=uuid.uuid4(), chunk_id=ch.id, key="category", value="A")
            m2 = Metadata(id=uuid.uuid4(), chunk_id=ch.id, key="priority", value=2)
            db_session.add(m1)
            db_session.add(m2)
        else:
            m1 = Metadata(id=uuid.uuid4(), chunk_id=ch.id, key="category", value="B")
            m2 = Metadata(id=uuid.uuid4(), chunk_id=ch.id, key="priority", value=2)
            db_session.add(m1)
            db_session.add(m2)
            
        db_session.commit()
        
    estimator = MetadataSelectivityEstimator(db_session)
    
    # 3. Assert selectivities
    # Empty filters
    assert estimator.estimate_selectivity(collection_id, None) == 1.0
    assert estimator.estimate_selectivity(collection_id, {}) == 1.0
    
    # Single key filter
    # "category" A is in 6/10 = 0.60
    assert estimator.estimate_selectivity(collection_id, {"category": "A"}) == pytest.approx(0.60)
    # "category" B is in 4/10 = 0.40
    assert estimator.estimate_selectivity(collection_id, {"category": "B"}) == pytest.approx(0.40)
    # "priority" 1 is in 4/10 = 0.40
    assert estimator.estimate_selectivity(collection_id, {"priority": 1}) == pytest.approx(0.40)
    
    # Combined AND filter
    # "category" A and "priority" 1 matches 4 chunks = 0.40
    assert estimator.estimate_selectivity(collection_id, {"category": "A", "priority": 1}) == pytest.approx(0.40)
    # "category" A and "priority" 2 matches 2 chunks = 0.20
    assert estimator.estimate_selectivity(collection_id, {"category": "A", "priority": 2}) == pytest.approx(0.20)
    
    # Non-existent filter should fall back to min floor of 0.001
    assert estimator.estimate_selectivity(collection_id, {"category": "NON_EXIST"}) == 0.001

def test_query_planner_integration_with_selectivity(db_session: Session):
    collection_id = uuid.uuid4()
    
    col = Collection(id=collection_id, name="opt_col", namespace="default", dimension=16, metric="Cosine")
    db_session.add(col)
    db_session.commit()
    
    # Ingest 10 chunks with category A
    for i in range(10):
        doc = Document(id=uuid.uuid4(), collection_id=collection_id, status="completed")
        db_session.add(doc)
        db_session.commit()
        ch = Chunk(id=uuid.uuid4(), document_id=doc.id, text_content=f"chunk {i}", chunk_index=0)
        db_session.add(ch)
        db_session.commit()
        m = Metadata(id=uuid.uuid4(), chunk_id=ch.id, key="category", value="A")
        db_session.add(m)
        db_session.commit()
        
    registry = StrategyRegistry()
    registry.register("EXACT", ExactRetrievalStrategy())
    
    catalog = MagicMock()
    stats = CollectionStatistics()
    stats.collection_size = 10
    stats.dimension = 16
    catalog.get_statistics.return_value = stats
    
    sel_estimator = MetadataSelectivityEstimator(db_session)
    planner = QueryPlanner(registry, catalog, CostEstimator(), sel_estimator)
    
    plan = planner.plan(collection_id=collection_id, k=5, filters={"category": "A"})
    
    # Verify that selectivity tracing was invoked and logged
    trace_joined = "".join(plan.decision_trace).lower()
    assert "resolving metadata selectivity" in trace_joined
    assert "selectivity estimate: 100.00%" in trace_joined
