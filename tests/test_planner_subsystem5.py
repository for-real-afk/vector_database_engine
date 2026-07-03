import os
import pytest
import uuid
from unittest.mock import MagicMock
import tempfile
from sqlalchemy.orm import Session

from models.database_models import Collection, Document, Chunk, Embedding
from services.retrieval.planner.strategy_registry import StrategyRegistry
from services.retrieval.planner.statistics import StatisticsCatalog, CollectionStatistics
from services.retrieval.planner.cost.cost_estimator import CostEstimator
from services.retrieval.planner.planner import QueryPlanner
from services.retrieval.planner.strategies.exact_strategy import ExactRetrievalStrategy
from services.retrieval.planner.feedback import PlannerFeedbackLoop
from services.retrieval.planner.explain import ExplainSearchExecutor, ExplainSearchResult, ActualExecutionMetrics
from storage.segments.writer import BinarySegmentSerializer
from core.config import settings

def test_explain_search_executor(db_session: Session):
    collection_id = uuid.uuid4()
    segment_id = uuid.uuid4()
    emb_id = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # 1. Create collection
        col = Collection(
            id=collection_id,
            name="explain_col",
            namespace="default",
            dimension=16,
            metric="Cosine"
        )
        db_session.add(col)
        db_session.commit()
        
        # 2. Add records
        doc = Document(id=uuid.uuid4(), collection_id=collection_id, status="completed")
        db_session.add(doc)
        db_session.commit()
        ch = Chunk(id=uuid.uuid4(), document_id=doc.id, text_content="explain match", chunk_index=0)
        db_session.add(ch)
        db_session.commit()
        emb = Embedding(id=emb_id, chunk_id=ch.id, segment_id=segment_id, vector_idx=0, vector_data=[0.1]*16)
        db_session.add(emb)
        db_session.commit()

        # Write segment file
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

        # 3. Setup Planner
        registry = StrategyRegistry()
        registry.register("EXACT", ExactRetrievalStrategy())
        
        catalog = StatisticsCatalog(db_session)
        estimator = CostEstimator()
        feedback = PlannerFeedbackLoop(learning_rate=0.5, initial_ratio=0.0001)
        
        planner = QueryPlanner(registry, catalog, estimator, feedback_loop=feedback)
        executor = ExplainSearchExecutor(planner)
        
        # 4. Run explain search
        explain_result = executor.explain_search(
            db=db_session,
            collection_id=collection_id,
            query_vector=[0.1]*16,
            k=5
        )
        
        assert isinstance(explain_result, ExplainSearchResult)
        assert explain_result.actual_metrics.result_count == 1
        assert len(explain_result.results) == 1
        assert explain_result.results[0]["id"] == emb_id
        
        # Verify JSON
        js = explain_result.to_json()
        assert '"actual_latency_ms"' in js
        assert '"strategy_name": "EXACT"' in js
        
        # Verify markdown report containing actual performance summary
        md = explain_result.to_markdown()
        assert "# 📋 Query Execution Plan Details" in md
        assert "Actual Execution Performance Summary" in md
        assert "Latency" in md

def test_planner_feedback_calibration():
    # Setup registry and catalog
    registry = StrategyRegistry()
    registry.register("EXACT", ExactRetrievalStrategy())
    
    catalog = MagicMock()
    stats = CollectionStatistics()
    stats.collection_size = 1000
    stats.dimension = 64
    catalog.get_statistics.return_value = stats
    
    # Init feedback loop with 0.5 learning rate, starting ratio 0.0001
    feedback = PlannerFeedbackLoop(learning_rate=0.5, initial_ratio=0.0001)
    
    planner = QueryPlanner(registry, catalog, CostEstimator(), feedback_loop=feedback)
    
    # First query plan
    plan1 = planner.plan(uuid.uuid4(), k=5)
    # Total cost of exact scan of 1000 items is: cpu_cost(1000) + io(0.0) + mem_cost(0.24MB * 1.0/1024 = ~0.24MB)
    # total_cost = 1000 + 0.24 = 1000.24 cost units.
    cost1 = plan1.cost_estimate.total_cost
    assert cost1 > 1000.0
    
    # Predict latency initially (using initial_ratio = 0.0001)
    # Expected: cost1 * 0.0001 = ~0.1 ms
    pred_latency1 = feedback.calibrate_latency(cost1)
    assert pred_latency1 == pytest.approx(cost1 * 0.0001)
    
    # Record actual query run: actual latency = 50 ms (much slower than predicted!)
    # Observed ratio = 50 / 1000.24 = 0.049987...
    # Calibrated ratio = 0.5 * 0.049987 + 0.5 * 0.0001 = 0.02504...
    feedback.record_execution(cost1, actual_latency_ms=50.0, strategy_name="EXACT")
    
    expected_ratio = 0.5 * (50.0 / cost1) + 0.5 * 0.0001
    assert feedback.calibration_ratio == pytest.approx(expected_ratio)
    
    # Plan again
    plan2 = planner.plan(uuid.uuid4(), k=5)
    
    # Latency prediction should now adaptively adjust to reflect the new multiplier calibration!
    # Expected: cost2 * 0.02504 = ~25.0 ms
    pred_latency2 = plan2.cost_estimate.assumptions.get("predicted_latency_ms")
    assert pred_latency2 == pytest.approx(plan2.cost_estimate.total_cost * expected_ratio)
    assert pred_latency2 > pred_latency1
