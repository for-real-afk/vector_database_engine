import pytest
from dataclasses import FrozenInstanceError
import json

from services.retrieval.planner.cost.cost_estimate import CostEstimate
from services.retrieval.planner.cost.cpu_provider import CPUCostProvider
from services.retrieval.planner.cost.cost_estimator import CostEstimator
from services.retrieval.planner.statistics import CollectionStatistics

def test_cost_estimate_immutability_and_comparisons():
    est1 = CostEstimate(
        cpu_cost=10.0, graph_cost=0.0, io_cost=5.0, memory_cost=1.0,
        cache_cost=0.0, metadata_cost=0.0, ranking_cost=0.0, serialization_cost=0.0,
        total_cost=16.0, confidence_score=0.9, assumptions={"k": 5}
    )
    
    # Assert immutability
    with pytest.raises((FrozenInstanceError, AttributeError)):
        est1.total_cost = 20.0
        
    est2 = CostEstimate(
        cpu_cost=20.0, graph_cost=0.0, io_cost=10.0, memory_cost=2.0,
        cache_cost=0.0, metadata_cost=0.0, ranking_cost=0.0, serialization_cost=0.0,
        total_cost=32.0, confidence_score=0.9, assumptions={"k": 5}
    )
    
    # Assert comparisons
    assert est1 < est2
    assert est2 > est1
    assert est1 <= est2
    assert est2 >= est2
    
    # Assert serialization
    d = est1.to_dict()
    assert d["total_cost"] == 16.0
    
    js = est1.to_json()
    assert '"total_cost": 16.0' in js
    
    md = est1.to_markdown()
    assert "### 📊 Cost Estimate Report" in md
    assert "Total Cost units" in md

def test_cpu_cost_provider_behavior():
    provider = CPUCostProvider(cost_per_distance_comp=2.0)
    
    stats = CollectionStatistics()
    stats.collection_size = 1000
    stats.dimension = 64
    
    # Exact strategy cost
    cost_exact = provider.calculate_cost(stats, k=5, selectivity=1.0, mode="BALANCED", strategy="EXACT")
    # Cost should be 1000 * 2.0 = 2000.0
    assert cost_exact == 2000.0
    
    # Mode multiplier: LOWEST_LATENCY has 1.5x multiplier
    cost_exact_lat = provider.calculate_cost(stats, k=5, selectivity=1.0, mode="LOWEST_LATENCY", strategy="EXACT")
    assert cost_exact_lat == 3000.0
    
    # HNSW strategy cost with stats
    stats.graph_nodes = 1000
    stats.graph_layers = 3
    stats.average_degree = 16
    
    cost_hnsw = provider.calculate_cost(stats, k=5, selectivity=1.0, mode="BALANCED", strategy="HNSW")
    # Visited: 3 * 16 + 32 = 80
    # Computations: 80 * 16 = 1280
    # Cost: 1280 * 2.0 = 2560.0
    assert cost_hnsw == 2560.0
    
    # Low selectivity filter penalty on HNSW (selectivity = 0.1)
    cost_hnsw_filt = provider.calculate_cost(stats, k=5, selectivity=0.1, mode="BALANCED", strategy="HNSW")
    # Should be multiplied by penalty: 1 / 0.1 = 10x
    assert cost_hnsw_filt == 2560.0 * 10.0

def test_cost_estimator_integration():
    estimator = CostEstimator()
    
    stats = CollectionStatistics()
    stats.collection_size = 500
    stats.dimension = 64
    
    est = estimator.estimate_cost(stats, k=5, selectivity=1.0, mode="BALANCED", strategy="EXACT")
    
    assert isinstance(est, CostEstimate)
    assert est.cpu_cost == 500.0
    assert est.total_cost == 500.0
    assert est.confidence_score == 1.0
    
    # If stats are empty, confidence score should drop
    stats_empty = CollectionStatistics()
    stats_empty.collection_size = 0
    est_empty = estimator.estimate_cost(stats_empty, k=5, strategy="EXACT")
    assert est_empty.confidence_score == 0.3
