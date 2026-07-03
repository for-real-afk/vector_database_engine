import pytest
from services.retrieval.planner.statistics import CollectionStatistics
from services.retrieval.planner.cost.io_provider import IOCostProvider
from services.retrieval.planner.cost.memory_provider import MemoryCostProvider
from services.retrieval.planner.cost.cost_estimator import CostEstimator

def test_io_cost_provider_calculations():
    provider = IOCostProvider(cost_seq_page_read=1.0, cost_rand_page_read=4.0, cache_miss_penalty=2.5)
    
    stats = CollectionStatistics()
    stats.collection_size = 1000
    stats.dimension = 128
    
    # 1. 100% cache hit -> IO cost is 0.0
    stats.cache_hit_ratio = 1.0
    stats.cache_miss_ratio = 0.0
    io_cached = provider.calculate_cost(stats, k=5, strategy="EXACT")
    assert io_cached == 0.0
    
    # 2. 0% cache hit (100% miss) -> sequential pages accessed
    stats.cache_hit_ratio = 0.0
    stats.cache_miss_ratio = 1.0
    
    # Calculate segment page sizing:
    # Record size = 32 + 512 + 120 = 664 bytes
    # File bytes = 64 + 1000 * 664 = 664,064 bytes
    # Pages count = ceil(664,064 / 4096) = 163 pages
    io_exact = provider.calculate_cost(stats, k=5, strategy="EXACT")
    # seq read: 163 pages * 1.0 cost_seq = 163.0
    assert io_exact == 163.0
    
    # 3. HNSW random read cost:
    stats.graph_nodes = 1000
    stats.graph_layers = 3
    stats.average_degree = 16
    io_hnsw = provider.calculate_cost(stats, k=5, strategy="HNSW")
    # expected visited: 3 * 16 + 32 = 80
    # pages accessed: min(163, 80) = 80
    # faults: 80 * 1.0 miss = 80
    # rand read: 80 * 4.0 cost_rand * 2.5 miss_penalty = 800.0
    assert io_hnsw == 800.0

def test_memory_cost_provider_calculations():
    provider = MemoryCostProvider(memory_scale=1.0)
    
    stats = CollectionStatistics()
    stats.collection_size = 10000
    stats.dimension = 128
    
    # Exact scan memory
    mem_exact = provider.calculate_cost(stats, k=5, strategy="EXACT")
    # Active size: 10000
    # Bytes: 10000 * (512 + 120 + 4) = 6,360,000 bytes (~6.06 MB)
    assert mem_exact == pytest.approx(6360000 / (1024 * 1024))
    
    # HNSW graph memory
    stats.graph_nodes = 10000
    stats.average_degree = 16
    mem_hnsw = provider.calculate_cost(stats, k=5, strategy="HNSW")
    # Node bytes: 38 + 16 * 8 + 512 = 678 bytes
    # Graph resident: 10000 * 678 = 6,780,000 bytes
    # Temp queue bytes is small (~ visited node bitmap: 10000/8 = 1250 bytes)
    # Total: ~6.46 MB
    assert mem_hnsw == pytest.approx((6780000 + 1250 + 32 * 24) / (1024 * 1024), rel=1e-3)
    
    # LOWEST_MEMORY mode doubles memory cost value to penalize memory footprint
    mem_hnsw_low = provider.calculate_cost(stats, k=5, mode="LOWEST_MEMORY", strategy="HNSW")
    assert mem_hnsw_low == mem_hnsw * 2.0

def test_cost_estimator_complete_integration():
    estimator = CostEstimator()
    
    stats = CollectionStatistics()
    stats.collection_size = 1000
    stats.dimension = 128
    stats.cache_hit_ratio = 0.5
    stats.cache_miss_ratio = 0.5
    stats.graph_nodes = 1000
    stats.graph_layers = 3
    stats.average_degree = 16
    
    est = estimator.estimate_cost(stats, k=5, selectivity=1.0, mode="BALANCED", strategy="HNSW")
    
    # Verify that all component costs are non-zero
    assert est.cpu_cost > 0.0
    assert est.graph_cost > 0.0
    assert est.io_cost > 0.0
    assert est.memory_cost > 0.0
    assert est.total_cost == est.cpu_cost + est.graph_cost + est.io_cost + est.memory_cost
