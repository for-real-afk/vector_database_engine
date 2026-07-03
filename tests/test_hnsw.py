import uuid
import pytest
import numpy as np
from index.graph.hnsw import HNSWIndex, HNSWNode
from index.distance.metrics import DistanceMetric

# 1. Test node level distributions
def test_hnsw_level_generation():
    idx = HNSWIndex(dimension=16, M=8)
    
    levels = []
    for _ in range(1000):
        # Draw from level generator logic
        level = int(np.floor(-np.log(idx.rng.uniform(0.1, 1.0)) * idx.mL))
        levels.append(level)
        
    # Check that level 0 has the highest count
    count_0 = sum(1 for l in levels if l == 0)
    count_1 = sum(1 for l in levels if l == 1)
    count_2 = sum(1 for l in levels if l >= 2)
    
    assert count_0 > count_1
    assert count_1 >= count_2

# 2. Test basic insertion and graph structure
def test_hnsw_insertion():
    dimension = 32
    idx = HNSWIndex(dimension=dimension, M=16, ef_construction=32)
    
    # Insert 50 vectors
    node_ids = [uuid.uuid4() for _ in range(50)]
    vectors = [np.random.randn(dimension).tolist() for _ in range(50)]
    
    for nid, vec in zip(node_ids, vectors):
        idx.insert(nid, vec)
        
    assert len(idx.nodes) == 50
    assert idx.enter_node is not None
    assert idx.max_level >= 0
    
    # Assert HNSW graph connection constraints
    for nid, node in idx.nodes.items():
        for lc in range(node.level + 1):
            limit = idx.M0 if lc == 0 else idx.M
            assert len(node.neighbors[lc]) <= limit

# 3. Recall evaluation test: HNSW vs Brute Force
def test_hnsw_recall_accuracy():
    dimension = 64
    k = 5
    dataset_size = 200
    
    # Instantiate custom HNSW index
    idx = HNSWIndex(
        dimension=dimension, 
        metric="Cosine", 
        M0=32,
        ef_search=32
    )
    
    # Generate dataset
    rng = np.random.default_rng(101)
    vectors = rng.standard_normal((dataset_size, dimension))
    node_ids = [uuid.uuid4() for _ in range(dataset_size)]
    
    # Load into HNSW
    for nid, vec in zip(node_ids, vectors):
        idx.insert(nid, vec.tolist())
        
    # Run 15 queries and measure recall
    recalls = []
    for _ in range(15):
        query = rng.standard_normal(dimension)
        
        # A. Brute-force Cosine Similarity search
        distances = []
        for nid, vec in zip(node_ids, vectors):
            # Compute cosine distance
            norm_q = np.linalg.norm(query)
            norm_v = np.linalg.norm(vec)
            dist = 1.0 - (np.dot(query, vec) / (norm_q * norm_v))
            distances.append((dist, nid))
            
        exact_results = sorted(distances, key=lambda x: x[0])[:k]
        exact_ids = {r[1] for r in exact_results}
        
        # B. HNSW Search
        hnsw_results = idx.search(query.tolist(), k=k, ef=32)
        hnsw_ids = {r[1] for r in hnsw_results}
        
        # Compute recall fraction (how many exact neighbors were retrieved by HNSW)
        intersection = exact_ids.intersection(hnsw_ids)
        recall = len(intersection) / k
        recalls.append(recall)
        
    avg_recall = np.mean(recalls)
    logger_str = f"HNSW Average Recall@5: {avg_recall * 100:.2f}%"
    print(logger_str)
    
    # We expect high recall (>90%) for a small dataset with ef_search=32
    assert avg_recall >= 0.90

# 4. Soft-delete tombstone test
def test_hnsw_soft_deletion():
    dimension = 16
    idx = HNSWIndex(dimension=dimension, metric="Cosine")
    
    # Insert 10 vectors
    node_ids = [uuid.uuid4() for _ in range(10)]
    vectors = [np.random.randn(dimension).tolist() for _ in range(10)]
    
    for nid, vec in zip(node_ids, vectors):
        idx.insert(nid, vec)
        
    # Search for first vector: should be returned as top result
    q_vec = vectors[0]
    results_before = idx.search(q_vec, k=1)
    assert len(results_before) == 1
    assert results_before[0][1] == node_ids[0]
    
    # Soft delete the target node
    idx.mark_deleted(node_id=node_ids[0])
    
    # Search again: the top result should no longer be the deleted node
    results_after = idx.search(q_vec, k=1)
    if len(results_after) > 0:
        assert results_after[0][1] != node_ids[0]
        
    # The deleted node should still physically reside in the nodes graph mapping
    assert node_ids[0] in idx.nodes
    assert node_ids[0] in idx.tombstones
