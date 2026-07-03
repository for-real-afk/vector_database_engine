import os
import uuid
import tempfile
import pytest
import numpy as np
from index.graph.hnsw import HNSWIndex
from index.serialization.persistence import HNSWSerializer, HNSWIndexManager

def test_hnsw_serialization_deserialization_flow():
    dimension = 32
    idx = HNSWIndex(dimension=dimension, metric="Cosine")
    
    # 1. Insert some vectors
    node_ids = [uuid.uuid4() for _ in range(20)]
    vectors = [np.random.randn(dimension).tolist() for _ in range(20)]
    
    # Setup mock vector resolver cache and segment coordinates
    vector_cache = {}
    segment_mappings = {}
    mock_seg_id = uuid.uuid4()
    
    for i, (nid, vec) in enumerate(zip(node_ids, vectors)):
        idx.insert(nid, vec)
        vector_cache[nid] = vec
        segment_mappings[nid] = (mock_seg_id, i)
        
    # Define vector resolver function
    def mock_resolver(n_uuid, s_uuid, v_idx):
        return vector_cache[n_uuid]

    # 2. Serialize graph to bytes
    serialized_data = HNSWSerializer.serialize(idx, segment_mappings)
    assert len(serialized_data) > 64 # Larger than header
    
    # 3. Deserialize back to reconstructed index
    reconstructed_idx, restored_mappings = HNSWSerializer.deserialize(
        serialized_data,
        dimension,
        "Cosine",
        mock_resolver
    )
    
    # 4. Assert graph equality
    assert len(reconstructed_idx.nodes) == len(idx.nodes)
    assert reconstructed_idx.max_level == idx.max_level
    assert reconstructed_idx.enter_node.id == idx.enter_node.id
    
    for nid, orig_node in idx.nodes.items():
        recon_node = reconstructed_idx.nodes[nid]
        assert orig_node.level == recon_node.level
        np.testing.assert_allclose(orig_node.vector, recon_node.vector)
        
        # Verify neighbor lists per layer
        for lc in range(orig_node.level + 1):
            orig_neighbor_ids = {n.id for n in orig_node.neighbors[lc]}
            recon_neighbor_ids = {n.id for n in recon_node.neighbors[lc]}
            assert orig_neighbor_ids == recon_neighbor_ids

def test_snapshot_and_restore_recall_check():
    dimension = 64
    idx = HNSWIndex(dimension=dimension, metric="L2", M=8, M0=16)
    
    node_ids = [uuid.uuid4() for _ in range(50)]
    vectors = [np.random.randn(dimension).tolist() for _ in range(50)]
    
    vector_cache = {}
    segment_mappings = {}
    mock_seg_id = uuid.uuid4()
    
    for i, (nid, vec) in enumerate(zip(node_ids, vectors)):
        idx.insert(nid, vec)
        vector_cache[nid] = vec
        segment_mappings[nid] = (mock_seg_id, i)
        
    def mock_resolver(n_uuid, s_uuid, v_idx):
        return vector_cache[n_uuid]
        
    # Mark a node as deleted (Tombstone)
    idx.mark_deleted(node_ids[0])
    
    # Record query results on the original index
    query_vec = np.random.randn(dimension).tolist()
    orig_results = idx.search(query_vec, k=5)
    
    # Snapshot to disk
    with tempfile.TemporaryDirectory() as temp_dir:
        HNSWIndexManager.snapshot(temp_dir, idx, segment_mappings)
        
        assert os.path.exists(os.path.join(temp_dir, "graph.bin"))
        assert os.path.exists(os.path.join(temp_dir, "metadata.json"))
        
        # Restore index from snapshot
        restored_idx, restored_mappings = HNSWIndexManager.restore(temp_dir, mock_resolver)
        
        # Search again on the restored index
        restored_results = restored_idx.search(query_vec, k=5)
        
        # Verify that search results match exactly (same candidates and distance scores!)
        assert len(restored_results) == len(orig_results)
        for orig_res, rest_res in zip(orig_results, restored_results):
            assert orig_res[0] == pytest.approx(rest_res[0], rel=1e-5) # Score
            assert orig_res[1] == rest_res[1] # UUID
            
        # Verify tombstone preservation
        assert node_ids[0] in restored_idx.tombstones
        assert node_ids[0] in restored_idx.nodes

def test_deserializer_corruption_guards():
    # Attempt deserialization with bad magic
    bad_data = b"BAD_MAGIC_HEADER_BYTES" + (b"\x00" * 100)
    
    def dummy_resolver(n, s, v):
        return []
        
    with pytest.raises(ValueError, match="Not a valid HNSW graph file"):
        HNSWSerializer.deserialize(bad_data, 16, "Cosine", dummy_resolver)
