import os
import uuid
import tempfile
import pytest
import numpy as np
from storage.segments.writer import BinarySegmentSerializer
from storage.cache.segment_cache import SegmentCacheManager
from core.config import settings

def test_cache_lazy_loading_and_counters():
    dimension = 128
    
    # 1. Create a mock segment and serialize it to a temp file
    seg_id = uuid.uuid4()
    records = []
    for _ in range(5):
        records.append({
            'id': uuid.uuid4(),
            'vector': np.random.randn(dimension).tolist(),
            'payload': {'text': 'cached text block'},
            'status': 1
        })
        
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        
        # Write binary segment file to temp segments directory
        os.makedirs(os.path.join(temp_dir, "segments"), exist_ok=True)
        segment_bytes = BinarySegmentSerializer.serialize(seg_id, records, dimension)
        segment_path = os.path.join(temp_dir, "segments", f"{seg_id}.bin")
        with open(segment_path, "wb") as f:
            f.write(segment_bytes)
            
        # 2. Instantiate cache manager
        manager = SegmentCacheManager(max_memory_bytes=10 * 1024 * 1024) # 10MB limit
        assert manager.hit_count == 0
        assert manager.miss_count == 0
        assert len(manager.cache) == 0
        
        # 3. Retrieve vector - should trigger Cache Miss & Disk Load
        v1 = manager.get_vector(seg_id, 0, dimension)
        assert len(v1) == dimension
        assert manager.miss_count == 1
        assert manager.hit_count == 0
        assert len(manager.cache) == 1
        
        # 4. Retrieve vector again - should trigger Cache Hit
        v2 = manager.get_vector(seg_id, 0, dimension)
        np.testing.assert_allclose(v1, v2)
        assert manager.miss_count == 1
        assert manager.hit_count == 1
        assert manager.hit_ratio == 0.5
        
        # 5. Retrieve payload and check
        payload = manager.get_payload(seg_id, 0, dimension)
        assert payload == {'text': 'cached text block'}

def test_lru_cache_eviction_mechanics():
    dimension = 64
    
    # Helper to generate a segment file
    def create_mock_segment(temp_dir, seg_uuid, count=10):
        records = []
        for _ in range(count):
            records.append({
                'id': uuid.uuid4(),
                'vector': np.random.randn(dimension).tolist(),
                'payload': {'text': 'lru segment buffer item text'},
                'status': 1
            })
        segment_bytes = BinarySegmentSerializer.serialize(seg_uuid, records, dimension)
        segment_path = os.path.join(temp_dir, "segments", f"{seg_uuid}.bin")
        with open(segment_path, "wb") as f:
            f.write(segment_bytes)
            
    # Setup 3 segments
    seg1 = uuid.uuid4()
    seg2 = uuid.uuid4()
    seg3 = uuid.uuid4()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.STORAGE_ROOT = temp_dir
        os.makedirs(os.path.join(temp_dir, "segments"), exist_ok=True)
        
        create_mock_segment(temp_dir, seg1)
        create_mock_segment(temp_dir, seg2)
        create_mock_segment(temp_dir, seg3)
        
        # Calculate memory footprint. A segment of 10 vectors of dimension 64 is:
        # Vector size: 10 * 64 * 4 = 2560 bytes.
        # Payload size is approximately 10 * 45 = 450 bytes.
        # Total cached segment size is about 3000 bytes.
        # We set max_memory_bytes to exactly 7000 bytes.
        # This will fit 2 segments (6000 bytes) but NOT 3 segments!
        manager = SegmentCacheManager(max_memory_bytes=7000)
        
        # Load segment 1 -> cache size = 1
        manager.get_vector(seg1, 0, dimension)
        assert seg1 in manager.cache
        
        # Load segment 2 -> cache size = 2
        manager.get_vector(seg2, 0, dimension)
        assert seg2 in manager.cache
        
        # Access segment 1 again to make it the most recently used (MRU)
        manager.get_vector(seg1, 0, dimension)
        
        # Load segment 3 -> exceeds 7000 bytes capacity!
        # Least recently used (LRU) was segment 2, so seg2 must be evicted,
        # while seg1 (accessed recently) and seg3 remain.
        manager.get_vector(seg3, 0, dimension)
        
        assert seg3 in manager.cache
        assert seg1 in manager.cache
        assert seg2 not in manager.cache # Evicted!
        
        assert manager.current_memory_bytes <= 7000

def test_cache_clear():
    manager = SegmentCacheManager()
    manager.hit_count = 10
    manager.miss_count = 5
    manager.current_memory_bytes = 40960
    manager.cache[uuid.uuid4()] = None
    manager.lru_order.append(uuid.uuid4())
    
    manager.clear()
    
    assert manager.hit_count == 0
    assert manager.miss_count == 0
    assert manager.current_memory_bytes == 0
    assert len(manager.cache) == 0
    assert len(manager.lru_order) == 0
