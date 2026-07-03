import math
import logging
from services.retrieval.planner.cost.base_provider import CostProvider
from services.retrieval.planner.statistics import CollectionStatistics
from core.config import settings

logger = logging.getLogger(__name__)

class IOCostProvider(CostProvider):
    """
    Estimates relative Disk I/O cost units using a page-based read model.
    Models sequential block reads vs random seeks with cache hit mitigations.
    """
    PAGE_SIZE_BYTES = 4096  # Standard 4KB OS Disk Page size

    def __init__(
        self, 
        cost_seq_page_read: float = 1.0, 
        cost_rand_page_read: float = 4.0,
        cache_miss_penalty: float = 2.5
    ):
        self.cost_seq_page_read = cost_seq_page_read
        self.cost_rand_page_read = cost_rand_page_read
        self.cache_miss_penalty = cache_miss_penalty

    def calculate_cost(
        self, 
        stats: CollectionStatistics, 
        k: int, 
        selectivity: float = 1.0, 
        mode: str = "BALANCED",
        strategy: str = "EXACT"
    ) -> float:
        """
        Estimate page faults and calculate total relative Disk I/O cost.
        """
        strategy_upper = strategy.upper()
        mode_upper = mode.upper()

        # Mode weights multiplier
        mode_multiplier = 1.0
        if mode_upper == "LOWEST_IO":
            mode_multiplier = 2.0  # penalize IO higher to choose memory-resident paths
        elif mode_upper == "LOWEST_MEMORY":
            mode_multiplier = 0.5  # tolerate higher IO to save memory footprint

        # Calculate approximate segment file size in pages
        # Record entry (32B), vector floats (4B * dim), payload text (approx 120B)
        record_bytes = 32 + (stats.dimension * 4) + 120
        total_segment_bytes = 64 + (stats.collection_size * record_bytes)
        segment_pages = max(1, math.ceil(total_segment_bytes / self.PAGE_SIZE_BYTES))

        # Check cache hit/miss ratio (default to cache miss if stats missing)
        miss_ratio = stats.cache_miss_ratio
        if stats.collection_size > 0 and stats.cache_hit_ratio == 0.0 and stats.cache_miss_ratio == 0.0:
            miss_ratio = 1.0  # Default to full cache miss on cold start

        if strategy_upper == "EXACT":
            # Exact retrieval executes sequential scan over the entire segment
            pages_read = segment_pages * selectivity
            page_faults = pages_read * miss_ratio
            
            base_io = page_faults * self.cost_seq_page_read
            return base_io * mode_multiplier

        elif strategy_upper == "HNSW":
            if stats.graph_nodes == 0:
                return float('inf')

            # HNSW executes random seeks to fetch traversal nodes
            ef_search = 32.0
            expected_visited = (stats.graph_layers * stats.average_degree) + ef_search
            
            # Visited nodes are located at random positions in the segments file
            # Approximate random pages read (limited by total pages in the file)
            pages_accessed = min(float(segment_pages), expected_visited)
            page_faults = pages_accessed * miss_ratio

            # Random reads are weighted significantly higher than sequential reads
            base_io = page_faults * self.cost_rand_page_read * self.cache_miss_penalty
            return base_io * mode_multiplier

        else:
            return 0.0
