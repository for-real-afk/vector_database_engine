import logging
from services.retrieval.planner.cost.base_provider import CostProvider
from services.retrieval.planner.statistics import CollectionStatistics

logger = logging.getLogger(__name__)

class MemoryCostProvider(CostProvider):
    """
    Estimates relative memory footprint cost units.
    Separates resident data (loaded segments) from temporary search working memory.
    """
    # 1 Cost unit represents 1 Megabyte (MB) of RAM allocation
    BYTES_TO_UNIT_SCALE = 1.0 / (1024 * 1024)

    def __init__(self, memory_scale: float = 1.0):
        self.memory_scale = memory_scale

    def calculate_cost(
        self, 
        stats: CollectionStatistics, 
        k: int, 
        selectivity: float = 1.0, 
        mode: str = "BALANCED",
        strategy: str = "EXACT"
    ) -> float:
        """
        Estimate memory usage footprint and calculate total relative memory cost.
        """
        strategy_upper = strategy.upper()
        mode_upper = mode.upper()

        # Mode weights multiplier (Feature 11)
        mode_multiplier = 1.0
        if mode_upper == "LOWEST_MEMORY":
            mode_multiplier = 2.0  # Penalize memory-heavy paths to prefer disk operations
        elif mode_upper == "LOWEST_LATENCY":
            mode_multiplier = 0.5  # Ignore memory footprints to optimize speed

        # Size of single float vector
        vector_bytes = stats.dimension * 4
        # Size of payload
        payload_bytes = 120

        if strategy_upper == "EXACT":
            # Exact scans load vectors + payloads in memory
            active_size = max(1, int(stats.collection_size * selectivity))
            
            resident_bytes = active_size * (vector_bytes + payload_bytes)
            # Distance comparison score buffer (float32 array)
            temp_bytes = active_size * 4.0
            
            total_bytes = resident_bytes + temp_bytes
            return total_bytes * self.BYTES_TO_UNIT_SCALE * self.memory_scale * mode_multiplier

        elif strategy_upper == "HNSW":
            if stats.graph_nodes == 0:
                return float('inf')

            # HNSW holds the connection lists of node references in resident memory
            # For node: UUID (16B), Segment (16B), Level (2B), Offset (4B) = 38B
            # average connections M = average_degree (16 refs * 8B per pointer = 128B)
            node_bytes = 38 + (stats.average_degree * 8) + vector_bytes
            resident_bytes = stats.graph_nodes * node_bytes

            # Temporary traversal structures
            # 1. Visited node bitmap (1 bit per graph node)
            visited_bitmap_bytes = max(1.0, stats.graph_nodes / 8.0)
            
            # 2. Priority queues and candidate list (size ef_search)
            ef_search = 32.0
            queue_element_bytes = 24.0 # (float score + node UUID)
            temp_bytes = visited_bitmap_bytes + (ef_search * queue_element_bytes)

            total_bytes = resident_bytes + temp_bytes
            return total_bytes * self.BYTES_TO_UNIT_SCALE * self.memory_scale * mode_multiplier

        else:
            return 0.0
