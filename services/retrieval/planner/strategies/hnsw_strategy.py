import uuid
from typing import Optional
from sqlalchemy.orm import Session

from services.retrieval.planner.strategy_registry import RetrievalStrategy
from services.retrieval.hybrid_search import HybridSearchCoordinator

class HNSWRetrievalStrategy(RetrievalStrategy):
    """
    Retrieval strategy wrapping custom multi-layer HNSW graph search.
    """
    def __init__(self, embedding_provider = None, cache_manager = None):
        self.embedding_provider = embedding_provider
        self.cache_manager = cache_manager

    def execute(
        self, 
        db: Session, 
        collection_id: uuid.UUID, 
        query_vector: list[float], 
        k: int, 
        allowed_ids: Optional[set[uuid.UUID]] = None
    ) -> list[dict]:
        # Build coordinator
        # Note: If embedding_provider is missing, we use a dummy since we pass query_vector directly.
        coordinator = HybridSearchCoordinator(db, self.embedding_provider, self.cache_manager)
        
        # In HNSWRetrievalStrategy, the planner might have resolved filters already to allowed_ids.
        # But wait! HybridSearchCoordinator.search takes 'filters: dict' and resolves it.
        # Let's inspect HybridSearchCoordinator.search:
        # If we want to strictly apply allowed_ids, we can either pass 'filters' or, if the coordinator allows it,
        # wait! HybridSearchCoordinator does HNSW searches:
        # results = hnsw_idx.search(query_vector, k=k*2, ef=64, allowed_ids=allowed_ids)
        # But wait, does HybridSearchCoordinator.search accept filters?
        # Yes, it resolves filters internally and runs HNSW index searches using allowed_ids!
        # If we pass filters, it works. But what if we already resolved allowed_ids and want to run it?
        # Let's see: `HybridSearchCoordinator.search` itself resolves filters.
        # So we can pass `filters=None` and pass allowed_ids inside execution, but since `search` doesn't take allowed_ids,
        # we can temporarily store resolved allowed_ids, or resolve them again inside the coordinator.
        # Resolving filters is fast, so we can simply pass `filters` to `search(...)` and let it resolve it inside!
        # Wait, what if we pass query_vector directly?
        results = coordinator.search(
            collection_id=collection_id,
            query_text="",
            query_vector=query_vector,
            filters=None, # will be resolved in execution or passed down
            k=k,
            alpha=1.0  # pure semantic HNSW search
        )
        
        # If allowed_ids is passed, filter in-memory as final post-filter safeguard
        if allowed_ids is not None:
            results = [r for r in results if r["id"] in allowed_ids]
            
        return results

    def estimate_cost(self, stats, k: int, filters: Optional[dict] = None) -> dict:
        return {}
