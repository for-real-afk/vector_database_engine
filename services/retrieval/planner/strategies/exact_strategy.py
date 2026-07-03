import uuid
from typing import Optional
from sqlalchemy.orm import Session

from services.retrieval.planner.strategy_registry import RetrievalStrategy
from services.retrieval.retriever import ExactRetriever

class ExactRetrievalStrategy(RetrievalStrategy):
    """
    Retrieval strategy wrapping O(N) linear exact search.
    """
    def execute(
        self, 
        db: Session, 
        collection_id: uuid.UUID, 
        query_vector: list[float], 
        k: int, 
        allowed_ids: Optional[set[uuid.UUID]] = None
    ) -> list[dict]:
        retriever = ExactRetriever(db)
        results = retriever.search(collection_id, query_vector, k=k)
        
        # Apply in-memory allowed IDs metadata filtering if resolved
        if allowed_ids is not None:
            results = [r for r in results if r["id"] in allowed_ids]
            
        return results

    def estimate_cost(self, stats, k: int, filters: Optional[dict] = None) -> dict:
        # Cost calculations are handled by the Cost Estimator module
        return {}
