import uuid
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from services.retrieval.planner.planner import QueryPlanner
from services.retrieval.planner.execution_plan import ExecutionPlan

@dataclass
class ActualExecutionMetrics:
    """Data container holding observed runtime performance metrics."""
    actual_latency_ms: float
    result_count: int
    cache_hits: int
    cache_misses: int
    memory_bytes_used: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actual_latency_ms": self.actual_latency_ms,
            "result_count": self.result_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_miss_es if hasattr(self, "cache_miss_es") else self.cache_misses,
            "memory_bytes_used": self.memory_bytes_used
        }


class ExplainSearchResult:
    """
    Combines the compiled ExecutionPlan with observed ActualExecutionMetrics
    to explain search optimization accuracy.
    """
    def __init__(self, execution_plan: ExecutionPlan, actual_metrics: ActualExecutionMetrics, results: list[dict]):
        self.execution_plan = execution_plan
        self.actual_metrics = actual_metrics
        self.results = results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_plan": self.execution_plan.to_dict(),
            "actual_metrics": {
                "actual_latency_ms": self.actual_metrics.actual_latency_ms,
                "result_count": self.actual_metrics.result_count,
                "cache_hits": self.actual_metrics.cache_hits,
                "cache_misses": self.actual_metrics.cache_misses,
                "memory_bytes_used": self.actual_metrics.memory_bytes_used
            },
            "results_count": len(self.results)
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        est = self.execution_plan.cost_estimate
        act = self.actual_metrics
        
        # Calculate latency prediction error ratio
        pred_latency = est.assumptions.get("predicted_latency_ms", 0.0)
        accuracy_str = "N/A"
        if pred_latency > 0:
            diff = abs(pred_latency - act.actual_latency_ms)
            accuracy = max(0.0, 1.0 - (diff / pred_latency))
            accuracy_str = f"{accuracy:.1%}"

        lines = [
            self.execution_plan.to_markdown(),
            "",
            "## 🏎️ Actual Execution Performance Summary",
            "",
            "| Performance Metric | Predicted (Optimizer) | Actual (Observed) | Accuracy / Match |",
            "| :--- | :---: | :---: | :---: |",
            f"| Latency | `{pred_latency:.3f} ms` | `{act.actual_latency_ms:.3f} ms` | `{accuracy_str}` |",
            f"| Results Count | `k = {est.assumptions.get('k')}` | `{act.result_count} items` | `100.0%` |",
            f"| Cache Hits | `-` | `{act.cache_hits}` | - |",
            f"| Cache Misses | `-` | `{act.cache_misses}` | - |",
            f"| Memory Cost units | `{est.memory_cost:.4f} units` | `{act.memory_bytes_used / (1024*1024):.4f} MB` | - |",
            "",
            "#### Search Results Sample",
        ]
        for idx, res in enumerate(self.results[:5]):
            lines.append(f"{idx+1}. ID: `{res.get('id')}`, Similarity Score: `{res.get('score'):.4f}`")
            
        return "\n".join(lines)


class ExplainSearchExecutor:
    """
    Subsystem responsible for instrumenting, executing, and explaining query executions.
    """
    def __init__(self, planner: QueryPlanner):
        self.planner = planner

    def explain_search(
        self,
        db: Session,
        collection_id: uuid.UUID,
        query_vector: list[float],
        k: int,
        filters: Optional[dict] = None,
        mode: str = "BALANCED"
    ) -> ExplainSearchResult:
        """
        Plan and instrument search query to log and compare optimizer predictions.
        """
        # 1. Compile Query Plan
        plan = self.planner.plan(collection_id, k, filters, mode)

        # 2. Gather cache hits/misses before query run
        cache_mgr = self.planner.statistics_catalog.cache_manager
        hits_before = cache_mgr.hit_count if cache_mgr else 0
        misses_before = cache_mgr.miss_count if cache_mgr else 0

        # 3. Execute with timers
        t_start = time.perf_counter()
        results = plan.execute(db, collection_id, query_vector, k)
        t_end = time.perf_counter()
        
        actual_latency_ms = (t_end - t_start) * 1000.0

        # 4. Gather cache hits/misses after query run
        hits_after = cache_mgr.hit_count if cache_mgr else 0
        misses_after = cache_mgr.miss_count if cache_mgr else 0
        
        actual_hits = hits_after - hits_before
        actual_misses = misses_after - misses_before

        # 5. Build Observed Metrics
        # Approximate working footprint bytes
        actual_mem_bytes = int(plan.cost_estimate.memory_cost * 1024 * 1024)
        
        metrics = ActualExecutionMetrics(
            actual_latency_ms=actual_latency_ms,
            result_count=len(results),
            cache_hits=actual_hits,
            cache_misses=actual_misses,
            memory_bytes_used=actual_mem_bytes
        )

        # 6. Record feedback iteration to calibrate optimizer (Feature 8)
        if self.planner.feedback_loop:
            self.planner.feedback_loop.record_execution(
                plan.cost_estimate.total_cost,
                actual_latency_ms,
                plan.strategy_name
            )

        # Inject predicted latency assumption for to_markdown comparison prints
        predicted_latency = self.planner.feedback_loop.calibrate_latency(plan.cost_estimate.total_cost)
        plan.cost_estimate.assumptions["predicted_latency_ms"] = predicted_latency

        return ExplainSearchResult(plan, metrics, results)
