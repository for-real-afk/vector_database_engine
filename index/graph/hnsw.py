import bisect
import logging
import uuid
import numpy as np
from typing import Tuple, Set, Optional

logger = logging.getLogger(__name__)

class HNSWNode:
    """
    Represents a single node inside the HNSW graph index.
    Does not copy raw payload text; refers to record UUID and vector coordinates.
    """
    def __init__(self, node_id: uuid.UUID, vector: np.ndarray, level: int):
        self.id = node_id
        self.vector = vector
        self.level = level
        # Adjacency list: layer_id -> list of HNSWNode objects
        self.neighbors: dict[int, list[HNSWNode]] = {i: [] for i in range(level + 1)}

    def __repr__(self):
        return f"HNSWNode(id={self.id}, level={self.level})"


class CandidateList:
    """
    Ordered priority queue helper for tracking search candidates.
    Maintains sorted order by distance (ascending).
    """
    def __init__(self):
        # List of tuples: (distance, HNSWNode)
        self.data: list[Tuple[float, HNSWNode]] = []

    def add(self, distance: float, node: HNSWNode):
        # Ensure we don't insert duplicate nodes in the search queue
        for d, n in self.data:
            if n.id == node.id:
                return
        bisect.insort(self.data, (distance, node), key=lambda x: x[0])

    def pop_closest(self) -> Tuple[float, HNSWNode]:
        return self.data.pop(0)

    def pop_furthest(self) -> Tuple[float, HNSWNode]:
        return self.data.pop()

    def furthest_distance(self) -> float:
        return self.data[-1][0] if self.data else float('inf')

    def closest_distance(self) -> float:
        return self.data[0][0] if self.data else float('inf')

    def __len__(self) -> int:
        return len(self.data)


class HNSWIndex:
    """
    Custom Hierarchical Navigable Small World (HNSW) Graph Index.
    Implements greedy search, layer navigation, bidirectional insertions, 
    heuristic neighbor diversification, and soft-delete tombstones.
    """
    def __init__(
        self, 
        dimension: int,
        metric: str = "Cosine",
        M: int = 16, 
        M0: int = 32, 
        ef_construction: int = 64, 
        ef_search: int = 32
    ):
        self.dimension = dimension
        self.metric = metric.upper()
        self.M = M
        self.M0 = M0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Level generation normalization factor
        self.mL = 1.0 / np.log(M)
        
        # Core Index structures
        self.nodes: dict[uuid.UUID, HNSWNode] = {}
        self.enter_node: Optional[HNSWNode] = None
        self.max_level = -1
        
        # Soft delete tombstone list (contains UUIDs of deleted records)
        self.tombstones: Set[uuid.UUID] = set()
        
        # Setup random generator for level allocations
        self.rng = np.random.default_rng(42) # fixed seed for deterministic indexing tests

    def _calculate_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calculate distance between two vectors based on configured metric."""
        if self.metric == "COSINE":
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            if norm_u == 0 or norm_v == 0:
                return 1.0
            return float(1.0 - (np.dot(u, v) / (norm_u * norm_v)))
        elif self.metric in ("L2", "EUCLIDEAN"):
            return float(np.linalg.norm(u - v))
        elif self.metric == "DOTPRODUCT":
            # For HNSW search, smaller distance is closer.
            # Convert dot product score to distance: e.g. negative dot product
            return float(-np.dot(u, v))
        elif self.metric == "MANHATTAN":
            return float(np.sum(np.abs(u - v)))
        else:
            # Fallback to Cosine Distance
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            if norm_u == 0 or norm_v == 0:
                return 1.0
            return float(1.0 - (np.dot(u, v) / (norm_u * norm_v)))

    def insert(self, node_id: uuid.UUID, vector: list[float]):
        """
        Insert a new node into the HNSW graph.
        """
        vec = np.array(vector, dtype=np.float32)
        if len(vec) != self.dimension:
            raise ValueError(f"Vector dimension {len(vec)} does not match index dimension {self.dimension}")

        # If node was in tombstones, remove it (re-inserted)
        self.tombstones.discard(node_id)

        # 1. Determine maximum level for this node
        # Draw from an exponential decay distribution
        level = int(np.floor(-np.log(self.rng.uniform(0.1, 1.0)) * self.mL))
        
        new_node = HNSWNode(node_id, vec, level)
        self.nodes[node_id] = new_node

        # If index is empty, initialize entry point
        if self.enter_node is None:
            self.enter_node = new_node
            self.max_level = level
            return

        curr_node = self.enter_node
        curr_dist = self._calculate_distance(vec, curr_node.vector)

        # 2. Greedy Search from top layer down to level + 1
        for lc in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in curr_node.neighbors.get(lc, []):
                    d = self._calculate_distance(vec, neighbor.vector)
                    if d < curr_dist:
                        curr_dist = d
                        curr_node = neighbor
                        changed = True

        # 3. Add node to levels from level down to 0
        # Tracks nearest candidates to feed enter point of next layer
        enter_points = [curr_node]
        
        for lc in range(min(level, self.max_level), -1, -1):
            # Find closest ef_construction candidates on this layer
            candidates = self._search_layer(vec, enter_points, self.ef_construction, lc)
            
            # Select neighbors using the heuristic for diversity
            neighbors_to_link = self._select_neighbors_heuristic(vec, candidates, self.M, lc)
            
            # Add bidirectional connections
            for neighbor in neighbors_to_link:
                # Add link from new_node to neighbor
                new_node.neighbors[lc].append(neighbor)
                # Add link from neighbor to new_node
                neighbor.neighbors[lc].append(new_node)
                
                # Enforce link limit on the neighbor
                limit = self.M0 if lc == 0 else self.M
                if len(neighbor.neighbors[lc]) > limit:
                    # Re-evaluate and shrink neighbor connections
                    shrunk_neighbors = self._select_neighbors_heuristic(
                        neighbor.vector, 
                        [(self._calculate_distance(neighbor.vector, n.vector), n) for n in neighbor.neighbors[lc]], 
                        limit,
                        lc
                    )
                    neighbor.neighbors[lc] = shrunk_neighbors

            # Setup candidates as entry points for the next lower layer
            enter_points = [c[1] for c in candidates.data]

        # 4. If new node's level exceeds max_level, update index entry point
        if level > self.max_level:
            self.enter_node = new_node
            self.max_level = level

    def search(
        self, 
        query_vector: list[float], 
        k: int = 5, 
        ef: int = None,
        allowed_ids: Set[uuid.UUID] = None
    ) -> list[Tuple[float, uuid.UUID]]:
        """
        Search the HNSW index for the Top-K nearest neighbors.
        Returns a list of tuples: (distance, UUID)
        """
        if self.enter_node is None:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        curr_node = self.enter_node
        curr_dist = self._calculate_distance(q_vec, curr_node.vector)
        ef_val = ef or self.ef_search

        # 1. Greedy search through upper layers
        for lc in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in curr_node.neighbors.get(lc, []):
                    d = self._calculate_distance(q_vec, neighbor.vector)
                    if d < curr_dist:
                        curr_dist = d
                        curr_node = neighbor
                        changed = True

        # 2. Search Layer 0 using Candidate List
        candidates = self._search_layer(q_vec, [curr_node], ef_val, 0, allowed_ids)
        
        # 3. Filter candidates for tombstones and retrieve Top-K
        results = []
        for dist, node in candidates.data:
            if node.id in self.tombstones:
                continue
            
            # Convert internal DOTPRODUCT distance back to score
            score = -dist if self.metric == "DOTPRODUCT" else dist
            results.append((score, node.id))
            if len(results) >= k:
                break
                
        return results

    def mark_deleted(self, node_id: uuid.UUID):
        """Soft delete a node: flag it in the tombstones list."""
        if node_id in self.nodes:
            self.tombstones.add(node_id)

    def _search_layer(
        self, 
        q_vec: np.ndarray, 
        enter_nodes: list[HNSWNode], 
        ef: int, 
        level_idx: int,
        allowed_ids: Set[uuid.UUID] = None
    ) -> CandidateList:
        """
        Search a single layer of HNSW index.
        """
        visited: Set[uuid.UUID] = set()
        v_queue = CandidateList() # candidates to evaluate (popped from closest)
        w_queue = CandidateList() # best results found so far (popped from furthest if len > ef)

        for node in enter_nodes:
            visited.add(node.id)
            d = self._calculate_distance(q_vec, node.vector)
            v_queue.add(d, node)
            if allowed_ids is None or node.id in allowed_ids:
                w_queue.add(d, node)

        while len(v_queue) > 0:
            curr_dist, curr_node = v_queue.pop_closest()
            
            # Stop condition: if closest candidate is further than furthest best result
            if curr_dist > w_queue.furthest_distance():
                break

            for neighbor in curr_node.neighbors.get(level_idx, []):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    d = self._calculate_distance(q_vec, neighbor.vector)
                    
                    if d < w_queue.furthest_distance() or len(w_queue) < ef:
                        v_queue.add(d, neighbor)
                        if allowed_ids is None or neighbor.id in allowed_ids:
                            w_queue.add(d, neighbor)
                            
                            # Shrink best results queue to maintain size limit
                            if len(w_queue) > ef:
                                w_queue.pop_furthest()

        return w_queue

    def _select_neighbors_heuristic(
        self, 
        base_vec: np.ndarray, 
        candidates, 
        M: int, 
        level_idx: int
    ) -> list[HNSWNode]:
        """
        Heuristic neighbor selection for pruning and connectivity diversification.
        Ensures neighbors are distributed in different directions rather than clustered.
        """
        result: list[HNSWNode] = []
        
        # Resolve candidates to sorted list of (distance, HNSWNode)
        if isinstance(candidates, CandidateList):
            cand_list = candidates.data
        else:
            # Assumes direct list of (distance, HNSWNode)
            cand_list = sorted(candidates, key=lambda x: x[0])

        for dist, candidate_node in cand_list:
            if len(result) >= M:
                break
                
            keep = True
            # Check if candidate is closer to any node in the result than to the base node
            for selected_node in result:
                d_between_selected = self._calculate_distance(candidate_node.vector, selected_node.vector)
                if d_between_selected < dist:
                    keep = False
                    break
            
            if keep:
                result.append(candidate_node)

        # Fallback if result is too small: append remaining candidates to reach M
        if len(result) < M:
            for dist, candidate_node in cand_list:
                if len(result) >= M:
                    break
                if candidate_node not in result:
                    result.append(candidate_node)

        return result
