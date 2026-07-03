import os
import logging
logger = logging.getLogger(__name__)
import json
import struct
import uuid
import numpy as np
from typing import Tuple, Callable
from index.graph.hnsw import HNSWIndex, HNSWNode
from core.config import settings

class HNSWSerializer:
    """
    Handles binary serialization and deserialization of the HNSW Graph Index (graph.bin).
    Format references PostgreSQL record UUIDs and Segment file coordinates
    instead of duplicating raw float embedding vectors on disk.
    """

    HEADER_FORMAT = "<4s H H 16s I 36x" # 64 Bytes
    # 4s  - Magic Number (b'VHNS')
    # H   - Version (1)
    # H   - Max Level (uint16)
    # 16s - Entry Node UUID bytes
    # I   - Node Count (uint32)
    # 36x - 36 Padding bytes

    NODE_BASE_FORMAT = "<16s 16s I H" # 38 Bytes
    # 16s - Node UUID bytes
    # 16s - Segment UUID bytes
    # I   - Vector Index (uint32)
    # H   - Level (uint16)

    @classmethod
    def serialize(cls, index: HNSWIndex, segment_mappings: dict[uuid.UUID, Tuple[uuid.UUID, int]]) -> bytes:
        """
        Convert HNSW graph index memory structure to binary bytes.
        - segment_mappings: dict mapping node UUID -> (segment_id, vector_idx)
        """
        node_count = len(index.nodes)
        max_level = index.max_level
        
        entry_node_bytes = b'\x00' * 16
        if index.enter_node is not None:
            entry_node_bytes = index.enter_node.id.bytes
            
        header = struct.pack(
            cls.HEADER_FORMAT,
            b'VHNS',
            1,
            max_level,
            entry_node_bytes,
            node_count
        )

        body = b''
        for node_id, node in index.nodes.items():
            # Get segment coordinate mappings
            # If not provided, fallback to default zero UUIDs
            seg_id, vec_idx = segment_mappings.get(node_id, (uuid.UUID(int=0), 0))
            
            # Pack node base metadata (38 bytes)
            node_base = struct.pack(
                cls.NODE_BASE_FORMAT,
                node_id.bytes,
                seg_id.bytes,
                vec_idx,
                node.level
            )
            
            # Pack neighbor counts per level
            # For level L, there are L + 1 counts (levels 0 to L)
            neighbor_counts = b''
            neighbor_ids = b''
            
            for lc in range(node.level + 1):
                links = node.neighbors[lc]
                count = len(links)
                neighbor_counts += struct.pack("<H", count)
                for neighbor in links:
                    neighbor_ids += neighbor.id.bytes
                    
            body += node_base + neighbor_counts + neighbor_ids

        return header + body

    @classmethod
    def deserialize(
        cls, 
        data: bytes, 
        dimension: int,
        metric: str,
        vector_resolver: Callable[[uuid.UUID, uuid.UUID, int], list[float]]
    ) -> Tuple[HNSWIndex, dict[uuid.UUID, Tuple[uuid.UUID, int]]]:
        """
        Reconstruct HNSWIndex from binary bytes.
        - vector_resolver: A callable matching (node_id, segment_id, vector_idx) -> list[float]
          to populate node vectors from segment caches during restoration.
        """
        if len(data) < 64:
            raise ValueError("Invalid graph data: Less than header size (64 bytes).")

        header_data = data[:64]
        magic, version, max_level, entry_node_bytes, node_count = struct.unpack(
            cls.HEADER_FORMAT,
            header_data
        )

        if magic != b'VHNS':
            raise ValueError(f"Invalid magic: {magic}. Not a valid HNSW graph file.")

        # Re-initialize index
        index = HNSWIndex(dimension=dimension, metric=metric)
        index.max_level = max_level

        # Read nodes block sequentially
        offset = 64
        
        # Temporary storage to reconstruct linkages after creating all node objects
        linkage_map: dict[uuid.UUID, dict[int, list[uuid.UUID]]] = {}
        segment_mappings: dict[uuid.UUID, Tuple[uuid.UUID, int]] = {}

        for _ in range(node_count):
            if offset + 38 > len(data):
                raise ValueError("Unexpected End of File: Graph data truncated in node metadata.")
                
            node_base_data = data[offset:offset + 38]
            node_id_bytes, seg_id_bytes, vec_idx, level = struct.unpack(
                cls.NODE_BASE_FORMAT,
                node_base_data
            )
            offset += 38

            node_id = uuid.UUID(bytes=node_id_bytes)
            seg_id = uuid.UUID(bytes=seg_id_bytes)
            
            # Fetch node vector from segments using the resolver
            try:
                vector = vector_resolver(node_id, seg_id, vec_idx)
            except Exception as e:
                # If vector resolution fails, fallback to zero vector to prevent block crash
                logger.warning(f"Vector resolution failed for node {node_id} during restore: {e}")
                vector = [0.0] * dimension
                
            vec_arr = np.array(vector, dtype=np.float32)

            # Create node object
            node = HNSWNode(node_id, vec_arr, level)
            index.nodes[node_id] = node
            
            # Save segment mapping
            segment_mappings[node_id] = (seg_id, vec_idx)

            # Read neighbor counts (L + 1 counts)
            node_linkages = {}
            for lc in range(level + 1):
                if offset + 2 > len(data):
                    raise ValueError("Unexpected End of File: Graph data truncated in neighbor counts.")
                count, = struct.unpack("<H", data[offset:offset + 2])
                offset += 2
                
                # Read neighbor IDs
                neighbor_ids = []
                for _ in range(count):
                    if offset + 16 > len(data):
                        raise ValueError("Unexpected End of File: Graph data truncated in neighbor list.")
                    n_bytes = data[offset:offset + 16]
                    offset += 16
                    neighbor_ids.append(uuid.UUID(bytes=n_bytes))
                    
                node_linkages[lc] = neighbor_ids
                
            linkage_map[node_id] = node_linkages

        # Reconstruct link pointer relationships
        for node_id, linkages in linkage_map.items():
            node = index.nodes[node_id]
            for lc, neighbor_ids in linkages.items():
                for nid in neighbor_ids:
                    if nid in index.nodes:
                        node.neighbors[lc].append(index.nodes[nid])

        # Set entry point
        entry_uuid = uuid.UUID(bytes=entry_node_bytes)
        if entry_uuid in index.nodes:
            index.enter_node = index.nodes[entry_uuid]

        return index, segment_mappings


class HNSWIndexManager:
    """
    Coordinates persistence files writing/reading (graph.bin + metadata.json).
    """
    @classmethod
    def snapshot(
        cls, 
        directory: str, 
        index: HNSWIndex, 
        segment_mappings: dict[uuid.UUID, Tuple[uuid.UUID, int]]
    ):
        """Write graph and metadata files to snapshot folder."""
        os.makedirs(directory, exist_ok=True)
        
        # 1. Write graph.bin
        graph_bytes = HNSWSerializer.serialize(index, segment_mappings)
        graph_path = os.path.join(directory, "graph.bin")
        with open(graph_path, "wb") as f:
            f.write(graph_bytes)
            
        # 2. Write metadata.json
        meta_dict = {
            "dimension": index.dimension,
            "metric": index.metric,
            "M": index.M,
            "M0": index.M0,
            "ef_construction": index.ef_construction,
            "ef_search": index.ef_search,
            "tombstones": [str(t) for t in index.tombstones]
        }
        
        meta_path = os.path.join(directory, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2)

        logger.info(f"Graph snapshot successfully written to {directory}")

    @classmethod
    def restore(
        cls, 
        directory: str,
        vector_resolver: Callable[[uuid.UUID, uuid.UUID, int], list[float]]
    ) -> Tuple[HNSWIndex, dict[uuid.UUID, Tuple[uuid.UUID, int]]]:
        """Restore HNSW index and coordinate mappings from files."""
        graph_path = os.path.join(directory, "graph.bin")
        meta_path = os.path.join(directory, "metadata.json")
        
        if not os.path.exists(graph_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Snapshot files not found inside {directory}")
            
        # 1. Read metadata
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
            
        # 2. Read graph binary
        with open(graph_path, "rb") as f:
            graph_bytes = f.read()
            
        dimension = meta_dict["dimension"]
        metric = meta_dict["metric"]
        
        # 3. Deserialize graph structure
        index, segment_mappings = HNSWSerializer.deserialize(
            graph_bytes, 
            dimension, 
            metric, 
            vector_resolver
        )
        
        # Configure matching HNSW params
        index.M = meta_dict.get("M", 16)
        index.M0 = meta_dict.get("M0", 32)
        index.ef_construction = meta_dict.get("ef_construction", 64)
        index.ef_search = meta_dict.get("ef_search", 32)
        
        # Restore soft delete list
        tombstones_strs = meta_dict.get("tombstones", [])
        index.tombstones = {uuid.UUID(t) for t in tombstones_strs}
        
        return index, segment_mappings
