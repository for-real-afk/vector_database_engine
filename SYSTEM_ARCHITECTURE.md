# SYSTEM_ARCHITECTURE.md: Production-Ready Vector Retrieval Engine

This document details the architectural design, storage layouts, transaction pipelines, indexing algorithms, memory model, and crash recovery mechanics for our custom **Vector Retrieval Engine**. 

---

## 1. System Topology & Layered Architecture

The system is designed with a strict separation of concerns, dividing execution into a **Control Plane** (handled via PostgreSQL for relational guarantees) and a **Data Plane** (handled via custom binary segments and memory-mapped HNSW graphs for extreme retrieval performance).

```
          +-------------------------------------------------+
          |                    API Layer                    |
          |       FastAPI / REST API / Authentication       |
          +-----------------------+-------------------------+
                                  |
                                  v
          +-----------------------+-------------------------+
          |                  Service Layer                  |
          |   Ingestion Pipe / Query Planner / Security     |
          +------------+-----------------------+------------+
                       |                       |
                       v                       v
          +------------+-----------+ +---------+------------+
          |    Retrieval Engine    | |    Control Plane     |
          |   (Query Execution)    | |  (PostgreSQL / ACID) |
          +------------+-----------+ +---------+------------+
                       |                       |
                       v                       v
          +------------+-----------+ +---------+------------+
          |       ANN Index        | |  Relational Storage  |
          |    (Custom HNSW)       | | Documents, Chunks,   |
          +------------+-----------+ | Metadata, Audit logs |
                       |             +----------------------+
                       v
          +------------+-----------+
          |  Persistence Layer     |
          | (Custom Binary Engine) |
          |   Segments & WAL       |
          +------------------------+
```

### Layer Rationale & Comparison

| Component / Goal | Our Approach | PostgreSQL / pgvector | Elasticsearch | Qdrant / Milvus |
| :--- | :--- | :--- | :--- | :--- |
| **Storage Architecture** | Hybrid (PostgreSQL for metadata, custom binary segments for vectors/payloads). | Everything in relational tables (heap files & TOAST). | Lucene immutable segments (inverted index + doc values). | Custom segment-based storage with RocksDB/raft metadata. |
| **Index Traversal** | HNSW graph loaded in-memory or memory-mapped (`mmap`) using flat binary structures. | Relational indexes (B-Tree/GIN) or pgvector's ivfflat/hnsw inside shared buffers. | HNSW search implemented over Lucene segments. | In-memory graphs pointing to segments. |
| **ACID Guarantees** | Relational data has full ACID. Vector/Segment updates are eventually consistent via WAL. | Full ACID on every insert/update via standard WAL. | Near-real-time (NRT) refresh, transactional guarantees at document-level only. | WAL-backed, eventually consistent index state. |

---

## 2. Storage Engine Design & Binary Layout

To minimize disk I/O, avoid garbage collection overhead, and enable SIMD instruction optimization, the retrieval engine uses a **Segment-Based Storage Model**.

### 2.1 The Hybrid Model
*   **PostgreSQL**: Manages namespaces, collections, document metadata, audit logs, and users. This guarantees relational constraint checking (e.g., verifying a collection exists, enforcing tenant ownership) before writing vector data.
*   **Custom Binary Segments (`storage/segments/`)**: Holds the raw vectors, local segment-level payload offsets, and deleted record lists.
*   **HNSW Graph File (`storage/snapshots/`)**: Holds the HNSW adjacency list.

### 2.2 Segment Architecture: Growing vs. Sealed
All vectors are written to an active **Growing Segment**.
1.  **Growing Segment**: A append-only binary file coupled with an in-memory lock-free skip-list or vector array.
2.  **Sealed Segment**: Once the growing segment reaches a configured threshold (e.g., 50,000 vectors), it is sealed (rendered immutable).
3.  **Compaction & Merging**: A background worker periodically merges multiple small sealed segments into a single larger sealed segment, purging deleted records.

### 2.3 Segment Binary File Layout (`segment.bin`)

To allow memory mapping (`mmap`), the binary structure uses explicit byte alignments:

```
+---------------------------------------------------------------------------------+
|                                 SEGMENT HEADER                                  |
| Magic (4B) | Version (2B) | Segment ID (16B) | Record Count (4B) | Deleted (4B) |
| Align (4B) | Created TS (8B) | Checksum CRC64 (8B)                              |
+---------------------------------------------------------------------------------+
|                                RECORD INDEX TABLE                               |
| For each record:                                                                |
| Record ID (16B UUID) | Vector Offset (8B) | Payload Offset (8B) | Status (1B)  |
+---------------------------------------------------------------------------------+
|                               VECTOR DATA BLOCK                                 |
| Contiguous, aligned to 64-byte boundaries (SIMD vector operations friendly)      |
| [ Float32 x Dimensions ] [ Padding to 64B ] ...                                 |
+---------------------------------------------------------------------------------+
|                                PAYLOAD BLOCK                                    |
| For each record:                                                                |
| Length (4B) | Serialized JSON Payload (UTF-8, Variable Length)                  |
+---------------------------------------------------------------------------------+
```

*   **Memory Alignment**: The Vector Data Block starts at an offset divisible by 64 bytes. This allows compilers to generate AVX-512 or ARM NEON instructions without alignment faults.
*   **Status Byte**: `0x01` = Active, `0x02` = Tombstone (Deleted). During search, vectors flagged as `0x02` are skipped.
*   **Checksums**: CRC64 is computed over the entire segment contents (excluding the checksum field itself) to detect silent bit rot.

### 2.4 Comparison on Storage
*   **PostgreSQL**: Uses fixed 8KB pages. Storing large float arrays inside pages leads to heavy page fragmentation and frequent TOAST table lookups.
*   **Qdrant**: Employs memory-mapped segments with a separate storage directory per collection. We follow this model, mapping each collection to a subdirectory within `storage/segments/`.

---

## 3. Query Execution & Write Pipelines

### 3.1 Write Pipeline (Ingestion Path)
The write pipeline guarantees durability while maintaining high throughput.

```
[Document Ingestion API Request]
              │
              ▼
[Document Cleaner & Chunker] (Recursive/Semantic/Fixed Chunks)
              │
              ▼
[Embedding Generation Provider] (Gemini, OpenAI, HuggingFace wrapper)
              │
              ▼
[Write-Ahead Log (WAL)] (Append-only write to storage/wal/wal.log + fsync)
              │
              ▼
┌─────────────┴─────────────┐
│                           │
▼                           ▼
[PostgreSQL Control Plane]  [Active Growing Segment]
- Insert Document Record    - Append Vector data to memory buffer
- Insert Chunk Metadata     - Write raw vector to growing segment file
- Commit ACID Transaction   - Update active segment index table
                            │
                            ▼
                    [Segment Full?]
                            ├── Yes ──> Seal Segment ──> Trigger Background HNSW Build
                            └── No  ──> Complete Write
```

### 3.2 Read Pipeline (Query Path)

```
[Search Query API Request]
            │
            ▼
[Authentication & Policy Enforcement] (Validate API token & Namespace/Tenant scopes)
            │
            ▼
[Query Text] ──> [Embedding Provider] ──> [Query Vector]
                                                │
                                                ▼
                                        [Query Planner]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
       [Metadata Pre-Filtering]                                       [Index Selection]
       - Query Postgres for match IDs                                 - Locate active HNSW Graph
       - Produce dynamic ID Allowed-List                              - Compute entry node
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                                                ▼
                                    [HNSW Graph Search]
                                    - Traversal constrained by Allowed-List
                                    - Evaluate distance via SIMD calculations
                                                │
                                                ▼
                                    [Candidate Collector]
                                    - Gather Top-K nearest vector IDs
                                                │
                                                ▼
                                    [Payload & Document Hydration]
                                    - Fetch raw text/metadata from Segment payload block
                                    - Fallback to PostgreSQL if cache miss
                                                │
                                                ▼
                                    [Post-Filtering & Scoring]
                                    - Score fusion (Keyword BM25 + Semantic Cosine)
                                    - Reciprocal Rank Fusion (RRF)
                                                │
                                                ▼
                                     [Response Formatted]
```

### 3.3 Comparison on Pre vs Post Filtering
*   **PostgreSQL (pgvector)**: Filters metadata using standard SQL where-clauses. If HNSW is used, it may struggle with pre-filtering because the index cannot easily be restricted dynamically during traversal, leading to the "single-point routing" problem where the graph search hits dead-ends.
*   **Milvus/Qdrant**: Solve this by performing **Single-Stage Hybrid Filtering** (checking metadata matches directly during HNSW graph node traversal). We implement this via an `Allowed-List` bitmap generated by metadata parameters, checked at each step of the HNSW search.

---

## 4. Custom HNSW Index Design & Serialization

We implement a custom Hierarchical Navigable Small World (HNSW) index based on the classic Malkov & Yashunin (2016) paper, but adapted for relational mapping.

### 4.1 Graph Architecture
Instead of storing raw embedding vectors directly inside the HNSW nodes, each node stores:
1.  **Segment ID (16B UUID)**: Points to the target storage segment.
2.  **Vector Offset / Index (4B)**: Locates the vector inside the segment file.
3.  **Adjacency List**: Arrays of pointers referencing other node structures across layers.

This design prevents vector duplication, keeping the graph footprint small enough to reside completely in RAM.

### 4.2 HNSW Graph Node Memory Structure
```python
class HNSWNode:
    segment_id: UUID       # 16 bytes
    vector_idx: int        # 4 bytes (offset in segment record index)
    # Adjacency list per layer: layer_id -> list of node references
    neighbors: dict[int, list[HNSWNode]] 
```

### 4.3 Graph Serialization Format (`graph.bin`)
To persist the graph without rebuilding it on startup:

```
+---------------------------------------------------------------------------------+
|                                 GRAPH HEADER                                    |
| Magic (4B) | Version (2B) | Segment ID (16B) | Max Layer (2B) | Entry Point (8B) |
+---------------------------------------------------------------------------------+
|                                   NODE INDEX                                    |
| For each node in the graph:                                                     |
| Node ID (8B) | Segment ID (16B) | Vector Index (4B) | Layer Count (2B)          |
|   For each layer:                                                               |
|     Neighbor Count (2B) | Neighbor Node IDs (Variable: 8B x Count)              |
+---------------------------------------------------------------------------------+
```

During startup, the engine performs a **zero-copy memory map (`mmap`)** of `graph.bin`, referencing nodes via offset-based indices, avoiding deserialization delays.

---

## 5. Memory Model & Caching

To scale execution speeds, a multi-tier memory hierarchy is maintained:

```
+------------------------------------------------------------+
|                        RAM (Engine)                        |
|                                                            |
|  +--------------------+  +------------------------------+  |
|  |    Active Graphs   |  |        Segment Cache         |  |
|  | HNSW Nodes & Links |  |   (mmap index & records)     |  |
|  +--------------------+  +------------------------------+  |
|                                                            |
|  +--------------------+  +------------------------------+  |
|  | Vector Cache (LRU) |  |    Payload / Chunk Cache     |  |
|  | Contiguous floats  |  |    Frequently accessed text  |  |
|  +--------------------+  +------------------------------+  |
+-----------------------------+------------------------------+
                              |
                     Buffer Pool Manager
                              |
                              v
+-----------------------------+------------------------------+
|                     Disk (Persistence)                     |
|                                                            |
|  +--------------------+  +------------------------------+  |
|  |   wal/wal.log      |  |      segments/*.bin          |  |
|  +--------------------+  +------------------------------+  |
|                                                            |
|  +--------------------+  +------------------------------+  |
|  |   PostgreSQL DB    |  |       snapshots/*.bin        |  |
|  +--------------------+  +------------------------------+  |
+------------------------------------------------------------+
```

### Caching Layers Design
1.  **HNSW Graph RAM Mapping**: The active graph indexes (`graph.bin`) are memory-mapped (`mmap`) or loaded directly into memory.
2.  **Vector Cache**: Keeps raw float arrays of hot segments in L1/L2 cache-friendly structures to speed up distance computations during brute-force fallbacks or HNSW exploration.
3.  **Buffer Pool Manager**: Implements an LRU page replacement algorithm. If raw vectors need verification (e.g. during a graph search verify step), the buffer pool reads the segment page without executing a PostgreSQL query.
4.  **No-SQL Retrieval Rule**: Retrieval requests (search queries) *must not* execute queries against PostgreSQL on the critical search path unless retrieving the final, raw non-cached payload metadata. All index traversal, candidate generation, and vector distance scoring must be executed against the cache/mmap layer.

---

## 6. Concurrency Control & Multi-Reader Single-Writer

To prevent search threads from blocking write threads:

### 6.1 Multi-Reader Single-Writer (MRSW) Lock
*   **Search Operations**: Acquire a Shared Lock (`SharedLock`) on the HNSW segment. Multiple threads can traverse the HNSW graph simultaneously.
*   **Write Operations**: Acquire an Exclusive Lock (`ExclusiveLock`) on the active **Growing Segment** append list. Since writes only append to the segment and do not modify historical graphs in-place immediately, reads continue unaffected on sealed indexes.

### 6.2 Lock-Free Graph Rebuilding & Double Buffering
When a background compaction finishes or a sealed segment graph is rebuilt:
1.  The worker generates a new HNSW graph in the background (`graph.bin.tmp`).
2.  Once finished, the engine performs a **Pointer Swap** (atomic swap) to point the search execution engine to the new graph structure.
3.  The old graph structure is marked for garbage collection and freed once all active search threads referencing it exit. This resembles the **Read-Copy-Update (RCU)** synchronization pattern used in operating system kernels.

---

## 7. Failure Recovery & Write-Ahead Log (WAL)

To prevent data loss and corruption during abrupt system power failures:

### 7.1 WAL Record Structure
Every change written to the database engine first appends a record to the WAL file:
*   `LSN` (Log Sequence Number) - 8 bytes
*   `Operation Type` (Insert, Delete) - 1 byte
*   `UUID Record ID` - 16 bytes
*   `Collection ID` - 16 bytes
*   `Vector Dimensions` - 2 bytes
*   `Vector Data` - Variable (`Dimensions` * 4 bytes)
*   `Payload Bytes` - Variable (JSON)
*   `CRC32` - 4 bytes

### 7.2 The Recovery Protocol
At startup, the engine runs the following verification check:
```
                       [Engine Boot Sequence]
                                 │
                                 ▼
                     [Open active wal/wal.log]
                                 │
                                 ▼
                    [Verify Checksums of Records]
                                 │
                                 ▼
              [Read segment.bin & graph.bin manifests]
                                 │
                                 ▼
           [Identify last committed LSN in Segment Header]
                                 │
                                 ▼
          [Are there WAL records with LSN > Segment LSN?]
                                 ├── Yes ──> Replay WAL Records:
                                 │           - Append vectors to segment
                                 │           - Append relational records to PG
                                 │           - Mark Graph for rebuild
                                 │
                                 └── No  ──> System Ready (Online)
```

---

## 8. Pluggable Architecture

The system utilizes strict interfaces using Python typing protocols or abstract base classes (`abc`). No implementation detail is coupled directly.

```
       +--------------------+
       | EmbeddingProvider  | <--- Interface
       +---------+----------+
                 |
        +--------+--------+
        |                 |
  +-----v------+    +-----v------+
  | OpenAIProv |    | GeminiProv | ...
  +------------+    +------------+

       +--------------------+
       |   DistanceMetric   | <--- Interface
       +---------+----------+
                 |
        +--------+--------+
        |                 |
  +-----v------+    +-----v------+
  | L2Distance |    | CosineDist | ...
  +------------+    +------------+
```

### Pluggable Interfaces
1.  **`EmbeddingProvider`**:
    *   Methods: `embed_text(text: str) -> list[float]`, `embed_batch(texts: list[str]) -> list[float]`
2.  **`ChunkerProvider`**:
    *   Methods: `split_document(doc: str, config: ChunkConfig) -> list[Chunk]`
3.  **`DistanceMetric`**:
    *   Methods: `calculate(v1: ndarray, v2: ndarray) -> float`
4.  **`StorageEngine`**:
    *   Methods: `write_vector(id: UUID, vector: ndarray, payload: dict)`, `read_vector(id: UUID) -> ndarray`

---

## 9. Benchmarking & Observability

### 9.1 Observability
We hook standard metric instrumentation points into the search query loop:
*   `search_latency_seconds`: Histogram measuring query routing to response time.
*   `hnsw_visited_nodes_count`: Counter tracking graph node hops (indicates graph search quality and index efficiency).
*   `cache_hits_total` (Labels: `type=vector`, `type=payload`): Tracking cache efficiency.
*   `index_recall_rate`: Computed periodically against a brute force exact search to measure accuracy.

### 9.2 The Benchmark Framework (`benchmarks/`)
We build an internal benchmarking harness containing:
*   `ingestion.py`: Spawns multiple threads inserting vectors in batches, outputting inserts per second.
*   `search.py`: Measures QPS (queries per second) at different `K` limits and different HNSW parameters (e.g. `ef_search`).
*   `recall.py`: Runs a golden dataset (e.g., SIFT10K or synthetic) on both the custom HNSW index and the exact numpy O(N) index, calculating standard Recall@K.

---

## 10. AWS Production-Ready Mapping

To ensure easy cloud migration, local service boundaries directly map to AWS services:

| Local Component | Local Mechanism | AWS Production Target | Migration Strategy |
| :--- | :--- | :--- | :--- |
| **Control Plane DB** | Local PostgreSQL Container | Amazon RDS PostgreSQL | Configure SQLAlchemy connection string. Migrate tables via Alembic. |
| **Blob / Segment Files** | local disk: `storage/segments` | Amazon S3 | Swap local `StorageProvider` with an S3-backed `StorageProvider` wrapper. |
| **HNSW Cache / WAL** | Local Disk / Memory | Amazon ElastiCache Redis / S3 | WAL logs copy to Amazon S3 for durability. Cached indexes stored on local NVMe instances. |
| **Background Workers** | In-Process Task Queue / Celery | Amazon ECS / AWS Batch + SQS | Package workers as ECS Tasks listening to SQS Queues. |
| **API Load Balancing** | Nginx | AWS Application Load Balancer (ALB) | Route API Gateway or ALB directly to ECS Fargate tasks. |
