# Production-Grade Vector Retrieval Engine (VHNS & VSEG)

VHNS (Vector Hybrid Navigable Small World) is a production-grade, database-grade semantic retrieval engine built from scratch. It functions as a lightweight alternative to dedicated vector databases for small-to-medium workloads, utilizing PostgreSQL as the relational control plane and aligned binary files as the high-performance vector storage engine.

---

## 🏗️ Core Architecture & Philosophy

The engine is built around a layered clean architecture:

```
        API Layer (FastAPI endpoints / Control Plane)
                    ↓
     Service Layer (Ingestion, Hybrid Search, Compaction)
                    ↓
 Retrieval Engine (In-graph filtering, exact similarity fallback)
                    ↓
  HNSW Index (VHNS graph) ───► Segment Cache (LRU Manager)
        ↓                              ↓
  snapshots/*.bin (Persisted)    segments/*.bin (Aligned vectors)
```

### Key Technical Details
1. **Zero Vector Duplication**: The HNSW graph structure (`graph.bin`) stores 16-byte segment coordinates (`segment_uuid` and `vector_idx`) rather than duplicating dense floating-point arrays on disk.
2. **SIMD-Aligned Binary Format (`VSEG`)**:
   - Header is exactly 64 bytes.
   - Vector blocks are aligned to **64-byte boundaries**, ensuring memory-mapped pointers align with CPU cache lines.
   - Float32 arrays support fast AVX-2 / AVX-512 vector arithmetic operations.
3. **Compact Index Records**:
   - HNSW index records are packed to exactly **32 bytes** (reducing offsets to 4-byte `uint32` values).
   - This allows 2 index records to sit on a single standard 64-byte CPU cache line, minimizing random cache-miss latencies.
4. **Relational Control Plane**:
   - PostgreSQL stores collections, chunk texts, document structures, audit logs, task lists, and metadata schemas.
   - Dialect-aware mappings (`FloatArrayType` and `JSONBType`) map natively to PostgreSQL properties during production runs, and serialize to JSON-structured text in SQLite during test suites.

---

## 📁 Repository Directory Structure

```text
vector_retrieval_engine/
├── core/
│   ├── config.py                 # Pydantic settings loading configuration
│   └── database.py               # Database engine session provider
├── models/
│   └── database_models.py        # SQLAlchemy schema models (FloatArrayType & JSONBType)
├── embeddings/
│   └── providers.py              # Pluggable OpenAI, Gemini, and Mock embeddings
├── index/
│   ├── distance/
│   │   └── metrics.py            # Cosine similarity, L2, Dot Product, Manhattan using NumPy
│   ├── graph/
│   │   └── hnsw.py               # Custom HNSW index construction and search
│   └── serialization/
│       └── persistence.py        # Binary graph serializer and snapshot manager (graph.bin)
├── services/
│   ├── chunking/
│   │   └── chunkers.py           # Pluggable text splitters (Fixed, Sliding, Recursive, Semantic)
│   ├── ingestion/
│   │   └── pipeline.py           # Document parser, hash checks, and segment appender
│   ├── retrieval/
│   │   ├── filter_resolver.py    # PostgreSQL JSONB metadata resolver
│   │   ├── hybrid_search.py      # Combines semantic vector similarity and keyword overlaps
│   │   └── retriever.py          # Exact similarity O(N) baseline retriever
│   └── worker/
│       └── worker.py             # Durable SQL-backed background worker queue
├── storage/
│   ├── cache/
│   │   ├── cached_segment.py     # In-memory numpy segment cache representation
│   │   └── segment_cache.py      # LRU cache manager with memory constraints eviction
│   └── segments/
│       ├── writer.py             # Binary segment serializer (VSEG format)
│       └── compactor.py          # Background segment lifecycle compactor and merger
└── tests/                        # 35 test verification suite
```

---

## 🛠️ Implemented Components (Phases 1 - 9)

### Phase 1: Relational Control Plane
* Maps schemas for `users`, `collections`, `documents`, `chunks`, `embeddings`, `metadata`, and `audit_logs`.
* Connects dynamically via SQLAlchemy and uses Alembic for schema migrations.

### Phase 2: Ingestion Pipeline
* Supports pluggable chunkers (Fixed, Sliding Window, Recursive Character, and Semantic sentence splitters).
* Embeds chunks via OpenAI, Gemini, or a deterministic keyword-correlated Mock provider.
* Appends vectors and payloads into aligned binary files on disk.

### Phase 3: Exact Cosine Similarity Retrieval
* A baseline exact retrieval scan using NumPy.
* Serves as a 100% precision benchmark to evaluate approximation indexes.

### Phase 4: Custom HNSW Index
* Multi-level navigable small world graph.
* Utilizes probability level generators for exponential node decay.
* Implements neighbor selection heuristics that prioritize linkage direction diversity to prevent disconnected cluster islands.
* Supports soft-delete tombstones.

### Phase 5: Index Persistence Engine
* Serializes HNSW graphs to binary (`graph.bin`) and metadata JSON (`metadata.json`).
* Loads existing graphs from snapshots on system startup rather than rebuilding them.

### Phase 6: Memory Management (Segment Cache)
* Lazily loads binary segments from disk to memory upon cache misses.
* Tracks NumPy vector matrix sizes and payload string allocations.
* Evicts cold segments using an LRU (Least Recently Used) policy when memory exceeds limits.

### Phase 7: Segment Lifecycles & Compaction
* Automatically seals growing segments when they reach capacity, building HNSW indices on the fly.
* Merges multiple small segments into a consolidated file, purging soft-deleted tombstones.
* Updates PostgreSQL database coordinates in a single database transaction.

### Phase 8: Hybrid Search & Metadata Pre-Filtering
* Decodes JSONB metadata constraints in PostgreSQL.
* Integrates allowed candidate lists into HNSW search loops (in-graph filtering).
* Fuses semantic similarity scores and keyword text overlap scores using a weighted parameter $\alpha$.

### Phase 9: Durable Worker Queue
* Submits tasks to a durable table (`task_logs`).
* Consumes task logs asynchronously in a daemon thread, locking records to prevent execution races.
* Captures traceback details on errors and recovers stuck tasks upon server crash/shutdown.

---

## 📐 Binary Storage Specifications

### 1. Vector Segment File (`VSEG` / `segment.bin`)
- **64-Byte File Header**:
  ```text
  [4B Magic Number: 'VSEG'] [2B Version] [16B Segment UUID]
  [4B Record Count] [4B Deleted Count] [8B Created Double Timestamp]
  [8B Checksum CRC32] [18B Padding]
  ```
- **32-Byte Record Table** (Repeated for `Record Count`):
  ```text
  [16B Record UUID] [4B Vector Offset] [4B Payload Offset]
  [4B Payload Length] [1B Status (1=Active, 2=Deleted)] [3B Padding]
  ```
- **Vector Block**: Contiguous float32 numbers.
- **Payload Block**: Length-prefixed UTF-8 encoded JSON strings.

### 2. Graph Index Snapshot (`VHNS` / `graph.bin`)
- **64-Byte File Header**:
  ```text
  [4B Magic Number: 'VHNS'] [2B Version] [2B Max Level]
  [16B Entry Node UUID] [4B Node Count] [36B Padding]
  ```
- **Nodes Block** (Repeated for `Node Count`):
  ```text
  [16B Node UUID] [16B Segment UUID] [4B Vector Index Offset]
  [2B Max Level] [Level counts of neighbors...] [Neighbor UUID lists...]
  ```

---

## 🧪 Running Verification Tests

The test suite runs on Python 3.10+ using pytest. It validates SQLite schemas compatibility layers, binary packing math, retrieval accuracy, and thread concurrency.

1. **Install dependencies**:
   ```bash
   pip install pytest numpy sqlalchemy pydantic-settings
   ```
2. **Execute tests**:
   ```bash
   python -m pytest
   ```

All 35 tests verify:
- In-graph filtering recall precision.
- Compact segment record merges.
- Background worker transaction thread safety.
- Cache evictions and hit-ratio metrics.
