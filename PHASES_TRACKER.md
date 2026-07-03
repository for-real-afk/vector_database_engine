# Production Vector Retrieval Engine - Phase Execution Tracker

This document tracks exactly what was implemented, the problems solved, and the architectural decisions made in each phase of development (Phases 0 through 9).

---

## 📂 Phase 0: System Architecture & Directory Layout

### What We Did:
- Wrote [SYSTEM_ARCHITECTURE.md](file:///d:/projects/vector_retrieval_engine/SYSTEM_ARCHITECTURE.md) outlining the storage layout (Header format, 64-byte vector alignment), query path, MRSW locking, WAL protocol, and AWS mapping.
- Established the repository directory layout, creating packages for embeddings, database models, graph indices, search services, caching manager, and segment serialization.

### Core Problems Solved:
- Unified the storage engine model where relational metadata sits in PostgreSQL and heavy float arrays sit in aligned binary segment files on disk to prevent database bloating.

---

## 🗄️ Phase 1: PostgreSQL Schema & Alembic Migrations

### What We Did:
- Implemented SQLAlchemy database tables `User`, `Collection`, `Document`, `Chunk`, `Embedding`, `Metadata`, and `AuditLog` inside [models/database_models.py](file:///d:/projects/vector_retrieval_engine/models/database_models.py).
- Configured Alembic migrations in [migrations/env.py](file:///d:/projects/vector_retrieval_engine/migrations/env.py) to read database configuration parameters dynamically from environment variables.
- Programmed dialect-aware custom columns (`FloatArrayType` and `JSONBType`) that map to native arrays/JSONB on PostgreSQL, and fall back to json-serialized text on SQLite.

### Core Problems Solved:
- Handled multi-namespace SQLite testing compatibility, allowing unit tests to run in-memory/file-backed without requiring a running PostgreSQL server.

---

## 📥 Phase 2: Document Ingestion Pipeline

### What We Did:
- Implemented pluggable chunkers inside [services/chunking/chunkers.py](file:///d:/projects/vector_retrieval_engine/services/chunking/chunkers.py): `FixedSizeChunker`, `SlidingWindowChunker`, `RecursiveCharacterChunker` (recursively splitting on separators lists), and `SemanticChunker` (sentence splits with adjacent vector distance checks).
- Coded pluggable providers inside [embeddings/providers.py](file:///d:/projects/vector_retrieval_engine/embeddings/providers.py) (Mock, OpenAI, and Gemini lazy-loaded wrappers).
- Wrote the binary segment serializer inside [storage/segments/writer.py](file:///d:/projects/vector_retrieval_engine/storage/segments/writer.py) to parse headers and write aligned arrays.
- Wrote the pipeline coordinator inside [services/ingestion/pipeline.py](file:///d:/projects/vector_retrieval_engine/services/ingestion/pipeline.py) managing duplicate hashes check, chunking, embedding, database writes, and file appends.

### Core Problems Solved:
- Engineered float32 alignment offsets to make memory-mapped vector buffers align with CPU cache lines.
- Restricted index records to exactly 32 bytes (using 4-byte `uint32` offsets instead of `uint64`), allowing 2 records per 64-byte CPU cache line.

---

## 🔍 Phase 3: Exact Cosine Similarity (O(N) Search Engine)

### What We Did:
- Created the math library [index/distance/metrics.py](file:///d:/projects/vector_retrieval_engine/index/distance/metrics.py) implementing Cosine, L2 (Euclidean), Dot Product, and Manhattan distance metrics using NumPy.
- Programmed [services/retrieval/retriever.py](file:///d:/projects/vector_retrieval_engine/services/retrieval/retriever.py) executing linear scans on segments, filtering tombstones, and ranking Top-K candidates.
- Upgraded the `MockEmbeddingProvider` to sum word-level vectors.

### Core Problems Solved:
- The word-overlap Mock provider mimics actual keyword/semantic similarity in local tests, preventing random orthogonal vector alignments from breaking search assertions.
- Created a 100% precision exact retriever baseline to measure HNSW recall.

---

## 🕸️ Phase 4: Custom HNSW Graph Index Implementation

### What We Did:
- Programmed [index/graph/hnsw.py](file:///d:/projects/vector_retrieval_engine/index/graph/hnsw.py), developing `HNSWNode` and `CandidateList` classes from scratch.
- Implemented multi-level node insertions with exponential probability layer decay.
- Implemented neighbor selection heuristics to diversify neighbor link distributions.
- Coded in-graph soft deletions (Tombstones).

### Core Problems Solved:
- Neighbor list capping forces strict degree bounds constraints.
- Neighbor selection heuristics prevent link clustering, keeping the graph highly navigable and preventing isolated subgraphs.

---

## 💾 Phase 5: HNSW Persistence Engine

### What We Did:
- Programmed HNSW binary serialization inside [index/serialization/persistence.py](file:///d:/projects/vector_retrieval_engine/index/serialization/persistence.py), saving node level linkages and segment coordinates (`segment_uuid` and `vector_idx`) into `graph.bin`.
- Saved index parameters (`M`, `M0`, `ef_construction`, `ef_search`) and soft-deleted tombstones lists inside `metadata.json`.
- Implemented `HNSWIndexManager` snapshot and restore routines.

### Core Problems Solved:
- Avoided duplicating heavy dense floats inside HNSW index files by storing 16-byte coordinates pointing to segment files, reducing graph snapshot sizes.

---

## 🧠 Phase 6: Memory Cache & Segment Caching Layer

### What We Did:
- Created [storage/cache/cached_segment.py](file:///d:/projects/vector_retrieval_engine/storage/cache/cached_segment.py) wrapping in-memory segments with 2D float32 matrices.
- Developed [storage/cache/segment_cache.py](file:///d:/projects/vector_retrieval_engine/storage/cache/segment_cache.py) containing `SegmentCacheManager` which lazy loads segment binary files on cache misses.
- Implemented an LRU (Least Recently Used) segment eviction policy triggered when total byte limits are exceeded.

### Core Problems Solved:
- Prevented thread locks under high read concurrency. Slicing raw NumPy matrices returns zero-copy pointers, keeping HNSW graph traversals fast.
- Corrected `.gitignore` package rules to track python cache managers.

---

## 🔄 Phase 7: Segment Lifecycle Management

### What We Did:
- Programmed [storage/segments/compactor.py](file:///d:/projects/vector_retrieval_engine/storage/segments/compactor.py) implementing should-compact checks and background segment merges.
- Programmed transactional metadata updates in PostgreSQL, updating collection coordinates (`segment_id` and `vector_idx`).
- Linked segment sealing triggers inside the ingestion pipeline to automatically compile and snapshot HNSW graph indices.

### Core Problems Solved:
- Tombstones are physically dropped during compaction writes, reducing disk footprint.
- All coordinate redirections occur inside a single SQL transaction. If database updates fail, the compactor rolls back and deletes the temp files, preventing orphaned indices.

---

## 🔀 Phase 8: Hybrid Retrieval & Metadata Pre-Filtering

### What We Did:
- Programmed [services/retrieval/filter_resolver.py](file:///d:/projects/vector_retrieval_engine/services/retrieval/filter_resolver.py) to parse metadata tags and return lists of allowed embedding UUIDs.
- Updated HNSW search traversals to support allowed candidate lists, implementing relaxed in-graph pre-filtering.
- Programmed the hybrid retrieval orchestrator in [services/retrieval/hybrid_search.py](file:///d:/projects/vector_retrieval_engine/services/retrieval/hybrid_search.py), combining vector similarity and keyword overlap scores.

### Core Problems Solved:
- In-graph filtering navigates through both matching and non-matching nodes to preserve path connectivity, but only returns nodes satisfying the metadata filter, avoiding the subgraph disconnection problems of strict pre-filtering.

---

## 🏭 Phase 9: Background Worker Queue & Tasks

### What We Did:
- Added the `TaskLog` model to SQLAlchemy database tables.
- Programmed the background worker loop in [services/worker/worker.py](file:///d:/projects/vector_retrieval_engine/services/worker/worker.py) to consume pending jobs (`BATCH_INGEST`, `COMPACT_SEGMENTS`, `BUILD_HNSW`).
- Implemented startup crash recovery resetting tasks stuck in `processing` status to `pending`.
- Updated test configuration to use file-backed SQLite database directories.

### Core Problems Solved:
- Replaced in-memory SQLite (`:memory:`) database connections in test fixtures with temp files, allowing worker threads and test runners to share state and preventing `database is locked` concurrency conflicts.
