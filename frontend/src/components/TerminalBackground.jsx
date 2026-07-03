import React, { useState, useEffect, useRef } from 'react'

const MOCK_LOGS = [
  "DB_ENGINE: Initializing WAL recovery protocol...",
  "WAL: Scanning storage/wal/wal.log...",
  "WAL: Replay completed successfully (LSN: 481029)",
  "STORAGE: Mapped 4 sealed segments in collection 'default'",
  "CACHE: Segment cache initialized (LRU, max_size=500MB)",
  "INDEX: Loading HNSW graph 'graph.bin' via mmap...",
  "INDEX: HNSW initialized successfully (max_layer: 4, entry_point_node_id: 1024)",
  "API: REST Engine starting on port 8000 (workers=4)...",
  "API: Health check passed - status OK",
  "QUERY_PLANNER: Optimized filter pipeline selected for collection 'default'",
  "ENGINE: Active growing segment created (ID: seg_grow_43102)",
  "WAL: Flushed log entry to disk (LSN: 481030, CRC32: 0x4D2A91F2)",
  "HNSW: Added node #1042 at Layer 0 (segment: seg_grow_43102)",
  "COMPACT_WORKER: Compaction threshold checked (0 growing, 4 sealed segments)",
  "BUFFER_POOL: Page read hit (segment: seg_seal_01024, offset: 40960)",
  "SIMD_ENGINE: Vector calculations aligned to 64-byte boundary (AVX2 mode active)",
  "SECURITY: Tenant validation passed for request namespace 'prod-customer-data'",
  "QUERY_PLANNER: Executing Hybrid Search pre-filter allowed-list check (matches: 482)",
  "HNSW: Graph search traversal complete (hops: 12, candidate_nodes_visited: 84)",
  "RERANKER: Score fusion completed for 20 candidates (RRF mode)",
  "BUFFER_POOL: Evicted page 142 from RAM cache (LRU rule applied)",
]

export default function TerminalBackground() {
  const [logs, setLogs] = useState([])
  const containerRef = useRef(null)

  useEffect(() => {
    // Fill first 10 logs instantly
    const initialLogs = Array.from({ length: 12 }, () => {
      const log = MOCK_LOGS[Math.floor(Math.random() * MOCK_LOGS.length)]
      const timestamp = new Date().toISOString().substring(11, 19)
      return `[${timestamp}] ${log}`
    })
    setLogs(initialLogs)

    const interval = setInterval(() => {
      setLogs(prev => {
        const nextLogs = [...prev]
        if (nextLogs.length > 25) {
          nextLogs.shift()
        }
        const randomLog = MOCK_LOGS[Math.floor(Math.random() * MOCK_LOGS.length)]
        const timestamp = new Date().toISOString().substring(11, 19)
        nextLogs.push(`[${timestamp}] ${randomLog}`)
        return nextLogs
      })
    }, 4500) // update every few seconds for a slow, organic feel

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="terminal-bg-sidebar">
      <div className="terminal-header">
        SYS_LOGS // VECTOR_ENGINE_DAEMON // EST. 2026
      </div>
      <div ref={containerRef} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {logs.map((log, index) => {
          // Highlight some keywords
          let content = log
          if (log.includes("WAL:")) {
            content = log.replace("WAL:", "WAL:")
            return (
              <div key={index} className="terminal-line">
                <span className="cyan">{log.split("WAL:")[0]}WAL:</span>
                {log.split("WAL:")[1]}
              </div>
            )
          } else if (log.includes("INDEX:") || log.includes("HNSW:")) {
            return (
              <div key={index} className="terminal-line">
                <span className="green">{log.split(/INDEX:|HNSW:/)[0]}INDEX:</span>
                {log.split(/INDEX:|HNSW:/)[1]}
              </div>
            )
          }
          return (
            <div key={index} className="terminal-line">
              {log}
            </div>
          )
        })}
      </div>
    </div>
  )
}
