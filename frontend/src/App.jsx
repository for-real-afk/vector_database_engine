import React, { useState } from 'react'
import TerminalBackground from './components/TerminalBackground'
import { 
  Database, 
  Terminal, 
  Search, 
  Settings, 
  Cpu, 
  Lock, 
  GitBranch, 
  Plus, 
  Play, 
  ShieldCheck 
} from 'lucide-react'

export default function App() {
  const [collections, setCollections] = useState([
    { id: '1', name: 'kb_wiki_docs', dimension: 1536, metric: 'Cosine', chunker: 'Recursive', count: 12450 },
    { id: '2', name: 'user_support_tickets', dimension: 384, metric: 'L2', chunker: 'Fixed', count: 852 }
  ])

  const [newCol, setNewCol] = useState({ name: '', dimension: 1536, metric: 'Cosine', chunker: 'Recursive' })
  const [query, setQuery] = useState('')
  const [kParam, setKParam] = useState(5)
  const [efSearch, setEfSearch] = useState(64)
  const [searchMetric, setSearchMetric] = useState('Cosine')
  
  const [searchResults, setSearchResults] = useState([
    { id: 'chunk_1294', score: 0.941, text: "HNSW is a graph-based indexing algorithm that creates a multi-layer graph of vectors, allowing logarithmic search complexity O(log N). Nodes on higher layers represent longer links, while lower layers contain local linkages.", collection: "kb_wiki_docs" },
    { id: 'chunk_4812', score: 0.887, text: "To maintain durability, all vector database modifications are written to a Write-Ahead Log (WAL) before updating the segment buffer structure. On crash recovery, the engine replays unprocessed WAL entries.", collection: "kb_wiki_docs" }
  ])

  const handleCreateCollection = (e) => {
    e.preventDefault()
    if (!newCol.name) return
    setCollections([
      ...collections,
      {
        id: String(collections.length + 1),
        name: newCol.name.toLowerCase().replace(/\s+/g, '_'),
        dimension: parseInt(newCol.dimension) || 1536,
        metric: newCol.metric,
        chunker: newCol.chunker,
        count: 0
      }
    ])
    setNewCol({ name: '', dimension: 1536, metric: 'Cosine', chunker: 'Recursive' })
  }

  const handleSearch = (e) => {
    e.preventDefault()
    // Simulated engine query path output
    setSearchResults([
      { id: 'chunk_8291', score: 0.923, text: `Matched snippet for query '${query}': Segments are compiled and stored as flat binary files. Raw float arrays are aligned to 64-byte chunks in segment.bin for optimized SIMD CPU execution.`, collection: "kb_wiki_docs" },
      { id: 'chunk_0102', score: 0.812, text: "Hybrid query planner combines exact BM25 keyword matching from PostgreSQL indexes with semantic cosine similarity searches on HNSW indexes.", collection: "kb_wiki_docs" }
    ])
  }

  return (
    <div className="app-container">
      {/* Faded Background log terminal */}
      <TerminalBackground />

      {/* Main dashboard content area */}
      <main className="main-dashboard">
        <header className="header">
          <div className="brand-section">
            <div className="logo-badge">V</div>
            <div className="logo-text">ANTIGRAVITY<span>_ENGINE</span></div>
          </div>
          <div className="status-indicator">
            <div className="dot-pulse"></div>
            <span>DAEMON: ONLINE (PORT 8000)</span>
          </div>
        </header>

        <div className="flex-column">
          {/* Top Panel - Engine Summary */}
          <div className="panel-card" style={{ borderLeft: '4px solid var(--accent-teal)' }}>
            <h2 className="panel-title"><Cpu size={16} /> Core Engine Status</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
              <div>
                <span className="form-label">Storage Engine</span>
                <p style={{ fontWeight: 'bold' }}>Hybrid (PG + Custom Segments)</p>
              </div>
              <div>
                <span className="form-label">Memory Caches</span>
                <p style={{ color: 'var(--accent-teal)' }}>Graph Cache: 100% | Vector Cache: LRU</p>
              </div>
              <div>
                <span className="form-label">Active Growing Segment</span>
                <p style={{ fontFamily: 'monospace' }}>seg_grow_091b (Vectors: 2,450 / 50,000)</p>
              </div>
              <div>
                <span className="form-label">WAL Sync State</span>
                <p style={{ color: 'var(--accent-green)' }}>ACTIVE (Durability level: Full)</p>
              </div>
            </div>
          </div>

          <div className="grid-cols-2">
            {/* Collection Management */}
            <div className="panel-card">
              <h2 className="panel-title"><Database size={16} /> Collections & Schema</h2>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '20px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left' }}>
                    <th style={{ padding: '8px 0', color: 'var(--text-muted)' }}>Name</th>
                    <th style={{ padding: '8px 0', color: 'var(--text-muted)' }}>Dims</th>
                    <th style={{ padding: '8px 0', color: 'var(--text-muted)' }}>Metric</th>
                    <th style={{ padding: '8px 0', color: 'var(--text-muted)' }}>Vectors</th>
                  </tr>
                </thead>
                <tbody>
                  {collections.map(c => (
                    <tr key={c.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                      <td style={{ padding: '12px 0', fontWeight: 'bold' }}>{c.name}</td>
                      <td style={{ padding: '12px 0', fontFamily: 'monospace' }}>{c.dimension}</td>
                      <td style={{ padding: '12px 0' }}>{c.metric}</td>
                      <td style={{ padding: '12px 0', color: 'var(--accent-teal)' }}>{c.count.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <form onSubmit={handleCreateCollection} style={{ borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
                <h3 style={{ fontSize: '13px', marginBottom: '15px', color: 'var(--text-muted)' }}>Create Collection</h3>
                <div className="form-group">
                  <label className="form-label">Collection Name</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    value={newCol.name} 
                    onChange={e => setNewCol({...newCol, name: e.target.value})}
                    placeholder="e.g. documentation_embeddings"
                  />
                </div>
                <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
                  <div style={{ flex: 1 }}>
                    <label className="form-label">Dimension</label>
                    <input 
                      type="number" 
                      className="form-input" 
                      value={newCol.dimension}
                      onChange={e => setNewCol({...newCol, dimension: e.target.value})}
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label className="form-label">Distance Metric</label>
                    <select 
                      className="form-input"
                      value={newCol.metric}
                      onChange={e => setNewCol({...newCol, metric: e.target.value})}
                    >
                      <option>Cosine</option>
                      <option>L2</option>
                      <option>DotProduct</option>
                      <option>Manhattan</option>
                    </select>
                  </div>
                </div>
                <button type="submit" className="btn" style={{ width: '100%' }}>
                  <Plus size={16} /> Add Collection
                </button>
              </form>
            </div>

            {/* HNSW Index visualizer Mock */}
            <div className="panel-card">
              <h2 className="panel-title"><GitBranch size={16} /> HNSW Live Graph Visualizer</h2>
              <div className="graph-container">
                {/* SVG representing visual graph traversal */}
                <svg width="100%" height="100%" style={{ position: 'absolute' }}>
                  <g opacity="0.3">
                    <line x1="50" y1="50" x2="150" y2="70" stroke="#1a2233" strokeWidth="2" />
                    <line x1="150" y1="70" x2="250" y2="40" stroke="#1a2233" strokeWidth="2" />
                    <line x1="250" y1="40" x2="350" y2="100" stroke="#1a2233" strokeWidth="2" />
                    <line x1="50" y1="180" x2="150" y2="190" stroke="#1a2233" strokeWidth="2" />
                    <line x1="150" y1="190" x2="280" y2="150" stroke="#1a2233" strokeWidth="2" />
                    <line x1="280" y1="150" x2="350" y2="180" stroke="#1a2233" strokeWidth="2" />
                  </g>
                  {/* Query traversal highlighted links */}
                  <g>
                    <line x1="50" y1="50" x2="150" y2="190" stroke="var(--accent-teal)" strokeWidth="2" strokeDasharray="4 4" />
                    <line x1="150" y1="190" x2="280" y2="150" stroke="var(--accent-cyan)" strokeWidth="2" />
                    <circle cx="50" cy="50" r="6" fill="var(--accent-teal)" />
                    <circle cx="150" cy="190" r="5" fill="var(--accent-teal)" />
                    <circle cx="280" cy="150" r="5" fill="var(--accent-cyan)" />
                    {/* Pulsing active candidate */}
                    <circle cx="280" cy="150" r="10" fill="none" stroke="var(--accent-cyan)" strokeWidth="2" opacity="0.6">
                      <animate attributeName="r" values="5;15;5" dur="2s" repeatCount="indefinite" />
                    </circle>
                  </g>
                </svg>
                <div style={{ position: 'absolute', bottom: '15px', left: '15px', zIndex: 10, fontSize: '11px' }}>
                  <span style={{ color: 'var(--accent-teal)' }}>● Entry point</span> | <span style={{ color: 'var(--accent-cyan)' }}>● Candidate Node</span> | <span style={{ color: 'var(--text-muted)' }}>● Graph links</span>
                </div>
                <div style={{ position: 'absolute', top: '15px', right: '15px', background: 'rgba(0,0,0,0.6)', padding: '4px 8px', borderRadius: '4px', fontSize: '11px', border: '1px solid var(--border-color)' }}>
                  <span>Hhops: 12 | max_layer: 4</span>
                </div>
              </div>
              <div style={{ marginTop: '20px' }}>
                <span className="form-label" style={{ marginBottom: '8px' }}>Graph Compaction & Optimizations</span>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button className="btn" style={{ flex: 1, backgroundColor: 'transparent', border: '1px solid var(--border-color)' }} onClick={() => alert('Compacting database segments...')}>
                    Run Compactor
                  </button>
                  <button className="btn" style={{ flex: 1, backgroundColor: 'transparent', border: '1px solid var(--border-color)' }} onClick={() => alert('Generating index snapshot graph.bin...')}>
                    Save Snapshot
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Search Sandbox Panel */}
          <div className="panel-card">
            <h2 className="panel-title"><Search size={16} /> Query Execution Sandbox</h2>
            <form onSubmit={handleSearch}>
              <div style={{ display: 'flex', gap: '15px', marginBottom: '20px' }}>
                <div style={{ flex: 2 }}>
                  <label className="form-label">Semantic Query Input</label>
                  <div style={{ position: 'relative' }}>
                    <input 
                      type="text" 
                      className="form-input" 
                      style={{ paddingLeft: '40px' }}
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      placeholder="Type text query to embed and search..."
                    />
                    <Search size={16} style={{ position: 'absolute', left: '14px', top: '12px', color: 'var(--text-muted)' }} />
                  </div>
                </div>
                <div>
                  <label className="form-label">Top K</label>
                  <input 
                    type="number" 
                    className="form-input" 
                    style={{ width: '80px' }}
                    value={kParam}
                    onChange={e => setKParam(e.target.value)}
                  />
                </div>
                <div>
                  <label className="form-label">ef_search</label>
                  <input 
                    type="number" 
                    className="form-input" 
                    style={{ width: '90px' }}
                    value={efSearch}
                    onChange={e => setEfSearch(e.target.value)}
                  />
                </div>
                <div style={{ alignSelf: 'flex-end' }}>
                  <button type="submit" className="btn">
                    <Play size={14} /> Execute Query
                  </button>
                </div>
              </div>
            </form>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <span className="form-label">Query Results (Candidates Sorted by Distance Metric)</span>
              {searchResults.map((res, i) => (
                <div key={i} style={{ backgroundColor: 'var(--bg-primary)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', gap: '20px' }}>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontSize: '11px', color: 'var(--accent-teal)', display: 'block', marginBottom: '6px' }}>
                      ID: {res.id} | Collection: {res.collection}
                    </span>
                    <p style={{ color: 'var(--text-primary)', fontSize: '13px' }}>{res.text}</p>
                  </div>
                  <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <span className="form-label" style={{ marginBottom: '2px' }}>Score</span>
                    <span style={{ fontSize: '18px', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>{res.score.toFixed(4)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <footer style={{ marginTop: '40px', borderTop: '1px solid var(--border-color)', paddingTop: '20px', display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <ShieldCheck size={12} style={{ color: 'var(--accent-teal)' }} /> Dynamic RBAC & Namespace Isolation Active
          </span>
          <span>v1.0.0-beta // Antigravity Vector Database Engine</span>
        </footer>
      </main>
    </div>
  )
}
