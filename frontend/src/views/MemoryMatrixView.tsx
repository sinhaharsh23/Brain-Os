import { useBrain } from "../store/useBrainStore"
import { useMockKnowledgeBase } from "../hooks/useMocks"

export default function MemoryMatrixView() {
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const model = useBrain((s) => s.model)
  const running = useBrain((s) => s.running)
  const { docs } = useMockKnowledgeBase()

  const totalTokens = tokens.length + generatedTokens.length
  const maxTokens = model?.context_length ?? 32768
  const kvCacheBytes = totalTokens * 24 * 64 * 4 * 2 // 24 layers, 64 dim, float16 (2 bytes), K+V
  const kvKb = (kvCacheBytes / 1024).toFixed(1)

  return (
    <div className="section-page-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div>
          <h2 className="section-title">MEMORY MATRIX (CONTEXT STORAGE & KV CACHE)</h2>
          <span className="section-subtitle text-dim text-xs">
            Dynamic transformer Key-Value cache and vectorized knowledge context
          </span>
        </div>
        <span className="badge-cyan">{totalTokens} TOKENS OCCUPIED</span>
      </div>

      <div className="memory-matrix-grid">
        {/* KV Cache Overview Card */}
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">TRANSFORMER KV CACHE ALLOCATION</span>
            <span className={`badge-${running ? "amber" : "emerald"}`}>
              {running ? "ALLOCATING" : "SYNCED"}
            </span>
          </div>
          <div className="kv-stats-row">
            <div className="stat-pill">
              <span className="stat-pill-label">OCCUPIED TOKENS</span>
              <span className="stat-pill-val text-cyan">{totalTokens} / {maxTokens}</span>
            </div>
            <div className="stat-pill">
              <span className="stat-pill-label">VRAM FOOTPRINT</span>
              <span className="stat-pill-val text-amber">{kvKb} KB</span>
            </div>
            <div className="stat-pill">
              <span className="stat-pill-label">CACHE SLOTS</span>
              <span className="stat-pill-val">2 KV Heads (GQA)</span>
            </div>
          </div>
          <div className="kv-cache-visual-strip">
            {Array.from({ length: 32 }).map((_, idx) => {
              const isFilled = idx < Math.ceil((totalTokens / maxTokens) * 32) || (idx < 6 && totalTokens > 0)
              return (
                <div
                  key={idx}
                  className={`kv-slot-cell ${isFilled ? "slot-filled" : ""}`}
                  title={`KV Cache Block #${idx}`}
                />
              )
            })}
          </div>
        </div>

        {/* Knowledge Base Documents (Context Feed) */}
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">KNOWLEDGE BASE INGESTION (RAG)</span>
            <span className="badge-outline text-xxs">TODO: RAG ENDPOINTS</span>
          </div>
          <div className="kb-docs-list">
            {docs.map((d) => (
              <div key={d.id} className="kb-doc-item flex items-center justify-between">
                <div className="doc-info">
                  <span className="doc-title text-bright text-xs">{d.title}</span>
                  <span className="doc-category text-dim text-xxs">{d.category} · {d.tokens} tokens</span>
                </div>
                <div className="doc-relevance text-emerald text-xs font-bold">
                  {(d.relevance * 100).toFixed(0)}% MATCH
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
