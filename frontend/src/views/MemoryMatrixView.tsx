import { useBrain } from "../store/useBrainStore"
import { useMockKnowledgeBase } from "../hooks/useMocks"

export default function MemoryMatrixView() {
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const telemetry = useBrain((s) => s.telemetry)
  const provider = useBrain((s) => s.provider)
  const cache = useBrain((s) => s.kvCache[s.currentStep])
  const model = useBrain((s) => s.model)
  const running = useBrain((s) => s.running)
  const { docs } = useMockKnowledgeBase()

  const totalTokens = telemetry?.tokens.sequence_tokens ?? (provider === "ollama" ? null : tokens.length + generatedTokens.length)
  const maxTokens = telemetry?.model.context_window ?? (provider === "ollama" ? null : model?.context_length) ?? null
  const kvKb = cache?.total_kb?.toFixed(1) ?? "Unavailable"

  return (
    <div className="section-page-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div>
          <h2 className="section-title">{provider === "ollama" ? "MEMORY MATRIX (CONTEXT & PROVIDER CACHE)" : "MEMORY MATRIX (CONTEXT STORAGE & KV CACHE)"}</h2>
          <span className="section-subtitle text-dim text-xs">
            {provider === "ollama" ? "Ollama prompt-cache counts when returned; transformer K/V tensors are unavailable." : "Captured transformer Key-Value cache and vectorized model input context"}
          </span>
        </div>
        <span className="badge-cyan">{totalTokens ?? "Unavailable"} SEQUENCE TOKENS</span>
      </div>

      <div className="memory-matrix-grid">
        {/* KV Cache Overview Card */}
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">{provider === "ollama" ? "OLLAMA PROMPT CACHE" : "TRANSFORMER KV CACHE"}</span>
            <span className={`badge-${running ? "amber" : "emerald"}`}>
              {running ? "ALLOCATING" : (cache ? "CAPTURED" : "UNAVAILABLE")}
            </span>
          </div>
          <div className="kv-stats-row">
            <div className="stat-pill">
              <span className="stat-pill-label">{provider === "ollama" ? "CACHED PROMPT TOKENS" : "KV CACHE SEQUENCE LENGTH"}</span>
              <span className="stat-pill-val text-cyan">{provider === "ollama" ? telemetry?.tokens.cached_prompt_tokens ?? "Unavailable" : `${cache?.seq_length ?? "Unavailable"} / ${maxTokens ?? "Unavailable"}`}</span>
            </div>
            <div className="stat-pill">
              <span className="stat-pill-label">{provider === "ollama" ? "K/V TENSOR BYTES" : "CAPTURED CACHE BYTES"}</span>
              <span className="stat-pill-val text-amber">{provider === "ollama" ? "Unavailable" : `${kvKb} KB`}</span>
            </div>
            <div className="stat-pill">
              <span className="stat-pill-label">{provider === "ollama" ? "K/V TENSOR METADATA" : "CACHE SLOTS"}</span>
              <span className="stat-pill-val">{provider === "ollama" ? "Unavailable" : `${model?.num_kv_heads ?? "Unavailable"} KV Heads`}</span>
            </div>
          </div>
          {provider !== "ollama" && <div className="kv-cache-visual-strip">
            {Array.from({ length: 32 }).map((_, idx) => {
              const isFilled = idx < Math.ceil(((cache?.seq_length ?? 0) / (maxTokens ?? 1)) * 32)
              return (
                <div
                  key={idx}
                  className={`kv-slot-cell ${isFilled ? "slot-filled" : ""}`}
                  title="Relative captured sequence length; not individual cache blocks"
                />
              )
            })}
          </div>}
        </div>

        {/* Knowledge Base (DEMO DATA only when enabled) Documents (Context Feed) */}
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">KNOWLEDGE BASE INGESTION (RAG)</span>
            <span className="badge-outline text-xxs">{docs.length ? "DEMO DATA" : "NO INDEX DATA"}</span>
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
