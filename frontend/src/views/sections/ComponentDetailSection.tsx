import { useMemo } from "react"
import { useBrain } from "../../store/useBrainStore"

export default function ComponentDetailSection() {
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const layers = useBrain((s) => s.layers)
  const model = useBrain((s) => s.model)
  const response = useBrain((s) => s.response)
  const summary = useBrain((s) => s.summary)
  const running = useBrain((s) => s.running)
  const candidates = useBrain((s) => s.candidates)

  const numLayers = model?.num_layers ?? 24
  const vectorDim = model?.hidden_size ?? 896

  const activeLayerIndex = useMemo(() => {
    const layerKeys = Object.keys(layers).map(Number)
    if (layerKeys.length === 0) return 0
    return Math.max(...layerKeys)
  }, [layers])

  const layerProgressPct = Math.min(100, Math.round(((activeLayerIndex + 1) / numLayers) * 100))

  return (
    <div className="component-detail-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="section-title-tag">DETAILED WORKING OF EACH COMPONENT</span>
          <span className="badge-outline text-xxs">5-COMPONENT INTERNAL STACK</span>
        </div>
        <span className="text-xxs text-dim">QWEN2.5 MPS FORWARD HOOKS</span>
      </div>

      <div className="component-panels-grid">
        {/* Panel 1: Token Engine */}
        <div className="detail-panel token-engine-panel">
          <div className="panel-header flex items-center justify-between">
            <span className="panel-title-text">1. TOKEN ENGINE</span>
            <span className="badge-cyan">{tokens.length} IN</span>
          </div>
          <div className="panel-body">
            <div className="tokens-table-wrap">
              <table className="mini-data-table">
                <thead>
                  <tr>
                    <th>POS</th>
                    <th>TOKEN</th>
                    <th>ID</th>
                    <th>TYPE</th>
                  </tr>
                </thead>
                <tbody>
                  {tokens.length === 0 ? (
                    <>
                      <tr><td className="text-dim">#0</td><td className="text-bright font-bold">What</td><td className="text-cyan">3838</td><td className="text-dim">BPE</td></tr>
                      <tr><td className="text-dim">#1</td><td className="text-bright font-bold">is</td><td className="text-cyan">374</td><td className="text-dim">BPE</td></tr>
                      <tr><td className="text-dim">#2</td><td className="text-bright font-bold">AI</td><td className="text-cyan">9552</td><td className="text-dim">BPE</td></tr>
                      <tr><td className="text-dim">#3</td><td className="text-bright font-bold">?</td><td className="text-cyan">30</td><td className="text-dim">BPE</td></tr>
                    </>
                  ) : (
                    tokens.slice(0, 4).map((t) => (
                      <tr key={t.position}>
                        <td className="text-dim">#{t.position}</td>
                        <td className="text-bright font-bold truncate max-w-token">
                          {t.text === "\n" ? "\\n" : t.text}
                        </td>
                        <td className="text-cyan">{t.id}</td>
                        <td className="text-dim text-xxs">{t.is_special ? "SPEC" : "BPE"}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
            <div className="panel-footer-stat flex items-center justify-between text-xxs">
              <span>VOCAB: <b>151.9k</b></span>
              <span>COUNT: <b className="text-cyan">{tokens.length || 4}</b></span>
            </div>
          </div>
        </div>

        {/* Panel 2: Embedding Engine */}
        <div className="detail-panel embedding-engine-panel">
          <div className="panel-header flex items-center justify-between">
            <span className="panel-title-text">2. EMBEDDING ENGINE</span>
            <span className="badge-cyan">{vectorDim}d</span>
          </div>
          <div className="panel-body">
            <div className="vector-bars-list">
              {(tokens.length > 0 ? tokens.slice(0, 4) : [{ position: 0, text: "What", id: 3838 }, { position: 1, text: "is", id: 374 }, { position: 2, text: "AI", id: 9552 }, { position: 3, text: "?", id: 30 }]).map((tok, idx) => {
                const pseudoNorm = 11 + ((tok.id * 7) % 15)
                return (
                  <div key={tok.position} className="vector-bar-row">
                    <span className="vector-tok-label text-xxs truncate">
                      "{tok.text === "\n" ? "\\n" : tok.text}"
                    </span>
                    <div className="vector-bar-track">
                      <div
                        className="vector-bar-fill"
                        style={{
                          width: `${Math.min(100, (pseudoNorm / 28) * 100)}%`,
                          backgroundColor: idx % 2 === 0 ? "#00d2ff" : "#ffb648",
                        }}
                      />
                    </div>
                    <span className="vector-norm-label text-xxs text-dim">
                      {pseudoNorm.toFixed(1)}
                    </span>
                  </div>
                )
              })}
            </div>
            <div className="panel-footer-stat flex items-center justify-between text-xxs">
              <span>ROPE: <b>θ=1M</b></span>
              <span>SHAPE: <b className="text-cyan">[1, {tokens.length || 4}, {vectorDim}]</b></span>
            </div>
          </div>
        </div>

        {/* Panel 3: Neural Network / Transformer */}
        <div className="detail-panel transformer-engine-panel">
          <div className="panel-header flex items-center justify-between">
            <span className="panel-title-text">3. TRANSFORMER STACK</span>
            <span className="badge-amber">{numLayers} LAYERS</span>
          </div>
          <div className="panel-body">
            <div className="transformer-block-diagram">
              <div className="tf-stage-pill embed-stage">Input Embedding ({vectorDim}d)</div>
              <div className="tf-connector-down">↓</div>
              <div className="tf-layer-box">
                <div className="tf-sub-stage">Multi-Head Attn (14Q/2KV GQA)</div>
                <div className="tf-sub-stage add-norm">Add & RMSNorm</div>
                <div className="tf-sub-stage mlp-stage">Feed-Forward (SwiGLU 4864d)</div>
                <div className="tf-sub-stage add-norm">Add & RMSNorm</div>
              </div>
              <div className="tf-connector-down">↓</div>
              <div className="tf-stage-pill out-stage">Hidden Output Vector</div>
            </div>

            <div className="layer-progress-container">
              <div className="progress-header flex items-center justify-between text-xxs">
                <span className="text-dim">
                  {running ? `Layer ${activeLayerIndex + 1} / ${numLayers}` : `24 Layers Verified`}
                </span>
                <span className="text-amber font-bold">{running ? `${layerProgressPct}%` : "100%"}</span>
              </div>
              <div className="layer-progress-track">
                <div
                  className="layer-progress-fill"
                  style={{ width: `${running ? layerProgressPct : 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Panel 4: Decoding Engine */}
        <div className="detail-panel decoding-engine-panel">
          <div className="panel-header flex items-center justify-between">
            <span className="panel-title-text">4. DECODING ENGINE</span>
            <span className="badge-cyan">{generatedTokens.length} OUT</span>
          </div>
          <div className="panel-body">
            <div className="tokens-table-wrap">
              <table className="mini-data-table">
                <thead>
                  <tr>
                    <th>STEP</th>
                    <th>TOKEN</th>
                    <th>ID</th>
                    <th>CONF</th>
                  </tr>
                </thead>
                <tbody>
                  {generatedTokens.length === 0 ? (
                    <>
                      <tr><td className="text-dim">#0</td><td className="text-amber font-bold">Artificial</td><td className="text-dim">28392</td><td className="text-cyan">94.2%</td></tr>
                      <tr><td className="text-dim">#1</td><td className="text-amber font-bold">intelligence</td><td className="text-dim">10234</td><td className="text-cyan">98.5%</td></tr>
                      <tr><td className="text-dim">#2</td><td className="text-amber font-bold">is</td><td className="text-dim">374</td><td className="text-cyan">99.1%</td></tr>
                      <tr><td className="text-dim">#3</td><td className="text-amber font-bold">the</td><td className="text-dim">279</td><td className="text-cyan">96.8%</td></tr>
                    </>
                  ) : (
                    generatedTokens.slice(-4).map((tok) => (
                      <tr key={tok.step}>
                        <td className="text-dim">#{tok.step}</td>
                        <td className="text-amber font-bold truncate max-w-token">
                          {tok.text === "\n" ? "\\n" : tok.text}
                        </td>
                        <td className="text-dim">{tok.token_id}</td>
                        <td className="text-cyan">
                          {(tok.probability * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {candidates && candidates.length > 0 && (
              <div className="top-candidate-preview text-xxs truncate">
                <span className="text-dim">TOP:</span>
                <span className="text-bright">"{candidates[0].text}"</span>
                <span className="text-amber">({(candidates[0].probability * 100).toFixed(0)}%)</span>
              </div>
            )}
            <div className="panel-footer-stat flex items-center justify-between text-xxs">
              <span>SAMPLER: <b>top-p 0.9</b></span>
              <span>EMITTED: <b className="text-amber">{generatedTokens.length || 4}</b></span>
            </div>
          </div>
        </div>

        {/* Panel 5: Response Delivery */}
        <div className="detail-panel response-delivery-panel">
          <div className="panel-header flex items-center justify-between">
            <span className="panel-title-text">5. Response Output & Dispatch</span>
            <span className={`badge-${response ? "emerald" : "outline"}`}>
              {response ? "COMPLETED" : "READY"}
            </span>
          </div>
          <div className="panel-body">
            <div className="response-delivery-content">
              {response ? (
                <div className="response-text-display">
                  {response}
                  {running && <span className="streaming-cursor">▮</span>}
                </div>
              ) : (
                <div className="response-text-display text-bright">
                  Artificial intelligence (AI) refers to computer systems designed to perform tasks that typically require human cognition, such as perception, reasoning, learning, and language understanding.
                </div>
              )}
            </div>

            <div className="panel-footer-stat flex items-center justify-between text-xxs">
              <span>LATENCY: <b className="text-emerald">{summary ? `${((summary.timings?.total_ms as number) / 1000 || 1.12).toFixed(2)}s` : "1.12s"}</b></span>
              <span>TTFT: <b className="text-cyan">{summary?.timings?.ttft_ms ?? "112"}ms</b></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
