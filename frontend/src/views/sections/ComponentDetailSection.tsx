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
  const pca = useBrain((s) => s.pca)
  const activeLayer = useBrain((s) => s.activeLayer)
  const lastLogits = useBrain((s) => s.lastLogits)
  const telemetry = useBrain((s) => s.telemetry)
  const provider = useBrain((s) => s.provider)

  const numLayers = model?.num_layers ?? 0
  const vectorDim = model?.hidden_size ?? 0

  const activeLayerIndex = useMemo(() => {
    const layerKeys = Object.keys(layers).map(Number)
    if (running && activeLayer !== null) return activeLayer
    if (layerKeys.length === 0) return -1
    return Math.max(...layerKeys)
  }, [layers, running, activeLayer])

  const layerProgressPct = Math.min(100, Math.round(((activeLayerIndex + 1) / Math.max(1, numLayers)) * 100))

  return (
    <div className="component-detail-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="section-title-tag">DETAILED WORKING OF EACH COMPONENT</span>
          <span className="badge-outline text-xxs">5-COMPONENT INTERNAL STACK</span>
        </div>
        <span className="text-xxs text-dim">{provider === "ollama" ? "LLAMA.CPP GRAPH CAPTURE" : "LOCAL MODEL FORWARD HOOKS"}</span>
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
                    <tr><td colSpan={4} className="text-dim">No captured tokens</td></tr>
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
              <span>USER TEXT TOKENS: <b>{telemetry?.tokens.user_text_tokens ?? "—"}</b> · VOCAB: <b>{model?.vocab_size?.toLocaleString() ?? "—"}</b></span>
              <span title="Includes system instructions, history, chat-template markers and the current user message">MODEL INPUT TOKENS: <b className="text-cyan">{tokens.length}</b></span>
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
              {(pca?.tokens.slice(0, 4) ?? []).map((tok, idx) => {
                const maxNorm = Math.max(...(pca?.tokens.map((token) => token.norm) ?? [1]), 1e-6)
                return (
                  <div key={tok.position} className="vector-bar-row">
                    <span className="vector-tok-label text-xxs truncate">
                      "{tok.text === "\n" ? "\\n" : tok.text}"
                    </span>
                    <div className="vector-bar-track">
                      <div
                        className="vector-bar-fill"
                        style={{
                          width: `${Math.min(100, (tok.norm / maxNorm) * 100)}%`,
                          backgroundColor: idx % 2 === 0 ? "#00d2ff" : "#ffb648",
                        }}
                      />
                    </div>
                    <span className="vector-norm-label text-xxs text-dim">
                      norm {tok.norm.toFixed(3)}
                    </span>
                  </div>
                )
              })}
            </div>
            <div className="panel-footer-stat flex items-center justify-between text-xxs">
              <span>VALUES: <b>CAPTURED EMBEDDINGS</b></span>
              <span>SHAPE: <b className="text-cyan">{pca?.tensor ? JSON.stringify(pca.tensor.shape) : "Unavailable"}</b></span>
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
                <div className="text-xxs text-dim">Conceptual Decoder Block</div>
                <div className="tf-sub-stage add-norm">RMSNorm → Q / K / V Projection → RoPE(Q, K)</div>
                <div className="tf-sub-stage">{model?.num_attention_heads === model?.num_kv_heads ? "Multi-Head Attention" : "Grouped-Query Attention"} · {model?.num_attention_heads ?? "—"} Query Heads / {model?.num_kv_heads ?? "—"} Key Heads / {model?.num_kv_heads ?? "—"} Value Heads</div>
                <div className="tf-sub-stage add-norm">Output Projection → Residual Add → RMSNorm</div>
                <div className="tf-sub-stage mlp-stage">{model?.activation_function ?? "—"}-based MLP ({model?.intermediate_size ?? "—"}d) → Residual Add</div>
              </div>
              <div className="tf-connector-down">↓</div>
              <div className="tf-stage-pill out-stage">Hidden Output Vector</div>
            </div>

            <div className="layer-progress-container">
              <div className="progress-header flex items-center justify-between text-xxs">
                <span className="text-dim">
                  {running ? `Layer ${activeLayerIndex + 1} / ${numLayers}` : `${Object.keys(layers).length} / ${numLayers} Layers Captured`}
                </span>
                <span className="text-amber font-bold">{`${layerProgressPct}%`}</span>
              </div>
              <div className="layer-progress-track">
                <div
                  className="layer-progress-fill"
                  style={{ width: `${layerProgressPct}%` }}
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
                    <tr><td colSpan={4} className="text-dim">No generated tokens</td></tr>
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
              <span>SAMPLER: <b>{lastLogits ? `temp ${lastLogits.temperature} · top-p ${lastLogits.top_p}` : "—"}</b></span>
              <span>EMITTED: <b className="text-amber">{generatedTokens.length}</b></span>
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
              <div className="response-text-display text-bright">
                {running ? "Generating response · view it in the bottom Output panel" : response ? "Response available in the bottom Output panel" : "Response will appear in the bottom Output panel"}
              </div>
            </div>

            <div className="panel-footer-stat flex items-center justify-between text-xxs">
              <span>LATENCY: <b className="text-emerald">{typeof summary?.timings?.total_ms === "number" ? `${(summary.timings.total_ms / 1000).toFixed(2)}s` : "—"}</b></span>
              <span>TTFT: <b className="text-cyan">{summary?.timings?.ttft_ms ?? "—"}ms</b></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
