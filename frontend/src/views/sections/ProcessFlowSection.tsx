import { useMemo } from "react"
import { useBrain } from "../../store/useBrainStore"

export default function ProcessFlowSection() {
  const telemetry = useBrain((s) => s.telemetry)
  const prompt = useBrain((s) => s.prompt)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const running = useBrain((s) => s.running)
  const currentStep = useBrain((s) => s.currentStep)
  const model = useBrain((s) => s.model)
  const summary = useBrain((s) => s.summary)
  const pca = useBrain((s) => s.pca)

  const isComplete = !running && generatedTokens.length > 0
  const totalTokens = tokens.length + generatedTokens.length
  const maxContext = model?.context_length ?? "—"
  const vectorDim = model?.hidden_size ?? "—"
  const numLayers = model?.num_layers ?? "—"
  const numParams = model?.num_params ? `${(model.num_params / 1e6).toFixed(0)}M` : "—"

  // 4x3 mini embedding heatmap
  const embeddingCells = useMemo(() => {
    if (pca?.tokens?.length) {
      const maximum = Math.max(...pca.tokens.map((token) => token.norm), 1e-6)
      return pca.tokens.slice(0, 12).map((t) => t.norm / maximum)
    }
    return []
  }, [pca])

  return (
    <div className="process-flow-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="section-title-tag">How Brain-OS 3.0 works — full process flow</span>
          <span className="badge-outline text-xxs">REAL-TIME HOOK SEQUENCE</span>
        </div>
        <div className="flow-status-badge">
          <span className={`status-dot ${running ? "engine-active" : isComplete ? "ok" : "dim"}`} />
          <span className="text-xxs">
            {running ? `STEP ${currentStep} COMPUTING` : isComplete ? "PIPELINE COMPLETED" : "STANDBY"}
          </span>
        </div>
      </div>

      {/* 6 Sequential Step Cards Left to Right with Arrows */}
      <div className="flow-cards-strip">
        {/* Step 1: User Input */}
        <div className={`flow-step-card ${prompt ? "active-card" : ""}`}>
          <div className="step-card-header flex items-center justify-between">
            <span className="step-num">01</span>
            <span className="step-badge">INPUT</span>
          </div>
          <div className="step-card-title">User Input</div>
          <div className="step-card-body">
            <div className="user-input-snippet">
              <span className="user-avatar-tiny">🧑‍💻</span>
              <span className="user-prompt-text text-bright truncate" title={prompt || "No input"}>
                {prompt ? `"${prompt}"` : "No input"}
              </span>
            </div>
            <div className="step-meta text-xxs text-dim">
              <span title="Includes system, history and chat-template markers">USER TEXT TOKENS: {telemetry?.tokens.user_text_tokens ?? "—"} · MODEL INPUT TOKENS: <b className="text-cyan">{tokens.length}</b></span>
            </div>
          </div>
        </div>

        <div className="flow-arrow-sep">➔</div>

        {/* Step 2: Tokenization */}
        <div className={`flow-step-card ${tokens.length > 0 ? "active-card" : ""}`}>
          <div className="step-card-header flex items-center justify-between">
            <span className="step-num">02</span>
            <span className="step-badge">BPE</span>
          </div>
          <div className="step-card-title">Tokenization</div>
          <div className="step-card-body">
            <div className="chips-preview-row">
              {tokens.length > 0 ? (
                tokens.slice(0, 4).map((tok) => (
                  <span key={tok.position} className="flow-tok-chip prompt truncate" title={`ID: ${tok.id}`}>
                    {tok.text === "\n" ? "\\n" : tok.text}
                  </span>
                ))
              ) : (
                <span className="text-dim">No captured tokens</span>
              )}
            </div>
            <div className="step-meta text-xxs text-dim">
              <span>TOTAL: <b className="text-cyan">{totalTokens}</b> / {maxContext}</span>
            </div>
          </div>
        </div>

        <div className="flow-arrow-sep">➔</div>

        {/* Step 3: Embedding */}
        <div className={`flow-step-card ${pca ? "active-card" : ""}`}>
          <div className="step-card-header flex items-center justify-between">
            <span className="step-num">03</span>
            <span className="step-badge">VECTOR</span>
          </div>
          <div className="step-card-title">Embedding</div>
          <div className="step-card-body">
            <div className="embed-grid-mini">
              {embeddingCells.map((val, i) => (
                <div
                  key={i}
                  className="embed-grid-cell"
                  style={{
                    backgroundColor: `rgba(0, 210, 255, ${val})`,
                    boxShadow: val > 0.7 ? "0 0 3px #00d2ff" : "none",
                  }}
                  title={`Token ${pca?.tokens[i]?.position}: norm ${pca?.tokens[i]?.norm.toFixed(3)}`}
                />
              ))}
            </div>
            <div className="step-meta text-xxs text-dim">
              <span>VECTOR DIM: <b className="text-cyan">{vectorDim}</b></span>
            </div>
          </div>
        </div>

        <div className="flow-arrow-sep">➔</div>

        {/* Step 4: Neural Network Inference */}
        <div className={`flow-step-card ${running ? "active-card pulse-card" : ""}`}>
          <div className="step-card-header flex items-center justify-between">
            <span className="step-num">04</span>
            <span className="step-badge">INFER</span>
          </div>
          <div className="step-card-title">Neural Network</div>
          <div className="step-card-body">
            <div className="node-tiers-diagram">
              <div className="tier-col">
                <span className="tier-dot input-dot" />
                <span className="tier-dot input-dot" />
              </div>
              <div className="tier-connector" />
              <div className={`tier-col hidden-tier ${running ? "pulsing-tier" : ""}`}>
                <span className="tier-dot hidden-dot" />
                <span className="tier-dot hidden-dot" />
                <span className="tier-dot hidden-dot" />
              </div>
              <div className="tier-connector" />
              <div className="tier-col">
                <span className="tier-dot output-dot" />
                <span className="tier-dot output-dot" />
              </div>
            </div>
            <div className="step-meta text-xxs text-dim">
              <span>LAYERS: <b className="text-amber">{numLayers}</b> · <b className="text-bright">{numParams}</b></span>
            </div>
          </div>
        </div>

        <div className="flow-arrow-sep">➔</div>

        {/* Step 5: Response Generation */}
        <div className={`flow-step-card ${generatedTokens.length > 0 ? "active-card" : ""}`}>
          <div className="step-card-header flex items-center justify-between">
            <span className="step-num">05</span>
            <span className="step-badge">DECODE</span>
          </div>
          <div className="step-card-title">Response Gen</div>
          <div className="step-card-body">
            <div className="chips-preview-row">
              {generatedTokens.length > 0 ? (
                generatedTokens.slice(-3).map((tok) => (
                  <span key={tok.step} className="flow-tok-chip generated truncate" title={`p: ${(tok.probability * 100).toFixed(0)}%`}>
                    {tok.text === "\n" ? "\\n" : tok.text}
                  </span>
                ))
              ) : (
                <span className="text-dim text-xxs">{running ? "Decoding..." : "Ready to emit"}</span>
              )}
              {running && <span className="streaming-cursor">▮</span>}
            </div>
            <div className="step-meta text-xxs text-dim">
              <span>OUTPUT TOK: <b className="text-amber">{generatedTokens.length}</b></span>
            </div>
          </div>
        </div>

        <div className="flow-arrow-sep">➔</div>

        {/* Step 6: Output to User */}
        <div className={`flow-step-card ${isComplete ? "active-card success-card" : ""}`}>
          <div className="step-card-header flex items-center justify-between">
            <span className="step-num">06</span>
            <span className="step-badge">OUTPUT</span>
          </div>
          <div className="step-card-title">Output to User</div>
          <div className="step-card-body">
            <div className="delivery-result-badge">
              <span className={`result-check ${isComplete ? "check-ok" : "check-wait"}`}>
                {isComplete ? "✓" : "⏳"}
              </span>
              <span className="result-text text-xxs">
                {isComplete ? "Response completed" : running ? "Streaming..." : "Standby"}
              </span>
            </div>
            <div className="step-meta text-xxs text-dim">
              {summary ? (
                <span>LATENCY: <b className="text-emerald">{((summary.timings?.total_ms as number) / 1000).toFixed(2)}s</b></span>
              ) : (
                <span>LATENCY: <b className="text-emerald">—</b></span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
