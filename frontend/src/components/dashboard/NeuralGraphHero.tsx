import { useEffect, useRef, useState, useMemo } from "react"
import { useBrain } from "../../store/useBrainStore"

export default function NeuralGraphHero() {
  const model = useBrain((s) => s.model)
  const running = useBrain((s) => s.running)
  const currentStep = useBrain((s) => s.currentStep)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const sessionId = useBrain((s) => s.sessionId)
  const layers = useBrain((s) => s.layers)
  const selectedToken = useBrain((s) => s.selectedToken)
  const selectToken = useBrain((s) => s.selectToken)
  const summary = useBrain((s) => s.summary)
  const candidates = useBrain((s) => s.candidates)
  const attentionLinks = useBrain((s) => s.attentionLinks)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [frame, setFrame] = useState(0)

  const numTokens = tokens.length + generatedTokens.length
  const activeStep = running ? currentStep : generatedTokens.length
  const capturedEdges = Object.values(attentionLinks).reduce((count, links) => count + links.length, 0)
  const attentionSpan = (start: number, end: number) => {
    const values = Object.values(attentionLinks).flat().filter((link) => link.head >= start && link.head <= end).map((link) => link.weight)
    return values.length ? (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2) : "—"
  }
  const hiddenSize = model?.hidden_size
  const kvWidth = model?.num_kv_heads && model?.head_dim ? model.num_kv_heads * model.head_dim : undefined
  const activeLayerState = Object.values(layers).find((layer) => layer.step === currentStep)

  const allTokens = useMemo(() => {
    const combined = [
      ...tokens.map((t) => ({ text: t.text, id: t.id, position: t.position, generated: false })),
      ...generatedTokens.map((t) => ({ text: t.text, id: t.token_id, position: t.position, generated: true })),
    ]
    if (combined.length <= 16) return combined
    return [...combined.slice(0, 4), ...combined.slice(-12)]
  }, [tokens, generatedTokens])

  useEffect(() => {
    let animId: number
    let f = 0
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const render = () => {
      f++
      if (f % 2 === 0) setFrame(f)

      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)

      ctx.strokeStyle = "rgba(20, 32, 54, 0.4)"
      ctx.lineWidth = 1
      const step = 32
      for (let x = 0; x < w; x += step) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, h)
        ctx.stroke()
      }
      for (let y = 0; y < h; y += step) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        ctx.stroke()
      }

      const pulseSpeed = running ? 0.025 : 0.008
      const particleCount = running ? 48 : 16

      for (let i = 0; i < particleCount; i++) {
        const t = ((f * pulseSpeed + i / particleCount) % 1)
        const x1 = w * 0.18
        const y1 = h * (0.15 + (i % 5) * 0.16)
        const x2 = w * 0.48
        const y2 = h * (0.12 + (i % 12) * 0.065)

        const cx1 = x1 + (x2 - x1) * 0.5
        const cy1 = y1
        const cx2 = x1 + (x2 - x1) * 0.5
        const cy2 = y2

        const omt = 1 - t
        const px = omt * omt * omt * x1 + 3 * omt * omt * t * cx1 + 3 * omt * t * t * cx2 + t * t * t * x2
        const py = omt * omt * omt * y1 + 3 * omt * omt * t * cy1 + 3 * omt * t * t * cy2 + t * t * t * y2

        ctx.beginPath()
        ctx.arc(px, py, running ? 2.5 : 1.5, 0, Math.PI * 2)
        ctx.fillStyle = running ? "rgba(76, 194, 255, 0.85)" : "rgba(76, 194, 255, 0.35)"
        ctx.shadowColor = "#4cc2ff"
        ctx.shadowBlur = running ? 8 : 4
        ctx.fill()
        ctx.shadowBlur = 0

        const rx1 = w * 0.52
        const ry1 = y2
        const rx2 = w * 0.82
        const ry2 = h * (0.12 + (i % 6) * 0.14)

        const rcx1 = rx1 + (rx2 - rx1) * 0.5
        const rcy1 = ry1
        const rcx2 = rx1 + (rx2 - rx1) * 0.5
        const rcy2 = ry2

        const rpx = omt * omt * omt * rx1 + 3 * omt * omt * t * rcx1 + 3 * omt * t * t * rcx2 + t * t * t * rx2
        const rpy = omt * omt * omt * ry1 + 3 * omt * omt * t * rcy1 + 3 * omt * t * t * rcy2 + t * t * t * ry2

        ctx.beginPath()
        ctx.arc(rpx, rpy, running ? 2.2 : 1.2, 0, Math.PI * 2)
        ctx.fillStyle = running ? "rgba(255, 182, 72, 0.85)" : "rgba(255, 182, 72, 0.3)"
        ctx.shadowColor = "#ffb648"
        ctx.shadowBlur = running ? 7 : 3
        ctx.fill()
        ctx.shadowBlur = 0
      }

      animId = requestAnimationFrame(render)
    }

    render()
    return () => cancelAnimationFrame(animId)
  }, [running])

  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.parentElement?.getBoundingClientRect()
      if (rect) {
        canvas.width = rect.width
        canvas.height = rect.height
      }
    }
    handleResize()
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  return (
    <div className="neural-graph-hero">
      <div className="neural-ribbon">
        <div className="ribbon-segment">
          <span className="ribbon-label">GRAPH:</span>
          <span className="ribbon-val cyan">QWEN2.5-0.5B-INSTRUCT</span>
        </div>
        <div className="ribbon-divider">|</div>
        <div className="ribbon-segment">
          <span className="ribbon-label">NODES:</span>
          <span className="ribbon-val">{model?.num_layers ?? "—"} LAYERS</span>
        </div>
        <div className="ribbon-divider">|</div>
        <div className="ribbon-segment">
          <span className="ribbon-label">EDGES:</span>
          <span className="ribbon-val amber">{capturedEdges ? `${capturedEdges} CAPTURED` : "—"}</span>
        </div>
        <div className="ribbon-divider">|</div>
        <div className="ribbon-segment">
          <span className="ribbon-label">BUS READERS:</span>
          <span className="ribbon-val">{numTokens || 14}</span>
        </div>
        <div className="ribbon-divider">|</div>
        <div className="ribbon-segment">
          <span className="ribbon-label">DEVICE:</span>
          <span className="ribbon-val green">APPLE SILICON MPS</span>
        </div>
        <div className="ribbon-divider">|</div>
        <div className="ribbon-segment">
          <span className="ribbon-label">T:</span>
          <span className="ribbon-val">{summary?.timings?.tokens_per_second ? `${summary.timings.tokens_per_second} t/s` : "1.17"}</span>
        </div>
        <div className="ribbon-divider">|</div>
        <div className="ribbon-segment">
          <span className="ribbon-label">FRAME:</span>
          <span className="ribbon-val mono">{frame}</span>
        </div>
      </div>

      <div className="neural-viewport">
        <canvas ref={canvasRef} className="neural-canvas" />

        <svg className="neural-svg" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="cyanGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22c55e" stopOpacity="0.8" />
              <stop offset="50%" stopColor="#4cc2ff" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#9d6bff" stopOpacity="0.8" />
            </linearGradient>
            <linearGradient id="amberGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#4cc2ff" stopOpacity="0.7" />
              <stop offset="60%" stopColor="#ffb648" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.9" />
            </linearGradient>
          </defs>

          {allTokens.map((_t, idx) => {
            const busY = 12 + idx * 5.2
            const leftModY = 16 + (idx % 5) * 16
            const rightModY = 12 + (idx % 6) * 14

            return (
              <g key={`spline-${idx}`}>
                <path
                  d={`M 19% ${leftModY}% C 33% ${leftModY}%, 38% ${busY}%, 47% ${busY}%`}
                  fill="none"
                  stroke={running ? "url(#cyanGradient)" : "rgba(76, 194, 255, 0.22)"}
                  strokeWidth={running && idx === activeStep % allTokens.length ? "2" : "1"}
                  strokeDasharray={running ? "4 2" : "none"}
                />
                <path
                  d={`M 53% ${busY}% C 62% ${busY}%, 68% ${rightModY}%, 81% ${rightModY}%`}
                  fill="none"
                  stroke={running ? "url(#amberGradient)" : "rgba(255, 182, 72, 0.2)"}
                  strokeWidth={running && idx === activeStep % allTokens.length ? "2" : "1"}
                  strokeDasharray={running ? "4 2" : "none"}
                />
              </g>
            )
          })}
        </svg>

        <div className="neural-nodes-layer">
          <div className="neural-column left-modules">
            <div className={`neural-card ${running ? "active-glow green" : ""}`}>
              <div className="card-tag green">input_prompt</div>
              <div className="card-fields">
                <div className="field-row"><span>tokens</span><span className="val green">{tokens.length}</span></div>
                <div className="field-row"><span>vocab</span><span className="val">{model?.vocab_size?.toLocaleString() ?? "—"}</span></div>
                <div className="field-row"><span>chat_template</span><span className="val">qwen</span></div>
              </div>
              <div className="card-port right" />
            </div>

            <div className={`neural-card ${running ? "active-glow cyan" : ""}`}>
              <div className="card-tag cyan">embed_tokens</div>
              <div className="card-fields">
                <div className="field-row"><span>hidden_dim</span><span className="val cyan">{hiddenSize ?? "—"}</span></div>
                <div className="field-row"><span>pca_project</span><span className="val">3D_norm</span></div>
                <div className="field-row"><span>dtype</span><span className="val">float32</span></div>
              </div>
              <div className="card-port right" />
            </div>

            <div className={`neural-card ${running ? "active-glow amber" : ""}`}>
              <div className="card-tag amber">rope_embeddings</div>
              <div className="card-fields">
                <div className="field-row"><span>theta</span><span className="val amber">{String(model?.extra?.rope_theta ?? "—")}</span></div>
                <div className="field-row"><span>max_pos</span><span className="val">{model?.max_position_embeddings ?? "—"}</span></div>
                <div className="field-row"><span>scaling</span><span className="val">{String(model?.extra?.rope_scaling ?? "—")}</span></div>
              </div>
              <div className="card-port right" />
            </div>

            <div className={`neural-card ${running ? "active-glow purple" : ""}`}>
              <div className="card-tag purple">kv_cache_buffer</div>
              <div className="card-fields">
                <div className="field-row"><span>kv_heads</span><span className="val purple">2 (GQA)</span></div>
                <div className="field-row"><span>head_dim</span><span className="val">64</span></div>
                <div className="field-row"><span>cache_mode</span><span className="val">dynamic</span></div>
              </div>
              <div className="card-port right" />
            </div>

            <div className={`neural-card ${running ? "active-glow blue" : ""}`}>
              <div className="card-tag blue">causal_mask</div>
              <div className="card-fields">
                <div className="field-row"><span>is_causal</span><span className="val blue">true</span></div>
                <div className="field-row"><span>sliding_window</span><span className="val">32k</span></div>
                <div className="field-row"><span>attn_impl</span><span className="val">eager</span></div>
              </div>
              <div className="card-port right" />
            </div>
          </div>

          <div className="neural-column center-bus">
            <div className="bus-container">
              <div className="bus-header">
                <div className="bus-title">SHARED CONTEXT BUS</div>
                <div className="bus-badge">{numTokens} TOKENS</div>
              </div>

              <div className="bus-slots-list">
                {allTokens.length === 0 ? (
                  <div className="empty-bus-hint">
                    <span className="mono">00 [BUS_STANDBY]</span>
                    <span className="sub-hint">Submit prompt to activate context bus</span>
                  </div>
                ) : (
                  allTokens.map((t, idx) => {
                    const isSelected = selectedToken === t.position
                    const isGenerating = running && idx === (activeStep % allTokens.length)
                    return (
                      <div
                        key={`bus-slot-${t.position}-${idx}`}
                        className={`bus-slot ${t.generated ? "generated" : "prompt"} ${isGenerating ? "generating" : ""} ${isSelected ? "selected" : ""}`}
                        onClick={() => selectToken(isSelected ? null : t.position)}
                        title={`Token #${t.position} (ID ${t.id})`}
                      >
                        <span className="slot-idx mono">{String(t.position).padStart(2, "0")}</span>
                        <div className="slot-node-dot" />
                        <span className="slot-text mono">
                          {t.text.replace(/Ġ/g, " ").replace(/Ċ/g, "\\n") || "•"}
                        </span>
                        <span className="slot-id mono">{t.id}</span>
                      </div>
                    )
                  })
                )}
              </div>
            </div>
          </div>

          <div className="neural-column right-modules">
            <div className={`neural-card ${running ? "active-glow blue" : ""}`}>
              <div className="card-port left" />
              <div className="card-tag blue">qkv_self_attention</div>
              <div className="card-fields">
                <div className="field-row"><span>q_proj</span><span className="val blue">{hiddenSize ?? "—"} → {hiddenSize ?? "—"}</span></div>
                <div className="field-row"><span>k_proj</span><span className="val">{hiddenSize ?? "—"} → {kvWidth ?? "—"}</span></div>
                <div className="field-row"><span>v_proj</span><span className="val">{hiddenSize ?? "—"} → {kvWidth ?? "—"}</span></div>
              </div>
            </div>

            <div className="scout-cluster">
              <div className={`neural-card mini ${running ? "active-glow amber" : ""}`}>
                <div className="card-port left" />
                <div className="card-tag amber">heads_0_3</div>
                <div className="mini-row"><span>mean</span><span className="val amber">{attentionSpan(0, 3)}</span></div>
              </div>
              <div className={`neural-card mini ${running ? "active-glow amber" : ""}`}>
                <div className="card-port left" />
                <div className="card-tag amber">heads_4_7</div>
                <div className="mini-row"><span>mean</span><span className="val amber">{attentionSpan(4, 7)}</span></div>
              </div>
              <div className={`neural-card mini ${running ? "active-glow amber" : ""}`}>
                <div className="card-port left" />
                <div className="card-tag amber">heads_8_13</div>
                <div className="mini-row"><span>mean</span><span className="val amber">{attentionSpan(8, 13)}</span></div>
              </div>
            </div>

            <div className={`neural-card ${running ? "active-glow amber" : ""}`}>
              <div className="card-port left" />
              <div className="card-tag amber">swiglu_mlp_block</div>
              <div className="card-fields">
                <div className="field-row"><span>gate_proj</span><span className="val amber">4,864</span></div>
                <div className="field-row"><span>up_proj</span><span className="val">4,864</span></div>
                <div className="field-row"><span>down_proj</span><span className="val">896</span></div>
              </div>
            </div>

            <div className={`neural-card ${running ? "active-glow cyan" : ""}`}>
              <div className="card-port left" />
              <div className="card-tag cyan">rmsnorm_residual</div>
              <div className="card-fields">
                <div className="field-row"><span>eps</span><span className="val cyan">1e-06</span></div>
                <div className="field-row"><span>layers</span><span className="val">{model?.num_layers ?? "—"}</span></div>
                <div className="field-row"><span>active_norm</span><span className="val">{activeLayerState?.norm?.toFixed(2) ?? "—"}</span></div>
              </div>
            </div>

            <div className={`neural-card ${running ? "active-glow green" : ""}`}>
              <div className="card-port left" />
              <div className="card-tag green">lm_head_logits</div>
              <div className="card-fields">
                <div className="field-row"><span>vocab_size</span><span className="val green">{model?.vocab_size?.toLocaleString() ?? "—"}</span></div>
                <div className="field-row"><span>top_candidate</span><span className="val green">{candidates?.[0]?.text ?? "—"}</span></div>
                <div className="field-row"><span>top_prob</span><span className="val">{candidates?.[0]?.probability ? `${(candidates[0].probability * 100).toFixed(1)}%` : "—"}</span></div>
              </div>
            </div>

            <div className={`neural-card ${running ? "active-glow purple" : ""}`}>
              <div className="card-port left" />
              <div className="card-tag purple">stream_writer</div>
              <div className="card-fields">
                <div className="field-row"><span>generated</span><span className="val purple">{generatedTokens.length}</span></div>
                <div className="field-row"><span>status</span><span className="val">{running ? "STREAMING" : "IDLE"}</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="neural-footer">
        <div className="footer-left">
          <span className="label">RUN:</span>
          <span className="val mono">{sessionId ? sessionId.slice(0, 12) : "STANDBY"}</span>
          <span className="muted">· graph.load: nodes={model?.num_layers ?? "—"} edges={capturedEdges || "—"}</span>
        </div>

        <div className="footer-center">
          <span className="label">DISPATCH:</span>
          <div className="dispatch-dots">
            {Array.from({ length: 8 }).map((_, i) => (
              <span
                key={i}
                className={`dot ${running ? (i <= (currentStep % 8) ? "active" : "standby") : "idle"}`}
                title={`Worker ${i + 1}`}
              />
            ))}
          </div>
          <span className="muted">parallel 8 · serial 1</span>
        </div>

        <div className="footer-right">
          <span className="label">GRAPH STAT:</span>
          <span className="muted">edges: <b>{capturedEdges || "—"}</b> · bus readers: <b>{numTokens || "—"}</b> · cost: <b className="green">{capturedEdges ? "LIVE" : "—"}</b></span>
        </div>
      </div>
    </div>
  )
}
