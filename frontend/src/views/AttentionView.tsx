import { useEffect, useRef, useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"
import type { AttentionResponse, TensorVectorResponse } from "../types"
import ExternalObservationNotice from "../components/ExternalObservationNotice"

function drawHeatmap(canvas: HTMLCanvasElement, data: AttentionResponse, tokens: { position: number; text: string }[], selectedToken: number | null) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  const dpr = window.devicePixelRatio || 1
  const w = 760
  const h = 640
  canvas.width = w * dpr
  canvas.height = h * dpr
  canvas.style.width = `${w}px`
  canvas.style.height = `${h}px`
  ctx.scale(dpr, dpr)
  ctx.fillStyle = "#0f1524"
  ctx.fillRect(0, 0, w, h)

  const weights = data.weights
  const total = weights.length
  const mw = w - 130
  const mh = h - 80
  const ch = mh / Math.max(total, 1)
  const maxW = Math.max(...weights.map((x) => x.weight), 1e-6)

  for (let i = 0; i < total; i++) {
    const v = weights[i]?.weight ?? 0
    const a = Math.min(1, v / maxW)
    ctx.fillStyle = `rgba(76, ${Math.round(140 + a * 80)}, 255, ${0.08 + a * 0.92})`
    ctx.fillRect(120, 50 + i * ch, Math.max(mw * v / maxW, 1), Math.max(ch - 1, 1))
  }

  ctx.fillStyle = "#7c89a8"
  ctx.font = "10px monospace"
  const step = Math.max(1, Math.floor(total / 24))
  for (let i = 0; i < total; i += step) {
    const sourcePosition = weights[i].token_index
    const t = tokens.find((token) => token.position === sourcePosition)
    ctx.textAlign = "right"
    ctx.fillText(t ? (t.text.length > 8 ? t.text.slice(0, 7) + "…" : t.text) : `#${sourcePosition}`, 112, 54 + i * ch + 3)
    ctx.textAlign = "left"
    ctx.fillText(`${sourcePosition === selectedToken ? "▶ " : ""}${(weights[i].weight * 100).toFixed(1)}%`, 124, 54 + i * ch + 3)
  }

  ctx.fillStyle = "#4cc2ff"
  ctx.font = "bold 11px monospace"
  ctx.textAlign = "left"
  ctx.fillText(
    `ATTENTION WEIGHTS · layer ${data.layer + 1} · head ${data.head} · row = token pos ${data.position} · real weights`,
    14,
    20,
  )
  if (selectedToken !== null) {
    const a = weights.find((item) => item.token_index === selectedToken)?.weight ?? 0
    ctx.fillText(`weight(pos→${selectedToken}) = ${(a * 100).toFixed(1)}%`, 14, 36)
  }
}

export default function AttentionView() {
  const sessionId = useBrain((s) => s.sessionId)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectedHead = useBrain((s) => s.selectedHead)
  const selectLayer = useBrain((s) => s.selectLayer)
  const selectHead = useBrain((s) => s.selectHead)
  const selectToken = useBrain((s) => s.selectToken)
  const selectedToken = useBrain((s) => s.selectedToken)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const model = useBrain((s) => s.model)
  const [data, setData] = useState<AttentionResponse | null>(null)
  const [queryFlow, setQueryFlow] = useState<TensorVectorResponse | null>(null)
  const [error, setError] = useState("")
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  const nativeCapture = model?.extra?.runtime === "llama.cpp"
  const attentionByStep = useBrain((s) => s.attentionLinksByStep)
  const running = useBrain((s) => s.running)
  const [queryPosition, setQueryPosition] = useState<number | null>(null)
  useEffect(() => { setQueryPosition(null) }, [sessionId, selectedToken])

  const layer = selectedLayer ?? 0
  const head = selectedHead ?? 0
  const selectedOutput = generatedTokens.find((token) => token.position === selectedToken)
  const selectedPosition = selectedOutput
    ? Math.max(0, selectedOutput.position - 1)
    : selectedToken ?? Math.max(0, tokens.length - 1)
  const position = queryPosition ?? (nativeCapture ? Math.max(tokens.length - 1, selectedPosition) : selectedPosition)
  const captureStep = position < tokens.length ? 0 : position - tokens.length + 1
  const attentionReady = attentionByStep[captureStep]?.[layer]
  const allTokens = [
    ...tokens.map((t) => ({ position: t.position, text: t.text })),
    ...generatedTokens.map((t) => ({ position: t.position, text: t.text })),
  ]

  useEffect(() => {
    let cancelled = false
    if (!sessionId) {
      setError("run an inference first")
      return
    }
    setError("")
    setData(null)
    setQueryFlow(null)
    if (nativeCapture && running && !attentionReady) return
    api.attention(sessionId, layer, head, position)
      .then((result) => { if (!cancelled) setData(result) })
      .catch((e) => {
        if (cancelled) return
        setError(String(e))
        setData(null)
      })
    api.qkv(sessionId, layer, "q", position, 8, head)
      .then((result) => { if (!cancelled) setQueryFlow(result) })
      .catch(() => { if (!cancelled) setQueryFlow(null) })
    return () => { cancelled = true }
  }, [sessionId, layer, head, position, attentionReady, nativeCapture, running])

  useEffect(() => {
    if (data && canvasRef.current) drawHeatmap(canvasRef.current, data, allTokens, selectedToken)
  }, [data, allTokens, selectedToken])

  const selectHeatmapPosition = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data?.weights.length) return
    const rect = event.currentTarget.getBoundingClientRect()
    const y = (event.clientY - rect.top) * (640 / rect.height)
    const rowHeight = (640 - 80) / data.weights.length
    const row = Math.max(0, Math.min(data.weights.length - 1, Math.floor((y - 50) / rowHeight)))
    if (y >= 50 && y <= 610) selectToken(data.weights[row].token_index)
  }

  if (inspectionMode === "limited") return <ExternalObservationNotice />

  return (
    <div className="heatmap-wrap">
      <div className="heatmap-controls">
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
          layer
          <select value={layer} onChange={(e) => selectLayer(Number(e.target.value))} disabled={!model}>
            {Array.from({ length: model?.num_layers ?? 0 }, (_, i) => (
              <option key={i} value={i}>
                {i + 1}
              </option>
            ))}
          </select>
        </label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
          head
          <select value={head} onChange={(e) => selectHead(Number(e.target.value))} disabled={!model}>
            {Array.from({ length: model?.num_attention_heads ?? 0 }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
          row (query token)
          <select value={position} onChange={(e) => setQueryPosition(Number(e.target.value))}>
            {allTokens.filter((t) => !nativeCapture || Boolean(attentionByStep[t.position < tokens.length ? 0 : t.position - tokens.length + 1]?.[layer]) && t.position >= tokens.length - 1).map((t) => (
              <option key={t.position} value={t.position}>
                pos {t.position}: {t.text.length > 18 ? t.text.slice(0, 16) + "…" : t.text}
              </option>
            ))}
          </select>
        </label>
        {data && (
          <span className="mono" style={{ fontSize: 10.5, color: "var(--muted)" }}>
            L2 {data.stats.l2_norm.toFixed(3)} · mean {data.stats.mean.toFixed(4)} · {data.is_full_matrix ? "full matrix" : "captured query row"}
          </span>
        )}
      </div>
      {error && <div className="err-box">{error}</div>}
      <canvas ref={canvasRef} className="heatmap" onClick={selectHeatmapPosition} aria-label="Attention heatmap; click a row to select its source token" title="Click a row to select its source token" style={{ cursor: "pointer" }} />
      <section className="attention-explainer" aria-label="Attention calculation and results">
        <header>
          <strong>WHAT THIS ATTENTION ROW DOES</strong>
          <span>Layer {layer + 1} · head {head} · input position {position}{selectedOutput ? ` → predicts “${selectedOutput.text}”` : ""}</span>
        </header>
        <p><b>Input:</b> the selected prompt/generated-context position supplies a query vector Q. This query asks which earlier token representations are useful for this prediction.</p>
        <p><b>Calculation:</b> each earlier key K is scored with QKᵀ/√dₖ. Softmax turns those scores into the real percentages shown in the heatmap. Causal masking prevents the query from looking at future tokens.</p>
        <p><b>Output:</b> each percentage weights its token's value vector V; the weighted values are combined into the context passed through the attention output projection and onward through the layer.</p>
        {data && <div className="attention-result-summary">
          <span><b>Captured distribution:</b> {data.weights.length} source positions · sum {data.stats.mean * data.weights.length ? data.weights.reduce((sum, item) => sum + item.weight, 0).toFixed(4) : "0.0000"}</span>
          <span><b>Highest-weight source:</b> {(() => {
            const highest = [...data.weights].sort((a, b) => b.weight - a.weight)[0]
            const token = highest && allTokens.find((item) => item.position === highest.token_index)
            return highest ? `position ${highest.token_index}${token ? ` “${token.text}”` : ""} · ${(highest.weight * 100).toFixed(2)}%` : "none captured"
          })()}</span>
          {queryFlow?.query_key_matches?.slice(0, 5).map((match) => {
            const source = allTokens.find((item) => item.position === match.token_index)
            const contribution = queryFlow.value_contributions?.find((item) => item.token_index === match.token_index)
            return <span key={match.token_index}><b>Position {match.token_index}</b> {source ? `“${source.text}”` : ""} · QKᵀ/√dₖ {match.scaled_score.toFixed(3)} · softmax {match.attention_probability.toFixed(4)} · weighted |V| {contribution?.contribution_norm.toFixed(3) ?? "—"}</span>
          })}
        </div>}
        {!data && !error && <p className="muted">Loading the captured attention row for this input position…</p>}
        {error && <p className="muted">This position has no retained attention capture. Select an available query row from this run.</p>}
      </section>
    </div>
  )
}
