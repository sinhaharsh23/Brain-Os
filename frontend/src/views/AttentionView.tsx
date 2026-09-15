import { useEffect, useRef, useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"
import type { AttentionResponse } from "../types"
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
  const total = data.total_tokens
  const mw = w - 130
  const mh = h - 60
  const cw = mw / Math.max(total, 1)
  const ch = mh / Math.max(total, 1)
  const maxW = Math.max(...weights.map((x) => x.weight), 1e-6)

  for (let i = 0; i < total; i++) {
    const v = weights[i]?.weight ?? 0
    const a = Math.min(1, v / maxW)
    ctx.fillStyle = `rgba(76, ${Math.round(140 + a * 80)}, 255, ${0.08 + a * 0.92})`
    ctx.fillRect(120 + i * cw, 20 + i * ch, Math.max(cw - 1, 1), Math.max(ch - 1, 1))
  }

  ctx.fillStyle = "#7c89a8"
  ctx.font = "10px monospace"
  const step = Math.max(1, Math.floor(total / 24))
  for (let i = 0; i < total; i += step) {
    const t = tokens[i]
    ctx.textAlign = "right"
    ctx.fillText(t ? (t.text.length > 8 ? t.text.slice(0, 7) + "…" : t.text) : `#${i}`, 112, 24 + i * ch + 3)
    ctx.textAlign = "left"
    ctx.fillText(i === selectedToken ? `▶` : `#${i}`, 124 + i * cw, 12)
  }

  ctx.fillStyle = "#4cc2ff"
  ctx.font = "bold 11px monospace"
  ctx.textAlign = "left"
  ctx.fillText(
    `ATTENTION MATRIX · layer ${data.layer + 1} · head ${data.head} · row = token pos ${data.position} · real weights`,
    14,
    20,
  )
  if (selectedToken !== null) {
    const a = weights[selectedToken]?.weight ?? 0
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
  const [error, setError] = useState("")
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const inspectionMode = useBrain((s) => s.inspectionMode)

  const layer = selectedLayer ?? 0
  const head = selectedHead ?? 0
  const position = selectedToken ?? Math.max(0, tokens.length - 1)
  const allTokens = [
    ...tokens.map((t) => ({ position: t.position, text: t.text })),
    ...generatedTokens.map((t) => ({ position: t.position, text: t.text })),
  ]

  useEffect(() => {
    if (!sessionId) {
      setError("run an inference first")
      return
    }
    setError("")
    api.attention(sessionId, layer, head, position)
      .then(setData)
      .catch((e) => {
        setError(String(e))
        setData(null)
      })
  }, [sessionId, layer, head, position])

  useEffect(() => {
    if (data && canvasRef.current) drawHeatmap(canvasRef.current, data, allTokens, selectedToken)
  }, [data, allTokens, selectedToken])

  if (inspectionMode === "limited") return <ExternalObservationNotice />

  return (
    <div className="heatmap-wrap">
      <div className="heatmap-controls">
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
          layer
          <select value={layer} onChange={(e) => selectLayer(Number(e.target.value))} disabled={!model}>
            {Array.from({ length: model?.num_layers ?? 24 }, (_, i) => (
              <option key={i} value={i}>
                {i + 1}
              </option>
            ))}
          </select>
        </label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
          head
          <select value={head} onChange={(e) => selectHead(Number(e.target.value))} disabled={!model}>
            {Array.from({ length: model?.num_attention_heads ?? 14 }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center" }}>
          row (query token)
          <select value={position} onChange={(e) => selectToken(Number(e.target.value))}>
            {allTokens.map((t) => (
              <option key={t.position} value={t.position}>
                pos {t.position}: {t.text.length > 18 ? t.text.slice(0, 16) + "…" : t.text}
              </option>
            ))}
          </select>
        </label>
        {data && (
          <span className="mono" style={{ fontSize: 10.5, color: "var(--muted)" }}>
            L2 {data.stats.l2_norm.toFixed(3)} · mean {data.stats.mean.toFixed(4)} · {data.is_full_matrix ? "full matrix" : "new-token row (KV cache)"}
          </span>
        )}
      </div>
      {error && <div className="err-box">{error}</div>}
      <canvas ref={canvasRef} className="heatmap" />
      <div className="muted" style={{ fontSize: 10.5, fontFamily: "var(--mono)" }}>
        Row = query token's attention distribution over all previous tokens (causal mask). Values are the model's real softmax attention weights.
      </div>
    </div>
  )
}
