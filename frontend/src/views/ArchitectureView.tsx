import { useEffect, useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"
import type { TensorVectorResponse } from "../types"
import ExternalObservationNotice from "../components/ExternalObservationNotice"

const VIEW_H = 760
const TOP = 90
const BOTTOM = 660
const CX = 420

export default function ArchitectureView() {
  const model = useBrain((s) => s.model)
  const layers = useBrain((s) => s.layers)
  const currentStep = useBrain((s) => s.currentStep)
  const selectLayer = useBrain((s) => s.selectLayer)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const qkvStats = useBrain((s) => s.qkvStats)
  const sessionId = useBrain((s) => s.sessionId)
  const selectedToken = useBrain((s) => s.selectedToken)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const selectedOutputToken = generatedTokens.find((token) => token.position === selectedToken)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  const [inputState, setInputState] = useState<TensorVectorResponse | null>(null)
  const [projectionData, setProjectionData] = useState<Partial<Record<"q" | "k" | "v" | "o", TensorVectorResponse | null>>>({})
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)

  const position = selectedOutputToken
    ? Math.max(0, selectedOutputToken.position - 1)
    : selectedToken ?? Math.max(0, tokens.length - 1)

  useEffect(() => {
    if (selectedLayer === null || !sessionId) {
      setInputState(null)
      setProjectionData({})
      setDetailError(null)
      return
    }
    let cancelled = false
    setDetailLoading(true)
    setDetailError(null)
    setInputState(null)
    setProjectionData({})
    const inputRequest = selectedLayer === 0
      ? api.embedding(sessionId, position)
      : api.hidden(sessionId, selectedLayer - 1, position)
    Promise.allSettled([
      inputRequest,
      api.qkv(sessionId, selectedLayer, "q", position, 8, 0),
      api.qkv(sessionId, selectedLayer, "k", position, 8, 0),
      api.qkv(sessionId, selectedLayer, "v", position, 8, 0),
      api.qkv(sessionId, selectedLayer, "o", position, 8),
    ]).then(([input, query, key, value, output]) => {
      if (cancelled) return
      if (input.status === "fulfilled") setInputState(input.value)
      const projections = {
        q: query.status === "fulfilled" ? query.value : null,
        k: key.status === "fulfilled" ? key.value : null,
        v: value.status === "fulfilled" ? value.value : null,
        o: output.status === "fulfilled" ? output.value : null,
      }
      setProjectionData(projections)
      if (query.status === "rejected" && key.status === "rejected" && value.status === "rejected") {
        setDetailError("No Q/K/V capture exists for this layer and token position. Run a Qwen inference and select a captured token.")
      }
      setDetailLoading(false)
    })
    return () => { cancelled = true }
  }, [selectedLayer, sessionId, position])

  if (inspectionMode === "limited") return <ExternalObservationNotice />
  if (!model) return <div className="muted" style={{ padding: 20 }}>model not loaded</div>

  const n = model.num_layers
  const slot = (BOTTOM - TOP) / n
  const boxH = Math.min(slot - 4, 20)

  const boxY = (i: number) => TOP + i * slot + slot / 2
  const label = (text: string, y: number, cls = "arch-label") => (
    <text x={CX} y={y} textAnchor="middle" className={cls}>
      {text}
    </text>
  )

  const selectedLayerState = selectedLayer === null ? null : layers[selectedLayer]
  const selectedQkv = selectedLayer === null ? [] : qkvStats.filter((item) => item.layer === selectedLayer)

  return (
    <div className="architecture-view">
    <svg className="arch-svg" viewBox={`0 0 ${CX * 2} ${VIEW_H}`} preserveAspectRatio="xMidYMid meet">
      {label(model.model_id, 30, "arch-title")}
      {label(`Transformer · ${n} layers · hidden ${model.hidden_size} · heads ${model.num_attention_heads} · MLP ${model.intermediate_size} · vocab ${model.vocab_size}`, 48)}

      {label("INPUT PROMPT", TOP - 18)}
      {label(`"${useBrain.getState().prompt.slice(0, 60) || "…"}"`, TOP - 6)}
      <line x1={CX} y1={TOP} x2={CX} y2={TOP + 4} stroke="#4cc2ff" />

      {label("TOKENIZER", TOP + 14)}
      {label(`${tokens.length} tokens (real IDs)`, TOP + 27)}

      <line x1={CX} y1={TOP + 34} x2={CX} y2={TOP + 40} stroke="#4cc2ff" />
      {label("EMBEDDINGS + POSITIONAL ENCODING", TOP + 50)}

      {Array.from({ length: n }, (_, i) => {
        const y = boxY(i)
        const act = layers[i]
        const active = act && act.step === currentStep
        const isSel = selectedLayer === i
        return (
          <g key={i} role="button" tabIndex={0} aria-label={`Select transformer layer ${i + 1}`} onClick={() => selectLayer(isSel ? null : i)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") selectLayer(isSel ? null : i) }} style={{ cursor: "pointer" }}>
            <rect
              x={CX - 150}
              y={y - boxH / 2}
              width={300}
              height={boxH}
              rx={4}
              fill={isSel ? "rgba(157,107,255,0.25)" : active ? "rgba(76,194,255,0.18)" : "rgba(20,32,64,0.6)"}
              stroke={isSel ? "#9d6bff" : active ? "#4cc2ff" : "#2a3d6e"}
              strokeWidth={isSel || active ? 1.6 : 1}
            />
            <text x={CX} y={y + 4} textAnchor="middle" className="arch-label" style={active ? { fill: "#4cc2ff" } : {}}>
              LAYER {i + 1}
            </text>
            {active && (
              <text x={CX + 165} y={y + 4} textAnchor="middle" className="arch-label" style={{ fill: "#8fd8ff" }}>
                |h|={act.norm.toFixed(2)}
              </text>
            )}
            <text x={CX - 165} y={y + 4} textAnchor="middle" className="arch-label" style={{ fill: "#9d6bff" }}>
              {i === 0 ? "SELF-ATTN" : ""}
            </text>
          </g>
        )
      })}

      <line x1={CX} y1={BOTTOM + 4} x2={CX} y2={BOTTOM + 12} stroke="#ffb648" />
      {label("LOGITS", BOTTOM + 20)}
      {label(`softmax over ${model.vocab_size.toLocaleString()} vocab (real)`, BOTTOM + 34)}
      {label("PROBABILITIES", BOTTOM + 48)}
      {label(`generated ${generatedTokens.length} tokens so far`, BOTTOM + 62)}
      <line x1={CX} y1={BOTTOM + 70} x2={CX} y2={BOTTOM + 76} stroke="#ffb648" />
      {label("FINAL RESPONSE", BOTTOM + 86)}
    </svg>
    {selectedLayer !== null && (
      <section className="architecture-layer-detail" aria-live="polite">
        <div className="layer-detail-heading">
          <strong>LAYER {selectedLayer + 1} DETAILS</strong>
          <span>{selectedLayerState?.status.startsWith("processing") ? `INPUT POSITION ${position} · COMPUTING` : selectedLayerState ? `INPUT POSITION ${position} · STEP ${selectedLayerState.step} · ${selectedLayerState.status.toUpperCase()}` : `INPUT POSITION ${position} · AWAITING CAPTURE`}</span>
        </div>
        <div className="layer-detail-metrics">
          <span>Hidden norm <b>{selectedLayerState && !selectedLayerState.status.startsWith("processing") ? selectedLayerState.norm.toFixed(3) : selectedLayerState ? "computing" : "not captured"}</b></span>
          <span>Mean <b>{selectedLayerState && !selectedLayerState.status.startsWith("processing") ? selectedLayerState.mean.toFixed(3) : selectedLayerState ? "computing" : "not captured"}</b></span>
          <span>Std <b>{selectedLayerState && !selectedLayerState.status.startsWith("processing") ? selectedLayerState.std.toFixed(3) : selectedLayerState ? "computing" : "not captured"}</b></span>
          <span>Time <b>{selectedLayerState && !selectedLayerState.status.startsWith("processing") ? `${selectedLayerState.timeMs.toFixed(2)} ms` : selectedLayerState ? "computing" : "not captured"}</b></span>
        </div>
        <div className="qkv-flow-explanation">
          <strong>REAL DATA FLOW AT INPUT POSITION {position}{selectedOutputToken ? ` → PREDICTS “${selectedOutputToken.text}”` : ""}</strong>
          <span><b>X input</b> {selectedLayer === 0 ? "token embedding enters Layer 1" : `hidden output from Layer ${selectedLayer}`} · {inputState ? `${inputState.dimension?.toLocaleString() ?? inputState.stats.n.toLocaleString()} dimensions · norm ${inputState.stats.l2_norm.toFixed(2)}` : detailLoading ? "loading captured input" : "input not captured"}</span>
          <span><b>Projections</b> Q = XWq, K = XWk, V = XWv · captured per-head projection samples are below.</span>
          <span><b>Use</b> score = QKᵀ / √dₖ → softmax(score) → weighted sum of V → O = Linear(weighted sum).</span>
        </div>
        {detailLoading && <div className="layer-detail-state">Loading captured values for this layer and token…</div>}
        <div className="layer-detail-qkv">
          {(["q", "k", "v", "o"] as const).map((name) => {
            const item = projectionData[name]
            const fallback = selectedQkv.find((stat) => stat.name === name)
            return <div className="layer-qkv-value" key={name}>
              <b>{name.toUpperCase()}</b>
              <span>{item ? `${item.stats.n.toLocaleString()} ${name === "o" ? "attention output" : "projected"} values · norm ${item.stats.l2_norm.toFixed(2)} · mean ${item.stats.mean.toFixed(3)}` : fallback ? `${fallback.stats.n.toLocaleString()} values captured · norm ${fallback.stats.l2_norm.toFixed(2)}` : detailLoading ? "loading" : "not captured for this token position"}</span>
              {item?.values && <code>{item.values.slice(0, 8).map((entry) => `${entry.index}:${entry.value.toFixed(3)}`).join("  ")}</code>}
              {item?.gqa && <small>query head {item.gqa.requested_query_head} → KV head {item.gqa.mapped_kv_head} · {item.gqa.attention_type}</small>}
            </div>
          })}
        </div>
        {projectionData.q?.query_key_matches && projectionData.q.query_key_matches.length > 0 && (
          <div className="layer-attention-results">
            <strong>Q·K SCORES AND SOFTMAX ATTENTION</strong>
            {projectionData.q.query_key_matches.slice(0, 4).map((match) => {
              const contribution = projectionData.q?.value_contributions?.find((item) => item.token_index === match.token_index)
              const token = [...tokens, ...generatedTokens].find((item) => item.position === match.token_index)
              return <span key={match.token_index}>token {match.token_index} {token ? `“${token.text}”` : ""} · QKᵀ/√dₖ {match.scaled_score.toFixed(3)} · softmax {match.attention_probability.toFixed(4)} · |weight×V| {contribution?.contribution_norm.toFixed(3) ?? "—"}</span>
            })}
          </div>
        )}
        {detailError && <div className="layer-detail-state">{detailError}</div>}
      </section>
    )}
    </div>
  )
}
