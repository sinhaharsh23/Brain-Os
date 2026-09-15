import { useBrain } from "../store/useBrainStore"
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
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const inspectionMode = useBrain((s) => s.inspectionMode)

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

  return (
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
          <g key={i} onClick={() => selectLayer(isSel ? null : i)} style={{ cursor: "pointer" }}>
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
  )
}
