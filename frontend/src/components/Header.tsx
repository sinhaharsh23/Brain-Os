import { useBrain } from "../store/useBrainStore"
import type { ViewMode } from "../store/useBrainStore"

const TABS: { id: ViewMode; label: string }[] = [
  { id: "brain", label: "Brain" },
  { id: "architecture", label: "Architecture" },
  { id: "attention", label: "Attention" },
  { id: "embedding", label: "Embeddings" },
  { id: "tokenflow", label: "Token Flow" },
  { id: "dev", label: "Dev" },
]

export default function Header() {
  const connected = useBrain((s) => s.connected)
  const model = useBrain((s) => s.model)
  const modelStatus = useBrain((s) => s.modelStatus)
  const running = useBrain((s) => s.running)
  const view = useBrain((s) => s.view)
  const setView = useBrain((s) => s.setView)
  const currentStep = useBrain((s) => s.currentStep)

  const statusColor = !connected ? "bad" : modelStatus === "error" ? "bad" : modelStatus === "loading" ? "warn" : running ? "warn" : "ok"
  const statusText = !connected ? "OFFLINE" : modelStatus === "loading" ? "LOADING MODEL…" : running ? `PROCESSING STEP ${currentStep}` : "LIVE"

  return (
    <div className="header">
      <div className="logo">BRAINOS</div>
      <div className="stat">
        MODEL <b>{model?.model_id ?? "—"}</b>
      </div>
      <div className="stat">
        DEVICE <b>{(model?.device ?? "—").toUpperCase()}</b>
      </div>
      <div className="stat">
        LAYERS <b>{model?.num_layers ?? "—"}</b> · HEADS <b>{model?.num_attention_heads ?? "—"}</b> · HIDDEN <b>{model?.hidden_size ?? "—"}</b>
      </div>
      <div className="stat">
        <span className={`status-dot ${statusColor}`} />
        {statusText}
      </div>
      <div className="view-tabs">
        {TABS.map((t) => (
          <button key={t.id} className={view === t.id ? "active" : ""} onClick={() => setView(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
    </div>
  )
}
