import { useBrain } from "../../store/useBrainStore"

interface LedgerEntry {
  id: string
  action: "READ" | "EXEC"
  resource: string
  metric: string
  valNorm: number
  status: "ACTIVE" | "SYNCED" | "COMMITTED"
  layerIdx?: number
}

export default function AccessLedgerCard() {
  const layers = useBrain((s) => s.layers)
  const running = useBrain((s) => s.running)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  const telemetry = useBrain((s) => s.telemetry)
  const kvCache = useBrain((s) => s.kvCache)
  const currentStep = useBrain((s) => s.currentStep)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectLayer = useBrain((s) => s.selectLayer)
  const model = useBrain((s) => s.model)

  const cache = kvCache[currentStep] ?? Object.values(kvCache).at(-1)
  const layerKeys = Object.keys(layers).map(Number).sort((a, b) => b - a).slice(0, 5)
  const norms = layerKeys.map((layer) => Math.abs(layers[layer]?.norm ?? 0))
  const maxNorm = Math.max(...norms, 1e-12)
  const ledgerEntries: LedgerEntry[] = []
  if (cache) ledgerEntries.push({
    id: `kv-cache-${cache.step}`, action: "READ", resource: "runtime.kv_cache",
    metric: `sequence ${cache.seq_length} · ${(cache.total_kb).toFixed(1)} KB · ${cache.num_layers} layers`,
    valNorm: model?.context_length ? Math.min(100, cache.seq_length / model.context_length * 100) : 0,
    status: running ? "ACTIVE" : "SYNCED",
  })
  for (const layer of layerKeys) {
    const state = layers[layer]
    if (!state) continue
    ledgerEntries.push({
      id: `layer-${layer}`, action: "EXEC", resource: `transformer.layer_${String(layer).padStart(2, "0")}.output_hidden_state`,
      metric: `captured output norm ${state.norm.toFixed(3)}`,
      valNorm: Math.abs(state.norm) / maxNorm * 100,
      status: running && state.active ? "ACTIVE" : "COMMITTED", layerIdx: layer,
    })
  }

  return (
    <div className="telemetry-card access-ledger-card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <span className="card-title">CAPTURE INDEX</span>
          <span className="badge-cyan">REAL CAPTURES ONLY</span>
        </div>
        <span className="text-xs text-muted mono">{telemetry?.model.num_layers ?? model?.num_layers ?? "—"} LAYERS</span>
      </div>

      <div className="ledger-table">
        <div className="ledger-row ledger-thead mono">
          <span className="col-act">OP</span>
          <span className="col-res">TARGET RESOURCE</span>
          <span className="col-metric">METRIC</span>
          <span className="col-stat">STATUS</span>
        </div>
        {ledgerEntries.length === 0 && <div className="muted empty-ledger-msg">{inspectionMode === "limited" ? "Internal tensor events unavailable through this provider." : "No cache or layer captures recorded for this run."}</div>}
        {ledgerEntries.map((item) => {
          const isSelected = item.layerIdx !== undefined && selectedLayer === item.layerIdx
          return (
            <div
              key={item.id}
              className={`ledger-row mono ${isSelected ? "selected-row" : ""}`}
              onClick={() => {
                if (item.layerIdx !== undefined) selectLayer(item.layerIdx)
              }}
              title={item.layerIdx !== undefined ? `Inspect Layer ${item.layerIdx}` : item.resource}
            >
              <span className={`badge-op op-${item.action.toLowerCase()}`}>
                [{item.action}]
              </span>
              <span className="col-res truncate">
                {item.resource}
              </span>
              <div className="col-metric flex items-center gap-1">
                <span className="metric-text">{item.metric}</span>
                <div className="metric-bar-bg">
                  <div className="metric-bar-fill" style={{ width: `${Math.max(5, item.valNorm)}%` }} />
                </div>
              </div>
              <span className={`badge-stat stat-${item.status.toLowerCase()}`}>
                {item.status}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
