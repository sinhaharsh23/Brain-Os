import { useMemo } from "react"
import { useBrain } from "../../store/useBrainStore"

interface LedgerEntry {
  id: string
  action: "READ" | "WRITE" | "AUTH" | "SYNC" | "EXEC"
  resource: string
  metric: string
  valNorm: number
  status: "ALLOWED" | "ACTIVE" | "SYNCED" | "COMMITTED"
  layerIdx?: number
}

export default function AccessLedgerCard() {
  const layers = useBrain((s) => s.layers)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const running = useBrain((s) => s.running)
  const monitoring = useBrain((s) => s.monitoring)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectLayer = useBrain((s) => s.selectLayer)
  const model = useBrain((s) => s.model)

  const totalTokens = tokens.length + generatedTokens.length

  const ledgerEntries = useMemo(() => {
    const list: LedgerEntry[] = []
    const layerKeys = Object.keys(layers).map(Number).sort((a, b) => b - a)

    // KV Cache entry
    list.push({
      id: "kv-cache",
      action: "READ",
      resource: `sys.kv_cache.seq_${totalTokens}`,
      metric: `${totalTokens} context tokens`,
      valNorm: model?.context_length ? Math.min(100, (totalTokens / model.context_length) * 100) : 0,
      status: running ? "ACTIVE" : "SYNCED",
    })

    // Process memory footprint if available
    if (monitoring?.process_ram_gb) {
      list.push({
        id: "sys-ram-alloc",
        action: "AUTH",
        resource: "mem.process.rss_footprint",
        metric: `${monitoring.process_ram_gb.toFixed(2)} GB`,
        valNorm: Math.min(100, (monitoring.process_ram_gb / 8) * 100),
        status: "ALLOWED",
      })
    }

    // Active layer norms
    for (const l of layerKeys.slice(0, 6)) {
      const st = layers[l]
      if (!st) continue
      const normVal = st.norm
      list.push({
        id: `layer-${l}`,
        action: l % 2 === 0 ? "WRITE" : "SYNC",
        resource: `weights.attn_qkv.layer_${String(l).padStart(2, "0")}`,
        metric: `norm ${normVal.toFixed(2)}`,
        valNorm: Math.min(100, (normVal / 30) * 100),
        status: running && st.active ? "ACTIVE" : "COMMITTED",
        layerIdx: l,
      })
    }

    return list.slice(0, 6)
  }, [layers, totalTokens, running, monitoring, model])

  return (
    <div className="telemetry-card access-ledger-card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <span className="card-title">ACCESS LEDGER</span>
          <span className="badge-cyan">AUTH: ACTIVE</span>
        </div>
        <span className="text-xs text-muted mono">{model?.num_layers ?? "—"} LAYERS REGISTERED</span>
      </div>

      <div className="ledger-table">
        <div className="ledger-row ledger-thead mono">
          <span className="col-act">OP</span>
          <span className="col-res">TARGET RESOURCE</span>
          <span className="col-metric">METRIC</span>
          <span className="col-stat">STATUS</span>
        </div>
        {ledgerEntries.length === 0 && <div className="muted empty-ledger-msg">no access events captured yet</div>}
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
