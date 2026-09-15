import { useState, useRef, useEffect, useMemo } from "react"
import { useBrain } from "../../store/useBrainStore"
import { send } from "../../ws/client"

export default function RightSidebar() {
  const monitoring = useBrain((s) => s.monitoring)
  const hardware = useBrain((s) => s.hardware)
  const model = useBrain((s) => s.model)
  const devLog = useBrain((s) => s.devLog)
  const running = useBrain((s) => s.running)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const response = useBrain((s) => s.response)
  const setStore = useBrain((s) => s.set)

  const [clearedLogs, setClearedLogs] = useState(false)
  const [activeFileModal, setActiveFileModal] = useState(false)
  const logScrollRef = useRef<HTMLDivElement | null>(null)

  const cpuPercent = monitoring?.cpu_percent ?? hardware?.cpu?.percent ?? null
  const gpuPercent = monitoring?.gpu_percent ?? null
  const ramUsed = monitoring?.ram_used_gb ?? hardware?.ram?.used_gb ?? null
  const ramTotal = monitoring?.ram_total_gb ?? hardware?.ram?.total_gb ?? null
  const contextUsed = tokens.length + generatedTokens.length
  const contextMax = model?.context_length ?? null
  const modelName = model?.model_id ?? "model metadata unavailable"

  useEffect(() => {
    if (logScrollRef.current) {
      logScrollRef.current.scrollTop = logScrollRef.current.scrollHeight
    }
  }, [devLog, running])

  const visibleLogs = clearedLogs ? [] : devLog

  const renderMiniSpark = (val: number | null, color: string) => {
    if (val === null) return <span className="text-dim text-xxs">UNAVAILABLE</span>
    const pts = [val * 0.7, val * 0.85, val * 0.75, val * 0.95, val]
    const maxVal = Math.max(10, val * 1.2)
    const pointsStr = pts
      .map((p, i) => `${i * 12},${22 - (p / maxVal) * 18}`)
      .join(" ")
    return (
      <svg viewBox="0 0 50 24" className="mini-spark-svg">
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          points={pointsStr}
        />
        <circle
          cx={4 * 12}
          cy={22 - (pts[4] / maxVal) * 18}
          r="2"
          fill={color}
          className={running ? "pulse-dot" : ""}
        />
      </svg>
    )
  }

  const handleNewChat = () => {
    setStore({
      prompt: "",
      tokens: [],
      generatedTokens: [],
      response: "",
      currentStep: 0,
      candidates: null,
      summary: null,
    })
  }

  const handleClearContext = () => {
    setStore({
      tokens: [],
      generatedTokens: [],
      response: "",
      layers: {},
      attentionLinks: {},
      currentStep: 0,
    })
    send("cancel")
  }

  const handleExportChat = () => {
    const data = {
      model: modelName,
      timestamp: new Date().toISOString(),
      tokensCount: contextUsed,
      prompt: tokens.map((t) => t.text).join(""),
      response,
      history: devLog,
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `brainos-chat-export-${Date.now()}.json`
    a.click()
  }

  const brainNodes = useMemo(() => [
    { cx: 35, cy: 30, r: 4 }, { cx: 55, cy: 22, r: 5 }, { cx: 75, cy: 32, r: 4 },
    { cx: 25, cy: 50, r: 4 }, { cx: 48, cy: 45, r: 6 }, { cx: 70, cy: 52, r: 5 },
    { cx: 38, cy: 68, r: 5 }, { cx: 62, cy: 72, r: 4 }, { cx: 85, cy: 48, r: 4 }
  ], [])

  return (
    <aside className="app-right-sidebar">
      {/* 1. System Overview */}
      <div className="right-card system-overview-card mono">
        <div className="right-card-header flex items-center justify-between">
          <span className="card-title-sm">SYSTEM OVERVIEW</span>
          <span className="live-badge">
            <span className="status-dot ok" /> SYNCED
          </span>
        </div>

        <div className="metrics-spark-grid">
          <div className="spark-stat-row">
            <div className="stat-meta">
              <span className="meta-label">CPU LOAD</span>
              <span className="meta-val">{cpuPercent === null ? "—" : `${cpuPercent.toFixed(0)}%`}</span>
            </div>
            {renderMiniSpark(cpuPercent, "#00d2ff")}
          </div>

          <div className="spark-stat-row">
            <div className="stat-meta">
              <span className="meta-label">GPU / MPS</span>
              <span className="meta-val text-amber">{gpuPercent === null ? "—" : `${gpuPercent.toFixed(0)}%`}</span>
            </div>
            {renderMiniSpark(gpuPercent, "#ffb648")}
          </div>

          <div className="spark-stat-row">
            <div className="stat-meta">
              <span className="meta-label">NEURAL ENGINE</span>
              <span className="meta-val text-emerald">{running ? "ACTIVE" : "STANDBY"}</span>
            </div>
            <span className="text-dim text-xxs">{running ? "INFERENCE EVENT" : "NO ACTIVE EVENT"}</span>
          </div>
        </div>

        <div className="overview-sub-metrics">
          <div className="sub-metric-item">
            <span className="sub-label">MEMORY</span>
            <span className="sub-val">{ramUsed === null || ramTotal === null ? "—" : `${ramUsed.toFixed(1)} / ${ramTotal.toFixed(0)} GB`}</span>
          </div>
          <div className="sub-metric-item">
            <span className="sub-label">CONTEXT</span>
            <span className="sub-val text-cyan">{contextUsed} / {contextMax ?? "—"}</span>
          </div>
        </div>

        <div className="active-model-strip">
          <span className="text-dim text-xxs">ACTIVE MODEL:</span>
          <span className="text-bright text-xs truncate" title={modelName}>{modelName}</span>
        </div>
      </div>

      {/* 2. Neural Engine Status */}
      <div className="right-card neural-status-card mono">
        <div className="right-card-header flex items-center justify-between">
          <span className="card-title-sm">NEURAL ENGINE STATUS</span>
          <span className={`engine-state-tag ${running ? "pulse-amber" : "state-active"}`}>
            {running ? "INFERENCING" : "STANDBY ACTIVE"}
          </span>
        </div>

        <div className="neural-brain-graphic">
          <svg viewBox="0 0 110 95" className="brain-svg">
            <defs>
              <radialGradient id="brainGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#00d2ff" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#00d2ff" stopOpacity="0" />
              </radialGradient>
            </defs>
            <circle cx="55" cy="48" r="42" fill="url(#brainGlow)" />
            <path
              d="M 35 30 L 55 22 L 75 32 L 70 52 L 48 45 L 25 50 L 38 68 L 62 72 L 70 52 M 48 45 L 75 32 M 35 30 L 48 45 M 55 22 L 48 45 M 62 72 L 48 45 M 70 52 L 85 48"
              stroke={running ? "#ffb648" : "rgba(0, 210, 255, 0.4)"}
              strokeWidth="1.2"
              fill="none"
              strokeDasharray={running ? "3,2" : undefined}
            />
            {brainNodes.map((n, idx) => (
              <circle
                key={idx}
                cx={n.cx}
                cy={n.cy}
                r={n.r}
                fill={running ? (idx % 2 === 0 ? "#ffb648" : "#00d2ff") : "#00d2ff"}
                className={running ? "pulse-dot" : ""}
              />
            ))}
          </svg>
          <div className="engine-readout text-xxs">
            <span className="text-cyan">MPS ACCELERATOR ACTIVE</span>
            <span className="text-dim">494M WEIGHT MATRIX LOADED</span>
          </div>
        </div>
      </div>

      {/* 3. Event Logs Feed */}
      <div className="right-card live-logs-card mono">
        <div className="right-card-header flex items-center justify-between">
          <span className="card-title-sm">EVENT LOGS</span>
          <div className="flex items-center gap-1">
            <button
              className="log-clear-btn text-xxs"
              onClick={() => setClearedLogs(!clearedLogs)}
              title="Clear or restore log view"
            >
              {clearedLogs ? "RESTORE" : "CLEAR"}
            </button>
          </div>
        </div>

        <div className="live-log-stream" ref={logScrollRef}>
          {visibleLogs.length === 0 ? (
            <div className="log-empty-hint text-dim text-xxs">
              [SYSTEM] Pipeline idling — dispatch inference prompt to stream events
            </div>
          ) : (
            visibleLogs.slice(-40).map((l, idx) => {
              const timeStr = new Date(l.ts).toLocaleTimeString("en-GB", { hour12: false })
              return (
                <div key={idx} className={`log-event-line ${l.level}`}>
                  <span className="log-time text-xxs text-dim">{timeStr}</span>
                  <span className="log-level-badge text-xxs">[{l.level.slice(0, 4).toUpperCase()}]</span>
                  <span className="log-text text-xxs">{l.message}</span>
                </div>
              )
            })
          )}
          {running && (
            <div className="log-event-line info pulse-row">
              <span className="log-time text-xxs text-dim">--:--:--</span>
              <span className="log-level-badge text-xxs">[EXEC]</span>
              <span className="log-text text-xxs">streaming generated tokens...</span>
            </div>
          )}
        </div>
      </div>

      {/* 4. Quick Actions */}
      <div className="right-card quick-actions-card mono">
        <div className="right-card-header">
          <span className="card-title-sm">QUICK ACTIONS</span>
        </div>

        <div className="quick-actions-grid">
          <button className="quick-act-btn" onClick={handleNewChat} title="Reset session prompt and response">
            <span className="act-icon">💬</span>
            <span className="act-label">NEW CHAT</span>
          </button>
          <button className="quick-act-btn" onClick={() => setActiveFileModal(true)} title="Upload context document">
            <span className="act-icon">📁</span>
            <span className="act-label">UPLOAD FILE</span>
          </button>
          <button className="quick-act-btn" onClick={handleClearContext} title="Clear KV cache context">
            <span className="act-icon">🧹</span>
            <span className="act-label">CLEAR CONTEXT</span>
          </button>
          <button className="quick-act-btn" onClick={handleExportChat} title="Export conversation as JSON">
            <span className="act-icon">💾</span>
            <span className="act-label">EXPORT CHAT</span>
          </button>
        </div>
      </div>

      {/* Upload File Modal */}
      {activeFileModal && (
        <div className="shutdown-modal-overlay">
          <div className="shutdown-modal mono">
            <div className="shutdown-modal-header">
              <span className="text-cyan">📁 UPLOAD CONTEXT FILE</span>
            </div>
            <p className="shutdown-modal-body text-xs">
              Attach document (.txt, .md, .json) to ingest into Memory Matrix and KV cache embeddings.
            </p>
            <input
              type="file"
              className="modal-file-input text-xs"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) {
                  const reader = new FileReader()
                  reader.onload = (evt) => {
                    const content = String(evt.target?.result ?? "")
                    setStore({ prompt: content.slice(0, 1000) })
                    setActiveFileModal(false)
                  }
                  reader.readAsText(file)
                }
              }}
            />
            <div className="shutdown-modal-actions">
              <button className="btn" onClick={() => setActiveFileModal(false)}>
                CANCEL
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}
