import { useState, useEffect } from "react"
import { useBrain } from "../../store/useBrainStore"
import type { ViewMode } from "../../store/useBrainStore"

interface NavItem {
  id: ViewMode
  label: string
  subtitle: string
  icon: string
  badge?: string
}

const NAV_ITEMS: NavItem[] = [
  { id: "neural-interface", label: "Neural Interface", subtitle: "Ask Anything", icon: "⚡" },
  { id: "neuro-core", label: "Neuro Core", subtitle: "AI Engine", icon: "🧠" },
  { id: "architecture", label: "Neural Network", subtitle: "Deep Learning", icon: "🕸️" },
  { id: "memory-matrix", label: "Memory Matrix", subtitle: "Context Storage", icon: "🗄️" },
  { id: "tokenflow", label: "Token Engine", subtitle: "Tokenizer & Embedding", icon: "🔤" },
  { id: "data-streams", label: "Data Streams", subtitle: "Real-time Data Flow", icon: "🌊" },
  { id: "model-hub", label: "Model Hub", subtitle: "LLM Management", icon: "📦" },
  { id: "training-hub", label: "Training Hub", subtitle: "Model Training", icon: "🎯" },
  { id: "plugins", label: "Plugins", subtitle: "Extensions", icon: "🔌" },
  { id: "system-monitor", label: "System Monitor", subtitle: "Performance", icon: "📊" },
  { id: "settings", label: "Settings", subtitle: "Configuration", icon: "⚙️" },
]

export default function LeftSidebar() {
  const currentView = useBrain((s) => s.view)
  const setView = useBrain((s) => s.setView)
  const [uptimeSec, setUptimeSec] = useState(52084)
  const [clockStr, setClockStr] = useState("")
  const [showShutdownConfirm, setShowShutdownConfirm] = useState(false)

  // Dynamic uptime and clock
  useEffect(() => {
    const updateTime = () => {
      const now = new Date()
      setClockStr(now.toISOString().replace("T", " ").slice(0, 19))
      setUptimeSec((u) => u + 1)
    }
    updateTime()
    const timer = setInterval(updateTime, 1000)
    return () => clearInterval(timer)
  }, [])

  const formatUptime = (totalSec: number) => {
    const h = Math.floor(totalSec / 3600)
    const m = Math.floor((totalSec % 3600) / 60)
    const s = totalSec % 60
    return `${h}h ${m}m ${String(s).padStart(2, "0")}s`
  }

  const isCurrent = (id: ViewMode) => {
    if (currentView === id) return true
    if (id === "neuro-core" && currentView === "brain") return true
    if (id === "data-streams" && currentView === "dev") return true
    return false
  }

  return (
    <aside className="app-left-sidebar">
      {/* Navigation List */}
      <div className="sidebar-nav-scroll">
        <div className="sidebar-section-heading mono">OPERATING SECTIONS</div>
        <nav className="sidebar-nav-list">
          {NAV_ITEMS.map((item) => {
            const active = isCurrent(item.id)
            return (
              <button
                key={item.id}
                className={`nav-item-btn ${active ? "active" : ""}`}
                onClick={() => setView(item.id)}
                title={`${item.label} — ${item.subtitle}`}
              >
                <span className="nav-item-icon">{item.icon}</span>
                <div className="nav-item-text">
                  <span className="nav-item-label">{item.label}</span>
                  <span className="nav-item-sub">{item.subtitle}</span>
                </div>
                {item.badge && <span className="nav-item-badge">{item.badge}</span>}
                {active && <span className="nav-active-pip" />}
              </button>
            )
          })}
        </nav>
      </div>

      {/* Build Info Card & Shutdown Action */}
      <div className="sidebar-footer">
        <div className="build-info-card mono">
          <div className="build-header flex items-center justify-between">
            <span className="build-version">v3.0.4-release</span>
            <span className="build-tag">STABLE</span>
          </div>
          <div className="build-metrics">
            <div className="build-row">
              <span className="text-dim">UPTIME:</span>
              <span className="text-bright">{formatUptime(uptimeSec)}</span>
            </div>
            <div className="build-row">
              <span className="text-dim">SYSTEM:</span>
              <span className="text-cyan text-xxs truncate">{clockStr}</span>
            </div>
          </div>
        </div>

        {/* Pinned Shutdown Action */}
        <button
          className="shutdown-action-btn mono"
          onClick={() => setShowShutdownConfirm(true)}
          title="Gracefully terminate BrainOS worker session"
        >
          <span className="shutdown-icon">⏻</span>
          <span className="shutdown-text">SHUTDOWN SYSTEM</span>
        </button>
      </div>

      {/* Shutdown Modal Confirmation */}
      {showShutdownConfirm && (
        <div className="shutdown-modal-overlay">
          <div className="shutdown-modal mono">
            <div className="shutdown-modal-header">
              <span className="text-amber">⚠️ SHUTDOWN CONFIRMATION</span>
            </div>
            <p className="shutdown-modal-body">
              Terminate active BrainOS 3.0 session and unload MPS neural models from memory?
            </p>
            <div className="shutdown-modal-actions">
              <button
                className="btn danger"
                onClick={() => {
                  setShowShutdownConfirm(false)
                  alert("BrainOS session standby: PyTorch MPS allocations preserved.")
                }}
              >
                CONFIRM SHUTDOWN
              </button>
              <button
                className="btn"
                onClick={() => setShowShutdownConfirm(false)}
              >
                CANCEL
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}
