import { useBrain } from "../../store/useBrainStore"
import type { ViewMode } from "../../store/useBrainStore"
import { useMockBattery } from "../../hooks/useMocks"
import { api } from "../../api/client"
import { disconnect } from "../../ws/client"

const QUICK_TABS: { id: ViewMode; label: string }[] = [
  { id: "neural-interface", label: "Neural Interface" },
  { id: "neuro-core", label: "Neuro Core" },
  { id: "architecture", label: "Architecture" },
  { id: "attention", label: "Attention" },
  { id: "embedding", label: "Embeddings" },
  { id: "tokenflow", label: "Token Flow" },
  { id: "dev", label: "Dev" },
]

export default function TopBar() {
  const connected = useBrain((s) => s.connected)
  const model = useBrain((s) => s.model)
  const modelStatus = useBrain((s) => s.modelStatus)
  const running = useBrain((s) => s.running)
  const currentStep = useBrain((s) => s.currentStep)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  const providerModel = useBrain((s) => s.providerModel)
  const provider = useBrain((s) => s.provider)
  const view = useBrain((s) => s.view)
  const setView = useBrain((s) => s.setView)
  const user = useBrain((s) => s.user)
  const authMode = useBrain((s) => s.authMode)
  const battery = useMockBattery()

  const isHealthy = connected && modelStatus === "loaded" && !running
  const displayedModel = inspectionMode === "limited" 
    ? (providerModel || provider || "external")
    : (model?.model_id ?? "Qwen/Qwen2.5-0.5B-Instruct")

  const isTabActive = (id: ViewMode) => {
    if (view === id) return true
    if (id === "neuro-core" && view === "brain") return true
    if (id === "dev" && view === "data-streams") return true
    return false
  }

  const handleLogout = () => {
    // Update the UI synchronously; network revocation must not leave the
    // operator looking at an authenticated dashboard while it is pending.
    void api.authLogout().catch(() => {})
    localStorage.removeItem("brainos_auth_token")
    disconnect()
    useBrain.getState().set({ user: null, authRequired: true, connected: false, runStatus: "IDLE" })
  }

  return (
    <header className="app-topbar">
      {/* Left: Brand / Title */}
      <div className="topbar-brand">
        <div className="brand-logo-cluster">
          <div className="brand-icon">
            <span className="brand-glyph">🧠</span>
          </div>
          <div className="brand-text">
            <span className="logo">BRAINOS</span>
            <span className="brand-version">3.0</span>
            <span className="brand-divider">/</span>
            <span className="brand-sub">Autonomous Intelligence OS</span>
          </div>
        </div>
      </div>

      {/* Center: System Status Strip */}
      <div className="topbar-status-strip mono">
        <div className="status-pill status-opt">
          <span className={`status-dot ${isHealthy ? "ok" : running ? "warn" : "bad"}`} />
          <span className="pill-text">SYSTEM STATUS: {connected ? "OPTIMAL" : "OFFLINE"}</span>
        </div>

        <div className="status-pill status-engine">
          <span className={`status-dot ${running ? "engine-active" : "ok"}`} />
          <span className="pill-text">
            NEURAL ENGINE: {running ? `ACTIVE · STEP ${currentStep}` : "ACTIVE"}
          </span>
        </div>

        <div className="status-pill status-model">
          <span className="pill-label">MODEL:</span>
          <span className="pill-val truncate" title={displayedModel}>
            {displayedModel}
          </span>
        </div>

        <div className="status-pill status-live">
          <span className="status-dot ok" />
          <span className="pill-text">LIVE</span>
        </div>
      </div>

      {/* Quick Navigation Tabs */}
      <div className="topbar-view-tabs mono">
        {QUICK_TABS.map((t) => (
          <button
            key={t.id}
            className={`topbar-tab-btn ${isTabActive(t.id) ? "active" : ""}`}
            onClick={() => setView(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Right: Power / User */}
      <div className="topbar-actions mono">
        <div className="battery-widget" title={`Power Source: ${battery.charging ? "AC Connected" : "Internal Battery"}`}>
          <span className="battery-icon">{battery.charging ? "⚡" : "🔋"}</span>
          <span className="battery-level">{battery.level}%</span>
          <span className="battery-mode text-dim">[PWR: {battery.charging ? "AC" : "BAT"}]</span>
        </div>

        <div className="user-profile-badge" title="Active Operator Session">
          <span className="user-avatar">🧑‍💻</span>
          <div className="user-details">
            <span className="user-name">{user?.username ?? (authMode === "local" ? "Local operator" : "Unauthenticated")}</span>
            <span className="user-mode text-cyan">{inspectionMode === "limited" ? "LIMITED" : "DEEP INSPECT"}</span>
          </div>
        </div>
      </div>
      {user && authMode === "multi_user" && (
        <button className="auth-logout-btn mono" onClick={handleLogout}>
          LOGOUT
        </button>
      )}
    </header>
  )
}
