import { useBrain } from "../../store/useBrainStore"
import type { ViewMode } from "../../store/useBrainStore"

interface BottomNavItem {
  id: ViewMode
  label: string
  icon: string
}

const BOTTOM_NAV_ITEMS: BottomNavItem[] = [
  { id: "neural-interface", label: "Interface", icon: "⚡" },
  { id: "neuro-core", label: "Neuro Core", icon: "🧠" },
  { id: "architecture", label: "Network", icon: "🕸️" },
  { id: "memory-matrix", label: "Memory", icon: "🗄️" },
  { id: "tokenflow", label: "Tokens", icon: "🔤" },
  { id: "data-streams", label: "Streams", icon: "🌊" },
  { id: "model-hub", label: "Models", icon: "📦" },
  { id: "training-hub", label: "Training", icon: "🎯" },
  { id: "system-monitor", label: "Monitor", icon: "📊" },
  { id: "settings", label: "Settings", icon: "⚙️" },
]

export default function BottomNav() {
  const currentView = useBrain((s) => s.view)
  const setView = useBrain((s) => s.setView)

  const isCurrent = (id: ViewMode) => {
    if (currentView === id) return true
    if (id === "neuro-core" && currentView === "brain") return true
    if (id === "data-streams" && currentView === "dev") return true
    return false
  }

  return (
    <nav className="app-bottom-icon-nav mono">
      {BOTTOM_NAV_ITEMS.map((item) => {
        const active = isCurrent(item.id)
        return (
          <button
            key={item.id}
            className={`bottom-nav-btn ${active ? "active" : ""}`}
            onClick={() => setView(item.id)}
            title={item.label}
          >
            <span className="bottom-nav-icon">{item.icon}</span>
            <span className="bottom-nav-label">{item.label}</span>
          </button>
        )
      })}
    </nav>
  )
}
