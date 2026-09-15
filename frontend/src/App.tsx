import { useEffect, useState } from "react"
import { useBrain } from "./store/useBrainStore"
import { connect, disconnect } from "./ws/client"
import { api } from "./api/client"
import TopBar from "./components/shell/TopBar"
import LeftSidebar from "./components/shell/LeftSidebar"
import RightSidebar from "./components/shell/RightSidebar"
import BottomNav from "./components/shell/BottomNav"
import BottomPanel from "./components/BottomPanel"
import AuthPanel from "./components/AuthPanel"

// Views
import NeuralInterfaceView from "./views/NeuralInterfaceView"
import DashboardView from "./components/dashboard/DashboardView"
import ArchitectureView from "./views/ArchitectureView"
import AttentionView from "./views/AttentionView"
import EmbeddingView from "./views/EmbeddingView"
import TokenFlowView from "./views/TokenFlowView"
import DevView from "./views/DevView"
import MemoryMatrixView from "./views/MemoryMatrixView"
import ModelHubView from "./views/ModelHubView"
import TrainingHubView from "./views/TrainingHubView"
import PluginsView from "./views/PluginsView"
import SystemMonitorView from "./views/SystemMonitorView"
import SettingsView from "./views/SettingsView"

function ViewRouter() {
  const view = useBrain((s) => s.view)
  switch (view) {
    case "neural-interface":
      return <NeuralInterfaceView />
    case "neuro-core":
    case "brain":
      return <DashboardView />
    case "architecture":
      return <ArchitectureView />
    case "attention":
      return <AttentionView />
    case "embedding":
      return <EmbeddingView />
    case "tokenflow":
      return <TokenFlowView />
    case "data-streams":
    case "dev":
      return <DevView />
    case "memory-matrix":
      return <MemoryMatrixView />
    case "model-hub":
      return <ModelHubView />
    case "training-hub":
      return <TrainingHubView />
    case "plugins":
      return <PluginsView />
    case "system-monitor":
      return <SystemMonitorView />
    case "settings":
      return <SettingsView />
    default:
      return <NeuralInterfaceView />
  }
}

export default function App() {
  const [ready, setReady] = useState(false)

  useEffect(() => {
    connect()
    api.health().catch(() => {})
    api.hardware().then((h) => useBrain.getState().set({ hardware: h })).catch(() => {})
    api.model().then((m) => useBrain.getState().set({ model: m, modelStatus: "loaded" })).catch(() => {})
    api.authMe().then((auth) => useBrain.getState().set({ authMode: auth.mode, user: auth.user, authRequired: auth.mode === "multi_user" && !auth.user })).catch((error) => {
      if (error?.status === 401) useBrain.getState().set({ authMode: "multi_user", authRequired: true })
    })
    setReady(true)
    return () => disconnect()
  }, [])

  if (!ready) return <div className="boot">BrainOS booting…</div>

  return (
    <div className="app brainos-app-shell">
      <TopBar />
      <div className="main-app-body">
        <LeftSidebar />
        <main className="center-panel">
          <ViewRouter />
        </main>
        <RightSidebar />
      </div>
      <BottomNav />
      <BottomPanel />
      <AuthPanel />
    </div>
  )
}
