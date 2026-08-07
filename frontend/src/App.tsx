import { useEffect, useState } from "react"
import { useBrain } from "./store/useBrainStore"
import { connect, disconnect } from "./ws/client"
import { api } from "./api/client"
import Header from "./components/Header"
import PromptBar from "./components/PromptBar"
import ModelExplorer from "./components/ModelExplorer"
import Inspector from "./components/Inspector"
import BottomPanel from "./components/BottomPanel"
import BrainView from "./views/BrainView"
import ArchitectureView from "./views/ArchitectureView"
import AttentionView from "./views/AttentionView"
import EmbeddingView from "./views/EmbeddingView"
import TokenFlowView from "./views/TokenFlowView"
import DevView from "./views/DevView"

function ViewRouter() {
  const view = useBrain((s) => s.view)
  switch (view) {
    case "architecture":
      return <ArchitectureView />
    case "attention":
      return <AttentionView />
    case "embedding":
      return <EmbeddingView />
    case "tokenflow":
      return <TokenFlowView />
    case "dev":
      return <DevView />
    case "brain":
    default:
      return <BrainView />
  }
}

export default function App() {
  const [ready, setReady] = useState(false)
  useEffect(() => {
    connect()
    api.health().catch(() => {})
    api.hardware().then((h) => useBrain.getState().set({ hardware: h })).catch(() => {})
    api.model().then((m) => useBrain.getState().set({ model: m, modelStatus: "loaded" })).catch(() => {})
    api.sessions().then((s) => useBrain.getState().set({ sessions: s })).catch(() => {})
    setReady(true)
    return () => disconnect()
  }, [])

  if (!ready) return <div className="boot">BrainOS booting…</div>

  return (
    <div className="app">
      <Header />
      <div className="main-grid">
        <aside className="left-panel">
          <ModelExplorer />
        </aside>
        <main className="center-panel">
          <ViewRouter />
        </main>
        <aside className="right-panel">
          <Inspector />
        </aside>
      </div>
      <PromptBar />
      <BottomPanel />
    </div>
  )
}
