import { useState } from "react"
import NeuralGraphHero from "./NeuralGraphHero"
import AnalyticsDeck from "../analytics/AnalyticsDeck"
import BrainScene from "../../brain3d/BrainScene"
import PromptBar from "../PromptBar"
import ExternalObservationNotice from "../ExternalObservationNotice"
import { useBrain } from "../../store/useBrainStore"

export default function DashboardView() {
  const modelStatus = useBrain((s) => s.modelStatus)
  const inferenceError = useBrain((s) => s.inferenceError)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  const connected = useBrain((s) => s.connected)
  const running = useBrain((s) => s.running)
  const paused = useBrain((s) => s.paused)
  const currentStep = useBrain((s) => s.currentStep)
  const model = useBrain((s) => s.model)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const layers = useBrain((s) => s.layers)
  const attentionLinks = useBrain((s) => s.attentionLinks)
  const response = useBrain((s) => s.response)
  const summary = useBrain((s) => s.summary)
  const [subMode, setSubMode] = useState<"neural-deck" | "3d-scene">("3d-scene")

  const activeLayers = Object.values(layers).filter((layer) => layer.step === currentStep)
  const capturedAttention = Object.values(attentionLinks).reduce((count, links) => count + links.length, 0)
  const latestToken = generatedTokens[generatedTokens.length - 1]
  const stage = inspectionMode === "limited"
    ? "EXTERNAL OBSERVATION"
    : running
    ? paused
      ? "PAUSED"
      : latestToken
      ? "STREAMING TOKEN"
      : activeLayers.length > 0
      ? "LAYER INFERENCE"
      : "DISPATCHING"
    : response
    ? "RUN COMPLETE"
    : connected
    ? "READY"
    : "BACKEND OFFLINE"
  const tps = summary?.timings?.tokens_per_second

  return (
    <div className="dashboard-container">
      <div className="dashboard-subnav">
        <div className="subnav-left">
          <span className="subnav-kicker mono">NEURO CORE / 01</span>
          <span className="subnav-title">Model observability</span>
          <span className={`subnav-dot ${running ? "running" : ""}`} />
          <span className="subnav-status mono">{stage}</span>
        </div>
        <div className="subnav-modes">
          <button
            className={`subnav-btn ${subMode === "3d-scene" ? "active" : ""}`}
            onClick={() => setSubMode("3d-scene")}
          >
            NEURAL FIELD
          </button>
          <button
            className={`subnav-btn ${subMode === "neural-deck" ? "active" : ""}`}
            onClick={() => setSubMode("neural-deck")}
          >
            TRACE GRAPH
          </button>
        </div>
      </div>

      <div className="dashboard-prompt-shell">
        <div className="dashboard-prompt-label mono">
          <span className="prompt-command-mark">⌁</span>
          <span>INFERENCE CONSOLE</span>
          <span className="prompt-label-line" />
          <span className="prompt-label-hint">PROMPT → TOKEN STREAM → OBSERVABILITY</span>
        </div>
        <PromptBar />
      </div>

      <div className="dashboard-scrollable-content">
        <section className="dashboard-hero-section">
          <div className="dashboard-hero-header">
            <div>
              <div className="hero-eyebrow mono">REAL-TIME ACTIVATION MAP</div>
              <h1>Neural field</h1>
              <p>Follow context as it traverses the transformer stack and resolves into output.</p>
            </div>
            <div className="hero-live-metrics mono">
              <div><span>MODEL</span><strong title={model?.model_id}>{model?.model_id?.split("/").pop() ?? "—"}</strong></div>
              <div><span>LAYERS CAPTURED</span><strong>{Object.keys(layers).length || "—"}</strong></div>
              <div><span>ATTENTION EDGES</span><strong>{capturedAttention || "—"}</strong></div>
            </div>
          </div>

          {inspectionMode === "limited" && (
            <div className="dashboard-observation-banner">
              <ExternalObservationNotice />
            </div>
          )}

          {subMode === "3d-scene" ? (
            <div className="dashboard-neural-stage">
              <div className="stage-corner stage-corner-tl mono">FIELD / 3D · DEPTH ENABLED</div>
              <div className="stage-corner stage-corner-tr mono">{tokens.length + generatedTokens.length} CONTEXT NODES</div>
              <BrainScene />
              <div className="neural-stage-overlay mono">
                <div className="stage-readout">
                  <span className="readout-label">CURRENT STAGE</span>
                  <strong className={running ? "active" : ""}>{stage}</strong>
                </div>
                <div className="stage-readout">
                  <span className="readout-label">STEP</span>
                  <strong>{running || currentStep > 0 ? String(currentStep).padStart(3, "0") : "—"}</strong>
                </div>
                <div className="stage-readout stage-token-readout">
                  <span className="readout-label">LATEST TOKEN</span>
                  <strong>{latestToken ? `${latestToken.text || "∅"} · ${(latestToken.probability * 100).toFixed(1)}%` : "awaiting token event"}</strong>
                </div>
              </div>
            </div>
          ) : (
            <NeuralGraphHero />
          )}

          <div className="dashboard-hero-footer mono">
            <span><i className={running ? "signal-live" : "signal-idle"} /> {connected ? "WS LINK" : "WS DISCONNECTED"}</span>
            <span>{inspectionMode === "limited" ? "private tensors unavailable" : `${tokens.length} prompt · ${generatedTokens.length} generated`}</span>
            <span>{tps ? `${tps} tok/s` : latestToken ? `${latestToken.time_ms} ms/token` : "throughput —"}</span>
          </div>
        </section>

        <section className="dashboard-analytics-section">
          <div className="analytics-section-heading">
            <div>
              <span className="hero-eyebrow mono">OPERATOR CONSOLE / 02</span>
              <h2>Signal deck</h2>
            </div>
            <span className="analytics-heading-status mono">{running ? "● STREAMING LIVE" : "○ AWAITING RUN"}</span>
          </div>
          <div className="dashboard-analytics-grid">
            <AnalyticsDeck />
          </div>
        </section>
      </div>

      {modelStatus !== "loaded" && (
        <div
          className="err-box"
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: 60,
            background: "rgba(10,14,26,0.92)",
            border: "1px solid var(--border-bright)",
            boxShadow: "0 0 24px rgba(0, 210, 255, 0.25)",
          }}
        >
          {modelStatus === "loading"
            ? "Loading Qwen2.5-0.5B-Instruct on Apple Silicon MPS…"
            : modelStatus === "error"
            ? `Model failed to load: ${inferenceError ?? "unknown error"}`
            : "Waiting for model…"}
        </div>
      )}
    </div>
  )
}
