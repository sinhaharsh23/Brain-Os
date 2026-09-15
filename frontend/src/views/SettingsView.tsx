import { useBrain } from "../store/useBrainStore"

export default function SettingsView() {
  const model = useBrain((s) => s.model)
  const inspectionMode = useBrain((s) => s.inspectionMode)

  return (
    <div className="section-page-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div>
          <h2 className="section-title">CONFIGURATION & SYSTEM SETTINGS</h2>
          <span className="section-subtitle text-dim text-xs">
            Inference engine defaults, WebSocket transport, and telemetry parameters
          </span>
        </div>
        <span className="badge-cyan">CONFIG SYNCED</span>
      </div>

      <div className="settings-grid">
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">INFERENCE ENGINE PREFERENCES</span>
          </div>
          <div className="setting-item flex items-center justify-between py-2">
            <div>
              <div className="setting-title text-bright text-xs">Inference Execution Backend</div>
              <div className="setting-desc text-dim text-xxs">PyTorch device dispatch for Apple Silicon</div>
            </div>
            <span className="badge-emerald">MPS (METAL ACCELERATED)</span>
          </div>
          <div className="setting-item flex items-center justify-between py-2">
            <div>
              <div className="setting-title text-bright text-xs">Default Sampling Temperature</div>
              <div className="setting-desc text-dim text-xxs">Softmax distribution randomness scaling</div>
            </div>
            <span className="text-cyan font-bold text-xs">0.7</span>
          </div>
          <div className="setting-item flex items-center justify-between py-2">
            <div>
              <div className="setting-title text-bright text-xs">Top-P Nucleus Sampling Threshold</div>
              <div className="setting-desc text-dim text-xxs">Cumulative probability cutoff for next token</div>
            </div>
            <span className="text-cyan font-bold text-xs">0.90</span>
          </div>
          <div className="setting-item flex items-center justify-between py-2">
            <div>
              <div className="setting-title text-bright text-xs">Top-K Candidate Cutoff</div>
              <div className="setting-desc text-dim text-xxs">Highest logit candidate pool size</div>
            </div>
            <span className="text-cyan font-bold text-xs">40</span>
          </div>
        </div>

        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">TRANSPORT & OBSERVATION PROTOCOL</span>
          </div>
          <div className="setting-item flex items-center justify-between py-2">
            <div>
              <div className="setting-title text-bright text-xs">WebSocket Endpoint</div>
              <div className="setting-desc text-dim text-xxs">Live async event bus bidirectional socket</div>
            </div>
            <span className="text-bright text-xs">ws://127.0.0.1:8765/ws</span>
          </div>
          <div className="setting-item flex items-center justify-between py-2">
            <div>
              <div className="setting-title text-bright text-xs">Active Observation Mode</div>
              <div className="setting-desc text-dim text-xxs">PyTorch forward hook tensor capture level</div>
            </div>
            <span className="badge-emerald">{inspectionMode === "limited" ? "LIMITED" : "DEEP INSPECTION"}</span>
          </div>
          <div className="setting-item flex items-center justify-between py-2">
            <div>
              <div className="setting-title text-bright text-xs">Active Context Window</div>
              <div className="setting-desc text-dim text-xxs">Maximum token sequence capacity</div>
            </div>
            <span className="text-bright text-xs">{model?.context_length ?? 32768} Tokens</span>
          </div>
        </div>
      </div>
    </div>
  )
}
