import { useBrain } from "../store/useBrainStore"

export default function ModelHubView() {
  const model = useBrain((s) => s.model)
  const modelStatus = useBrain((s) => s.modelStatus)
  const hardware = useBrain((s) => s.hardware)

  return (
    <div className="section-page-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div>
          <h2 className="section-title">MODEL HUB (LLM MANAGEMENT)</h2>
          <span className="section-subtitle text-dim text-xs">
            Local Apple Silicon Hugging Face weights and external API providers
          </span>
        </div>
        <span className="badge-emerald">1 MODEL ACTIVE</span>
      </div>

      <div className="model-hub-grid">
        {/* Active Local Model */}
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">PRIMARY LOCAL INSTANCE</span>
            <span className="badge-emerald">{modelStatus === "loaded" ? "MOUNTED" : "STANDBY"}</span>
          </div>
          <div className="model-detail-row">
            <span className="text-dim">MODEL ID:</span>
            <span className="text-bright font-bold">{model?.model_id ?? "Qwen/Qwen2.5-0.5B-Instruct"}</span>
          </div>
          <div className="model-detail-row">
            <span className="text-dim">WEIGHT ARCHITECTURE:</span>
            <span className="text-cyan">{model?.architecture ?? "Qwen2ForCausalLM"}</span>
          </div>
          <div className="model-detail-row">
            <span className="text-dim">DEVICE BACKEND:</span>
            <span className="text-emerald">{hardware?.backend ?? "MPS (Apple Silicon Metal)"}</span>
          </div>
          <div className="model-detail-row">
            <span className="text-dim">CACHE DIRECTORY:</span>
            <span className="text-dim text-xxs truncate">models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct</span>
          </div>
        </div>

        {/* External Observation Providers */}
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">EXTERNAL OBSERVATION PROVIDERS</span>
            <span className="badge-outline">META-OBSERVERS</span>
          </div>
          <div className="providers-list">
            <div className="provider-hub-item flex items-center justify-between">
              <div>
                <div className="provider-name text-bright">OpenAI (ChatGPT)</div>
                <div className="provider-modes text-xxs text-dim">gpt-4o-mini · LIMITED MODE</div>
              </div>
              <span className="badge-amber">EXTERNAL</span>
            </div>
            <div className="provider-hub-item flex items-center justify-between">
              <div>
                <div className="provider-name text-bright">Anthropic (Claude)</div>
                <div className="provider-modes text-xxs text-dim">claude-3-5-haiku · LIMITED MODE</div>
              </div>
              <span className="badge-amber">EXTERNAL</span>
            </div>
            <div className="provider-hub-item flex items-center justify-between">
              <div>
                <div className="provider-name text-bright">Google Gemini</div>
                <div className="provider-modes text-xxs text-dim">gemini-2.0-flash · LIMITED MODE</div>
              </div>
              <span className="badge-amber">EXTERNAL</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
