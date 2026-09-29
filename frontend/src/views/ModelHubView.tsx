import { useBrain } from "../store/useBrainStore"

export default function ModelHubView() {
  const provider = useBrain((s) => s.provider)
  const providerModel = useBrain((s) => s.providerModel)
  const telemetry = useBrain((s) => s.telemetry)
  const model = useBrain((s) => s.model)
  const modelStatus = useBrain((s) => s.modelStatus)
  const hardware = useBrain((s) => s.hardware)
  const activeModel = telemetry?.model
  const ollama = provider === "ollama"
  const parameterCount = activeModel?.parameter_count ?? (ollama ? null : model?.num_params)
  const formatParameters = (count: number | null | undefined) => count == null ? "Unavailable" : count >= 1e9 ? `${(count / 1e9).toFixed(2)}B` : count >= 1e6 ? `${(count / 1e6).toFixed(2)}M` : count.toLocaleString()

  return (
    <div className="section-page-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div>
          <h2 className="section-title">MODEL HUB (LLM MANAGEMENT)</h2>
          <span className="section-subtitle text-dim text-xs">
            Provider-aware local model and runtime metadata
          </span>
        </div>
        <span className={modelStatus === "loaded" ? "badge-emerald" : "badge-outline"}>{modelStatus === "loaded" ? "READY" : modelStatus.toUpperCase()}</span>
      </div>

      <div className="model-hub-grid">
        {/* Active Local Model */}
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">ACTIVE PROVIDER · {ollama ? "OLLAMA" : provider === "qwen-local" ? "HUGGING FACE LOCAL" : (provider ?? "UNSELECTED").toUpperCase()}</span>
            <span className={modelStatus === "loaded" ? "badge-emerald" : "badge-outline"}>{modelStatus === "loaded" ? "READY" : modelStatus.toUpperCase()}</span>
          </div>
          <div className="model-detail-row">
            <span className="text-dim">MODEL ID:</span>
            <span className="text-bright font-bold">{activeModel?.id ?? providerModel ?? model?.model_id ?? "Unavailable"}</span>
          </div>
          <div className="model-detail-row">
            <span className="text-dim">WEIGHT ARCHITECTURE:</span>
            <span className="text-cyan">{activeModel?.architecture ?? (ollama ? "Unavailable through Ollama metadata" : model?.architecture ?? "Unavailable")}</span>
          </div>
          <div className="model-detail-row">
            <span className="text-dim">RUNTIME / DEVICE:</span>
            <span className="text-emerald">{ollama ? "Ollama runtime" : activeModel?.device ?? model?.device ?? hardware?.backend ?? "Unavailable"}</span>
          </div>
          {[["PARAMETERS", formatParameters(parameterCount)], ["CONTEXT", activeModel?.context_window ?? (ollama ? null : model?.context_length)], ["RUNTIME DTYPE", activeModel?.runtime_dtype ?? (ollama ? null : model?.dtype)], ["QUANTIZATION", activeModel?.quantization ?? (ollama ? null : model?.quantization)]].map(([label, value]) => (
            <div className="model-detail-row" key={String(label)}><span className="text-dim">{label}:</span><span className="text-bright">{value == null ? "Unavailable" : String(value)}</span></div>
          ))}
          <div className="model-detail-row"><span className="text-dim">LAYERS / HIDDEN:</span><span className="text-bright">{activeModel?.num_layers != null ? `${activeModel.num_layers} / ${activeModel.hidden_size ?? "Unavailable"}` : ollama ? "Unavailable" : `${model?.num_layers ?? "Unavailable"} / ${model?.hidden_size ?? "Unavailable"}`}</span></div>
          <div className="model-detail-row"><span className="text-dim">ATTENTION HEADS:</span><span className="text-bright">{activeModel?.num_attention_heads != null ? `${activeModel.num_attention_heads} Query / ${activeModel.num_kv_heads ?? "Unavailable"} Key / ${activeModel.num_kv_heads ?? "Unavailable"} Value` : ollama ? "Unavailable" : model ? `${model.num_attention_heads} Query / ${model.num_kv_heads} Key / ${model.num_kv_heads} Value` : "Unavailable"}</span></div>
          <div className="model-detail-row"><span className="text-dim">CONFIG DTYPE:</span><span className="text-bright">{String(activeModel?.config_dtype ?? (ollama ? "Unavailable" : model?.extra.config_dtype ?? "Unavailable"))}</span></div>
          {ollama && <div className="model-detail-row"><span className="text-dim">DEEP PYTORCH INSPECTION:</span><span className="text-amber">Unavailable through Ollama</span></div>}
          <div className="model-detail-row">
            <span className="text-dim">NATIVE INSPECTION:</span>
            <span className="text-dim text-xxs truncate">{ollama ? "Switch to Hugging Face Local" : model?.capabilities?.forward_hooks ? "Available · actual model hooks" : "Unavailable"}</span>
          </div>
        </div>

        {/* External Observation Providers */}
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">PROVIDER OBSERVABILITY</span>
            <span className="badge-outline">CAPABILITY AWARE</span>
          </div>
          <div className="providers-list">
            <div className="provider-hub-item flex items-center justify-between">
              <div>
                <div className="provider-name text-bright">{ollama ? "Native Qwen Deep Inspection" : "Ollama Runtime Observability"}</div>
                <div className="provider-modes text-xxs text-dim">{ollama ? "Switch to Hugging Face Local for direct model tensors" : "Response, runtime counts, timing, metadata and supported logprobs"}</div>
              </div>
              <span className="badge-amber">{ollama ? "AVAILABLE IN NATIVE MODE" : "PROVIDER API"}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
