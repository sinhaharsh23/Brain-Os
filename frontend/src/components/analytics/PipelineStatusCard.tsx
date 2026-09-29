import { useBrain } from "../../store/useBrainStore"

interface PipelineStage {
  id: string
  name: string
  detail: string
  subtext: string
  activeWhen: "always" | "tokenizing" | "inferencing" | "logits" | "done"
}

export default function PipelineStatusCard() {
  const model = useBrain((s) => s.model)
  const hardware = useBrain((s) => s.hardware)
  const running = useBrain((s) => s.running)
  const currentStep = useBrain((s) => s.currentStep)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  const provider = useBrain((s) => s.provider)
  const telemetry = useBrain((s) => s.telemetry)
  const limited = inspectionMode === "limited"

  const stages: PipelineStage[] = [
    {
      id: "tok",
      name: limited ? "PROMPT EVALUATION" : "TOKENIZER",
      detail: limited ? telemetry?.tokens.model_input_tokens == null ? "Unavailable" : `${telemetry.tokens.model_input_tokens} runtime input tokens` : `${tokens.length} model input tokens`,
      subtext: limited ? "count from provider final metrics" : model ? `${model.vocab_size.toLocaleString()} vocabulary entries` : "metadata unavailable",
      activeWhen: "always",
    },
    {
      id: "embed",
      name: limited ? "EMBEDDING TENSORS" : "EMBED & ROPE",
      detail: limited ? "Unavailable through provider" : model ? `${model.hidden_size}d vector` : "Unavailable",
      subtext: limited ? "internal tensors are not exposed" : model ? "rotary position embeddings" : "metadata unavailable",
      activeWhen: "always",
    },
    {
      id: "attn",
      name: limited ? "ATTENTION / QKV" : "GQA ATTENTION",
      detail: limited ? "Unavailable through provider" : model ? `${model.num_attention_heads} Query / ${model.num_kv_heads} Key / ${model.num_kv_heads} Value heads` : "Unavailable",
      subtext: limited ? "deep tensor inspection requires native mode" : model ? "captured only when a real hook ran" : "metadata unavailable",
      activeWhen: "inferencing",
    },
    {
      id: "mlp",
      name: limited ? "MLP ACTIVATIONS" : "SWIGLU MLP",
      detail: limited ? "Unavailable through provider" : model ? `${model.intermediate_size} intermediate` : "Unavailable",
      subtext: limited ? "internal tensors are not exposed" : model ? `${model.num_layers} layers · hook capture only` : "metadata unavailable",
      activeWhen: "inferencing",
    },
    {
      id: "logits",
      name: limited ? "TOKEN LOGPROBS" : "LM HEAD LOGITS",
      detail: limited ? telemetry?.capabilities.logprobs ? "Returned by provider" : "Unavailable through provider" : "Captured from model logits",
      subtext: limited ? "probability shown only when returned" : "active sampler options",
      activeWhen: "logits",
    },
    {
      id: "stream",
      name: limited ? "RESPONSE STREAM" : "STREAM DISPATCH",
      detail: limited ? "Provider chunks" : `${generatedTokens.length} generated model tokens`,
      subtext: limited ? "token count finalized by runtime" : "WebSocket stream",
      activeWhen: "always",
    },
  ]

  const getStageStatus = (stage: PipelineStage) => {
    if (limited && ["embed", "attn", "mlp", "logits"].includes(stage.id)) return { text: "UNAVAILABLE", color: "idle" }
    if (!running) return { text: "IDLE", color: "idle" }
    if (stage.activeWhen === "logits" && currentStep > 0) return { text: "ACTIVE", color: "active" }
    if (stage.activeWhen === "inferencing") return { text: "COMPUTING", color: "computing" }
    return { text: "RUNNING", color: "running" }
  }

  return (
    <div className="telemetry-card pipeline-status-card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <span className="card-title">PIPELINE OBSERVER</span>
          <span className={`badge-${running ? "amber" : "emerald"}`}>
            {running ? `EXEC STEP ${currentStep}` : "STANDBY"}
          </span>
        </div>
        <span className="text-xs text-muted mono">
          {limited ? `${provider ?? "provider"} runtime · limited inspection` : model?.architecture ?? "architecture unavailable"}
        </span>
      </div>

      <div className="pipeline-grid">
        {stages.map((st) => {
          const status = getStageStatus(st)
          return (
            <div key={st.id} className={`pipeline-stage-item ${status.color}`}>
              <div className="stage-top flex items-center justify-between">
                <span className="stage-name mono">{st.name}</span>
                <span className={`stage-badge mono badge-${status.color}`}>
                  {status.text}
                </span>
              </div>
              <div className="stage-detail mono text-xs">{st.detail}</div>
              <div className="stage-subtext text-muted text-xxs mono">{st.subtext}</div>
            </div>
          )
        })}
      </div>

      <div className="pipeline-hw-bar mono text-xs">
        <span className="hw-tag">ACCELERATOR:</span>
        <span className="hw-val text-cyan">{hardware?.gpu_name ?? hardware?.backend ?? "accelerator unavailable"}</span>
        <span className="hw-tag">TORCH:</span>
        <span className="hw-val">{hardware?.torch_version ?? "—"}</span>
        <span className="hw-tag">DEVICE:</span>
        <span className="hw-val">{model?.device ?? hardware?.device ?? "—"}</span>
      </div>
    </div>
  )
}
