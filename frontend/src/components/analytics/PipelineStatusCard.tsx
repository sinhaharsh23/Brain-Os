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

  const stages: PipelineStage[] = [
    {
      id: "tok",
      name: "TOKENIZER",
      detail: `${tokens.length} prompt tok`,
      subtext: model ? `${model.vocab_size.toLocaleString()} vocabulary entries` : "metadata unavailable",
      activeWhen: "always",
    },
    {
      id: "embed",
      name: "EMBED & ROPE",
      detail: model ? `${model.hidden_size}d vector` : "—",
      subtext: model ? "rotary position embeddings" : "metadata unavailable",
      activeWhen: "always",
    },
    {
      id: "attn",
      name: "GQA ATTENTION",
      detail: model ? `${model.num_attention_heads}Q / ${model.num_kv_heads}KV` : "—",
      subtext: model ? "attention capture" : "metadata unavailable",
      activeWhen: "inferencing",
    },
    {
      id: "mlp",
      name: "SWIGLU MLP",
      detail: model ? `${model.intermediate_size} intermediate` : "—",
      subtext: model ? `${model.num_layers}-layer feedforward` : "metadata unavailable",
      activeWhen: "inferencing",
    },
    {
      id: "logits",
      name: "LM HEAD LOGITS",
      detail: "Top-k/p sampler",
      subtext: "Softmax temperature",
      activeWhen: "logits",
    },
    {
      id: "stream",
      name: "STREAM DISPATCH",
      detail: `${generatedTokens.length} gen tok`,
      subtext: "Async WebSocket SSE",
      activeWhen: "always",
    },
  ]

  const getStageStatus = (stage: PipelineStage) => {
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
          {model?.architecture ?? "architecture unavailable"}
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
