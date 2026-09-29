import TelemetryDetails from "../../components/TelemetryDetails"
import { useBrain } from "../../store/useBrainStore"

export default function PerformanceMetricsSection() {
  const telemetry = useBrain((s) => s.telemetry)
  const providerLogprobs = useBrain((s) => s.providerLogprobs)
  const model = useBrain((s) => s.model)
  const summary = useBrain((s) => s.summary)
  const monitoring = useBrain((s) => s.monitoring)
  const hardware = useBrain((s) => s.hardware)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const modelStatus = useBrain((s) => s.modelStatus)

  const inspectionMode = useBrain((s) => s.inspectionMode)
  const totalTokens = telemetry?.tokens.sequence_tokens ?? (inspectionMode === "limited" ? null : tokens.length + generatedTokens.length)
  const maxTokens = telemetry?.model.context_window ?? (inspectionMode === "limited" ? null : model?.context_length)
  const responseTimeSec = typeof summary?.timings?.total_ms === "number"
    ? (summary.timings.total_ms / 1000).toFixed(2)
    : "—"
  const throughputTps = typeof summary?.timings?.tokens_per_second === "number"
    ? summary.timings.tokens_per_second.toFixed(1)
    : "—"
  const selectedProbability = providerLogprobs.length ? Math.exp(providerLogprobs[providerLogprobs.length - 1].logprob) : generatedTokens.at(-1)?.probability
  const topConfidence = selectedProbability != null
    ? (selectedProbability * 100).toFixed(1)
    : "—"
  const ramUsed = monitoring?.ram_used_gb ?? hardware?.ram?.used_gb ?? null
  const ramTotal = monitoring?.ram_total_gb ?? hardware?.ram?.total_gb ?? null
  const cpuPercent = monitoring?.cpu_percent ?? hardware?.cpu?.percent ?? null

  const renderSparkline = (values: number[], strokeColor: string) => {
    if (!values.length || values.some((value) => !Number.isFinite(value))) return <span className="metric-spark-empty">NO DATA</span>
    const min = Math.min(...values)
    const max = Math.max(...values, min + 1)
    const pts = values
      .map((v, i) => {
        const x = (i / Math.max(1, values.length - 1)) * 44
        const y = 15 - ((v - min) / (max - min)) * 11
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })
      .join(" ")
    return (
      <svg viewBox="0 0 46 16" className="metric-spark-svg">
        <polyline fill="none" stroke={strokeColor} strokeWidth="1.5" points={pts} />
      </svg>
    )
  }

  return (
    <div className="bottom-metrics-split mono">
      {/* Section 4: System Performance Metrics (Bottom-Left) */}
      <div className="metrics-left-column">
        <div className="section-header-wrap flex items-center justify-between">
          <span className="section-title-tag">SYSTEM PERFORMANCE METRICS</span>
          <span className="text-xxs text-dim">SYSTEM TELEMETRY</span>
        </div>

        <div className="metrics-cards-grid">
          {/* Card 1: Response Time */}
          <div className="metric-stat-card">
            <div className="metric-card-top flex items-center justify-between">
              <span className="metric-name">RESPONSE TIME</span>
              {renderSparkline(responseTimeSec === "—" ? [] : [Number(responseTimeSec)], "#00d2ff")}
            </div>
            <div className="metric-main-val">
              <span className="text-bright">{responseTimeSec}</span>
              <span className="metric-unit">SEC</span>
            </div>
            <span className="metric-sub text-xxs text-dim">TTFT: {summary?.timings?.ttft_ms ?? "—"}{summary?.timings?.ttft_ms != null ? "ms" : ""}</span>
          </div>

          {/* Card 2: Tokens Processed */}
          <div className="metric-stat-card">
            <div className="metric-card-top flex items-center justify-between">
              <span className="metric-name">TOKENS PROCESSED</span>
          {renderSparkline(totalTokens ? [totalTokens] : [], "#ffb648")}
            </div>
            <div className="metric-main-val">
              <span className="text-amber">{totalTokens ?? "—"}</span>
              <span className="metric-unit">{maxTokens ? `/ ${maxTokens}` : totalTokens == null ? "UNAVAILABLE" : "TOKENS"}</span>
            </div>
            <span className="metric-sub text-xxs text-dim">Sequence tokens</span>
          </div>

          {/* Card 3: Throughput */}
          <div className="metric-stat-card">
            <div className="metric-card-top flex items-center justify-between">
              <span className="metric-name">END-TO-END TOKEN RATE</span>
              {renderSparkline(throughputTps === "—" ? [] : [Number(throughputTps)], "#22c55e")}
            </div>
            <div className="metric-main-val">
              <span className="text-emerald">{throughputTps}</span>
              <span className="metric-unit">E2E TOK/S</span>
            </div>
            <span className="metric-sub text-xxs text-dim">{model?.dtype ?? "—"}</span>
          </div>

          {/* Card 4: Top Confidence */}
          <div className="metric-stat-card">
            <div className="metric-card-top flex items-center justify-between">
              <span className="metric-name">SELECTED TOKEN PROBABILITY</span>
              {renderSparkline(topConfidence === "—" ? [] : [Number(topConfidence)], "#00d2ff")}
            </div>
            <div className="metric-main-val">
              <span className="text-cyan">{selectedProbability == null ? "—" : `${topConfidence}%`}</span>
              <span className="metric-unit">PROB</span>
            </div>
            <span className="metric-sub text-xxs text-dim">Probability of the selected token</span>
          </div>

          {/* Card 5: GPU Memory */}
          <div className="metric-stat-card">
            <div className="metric-card-top flex items-center justify-between">
              <span className="metric-name">SYSTEM UNIFIED MEMORY</span>
              {renderSparkline(ramUsed == null ? [] : [ramUsed], "#a855f7")}
            </div>
            <div className="metric-main-val">
              <span className="text-bright">{ramUsed == null ? "—" : ramUsed.toFixed(1)}</span>
              <span className="metric-unit">{ramTotal == null ? "GB" : `/ ${ramTotal.toFixed(0)} GB`}</span>
            </div>
            <span className="metric-sub text-xxs text-dim">Unified RAM</span>
          </div>

          {/* Card 6: Overall System Load */}
          <div className="metric-stat-card">
            <div className="metric-card-top flex items-center justify-between">
              <span className="metric-name">SYSTEM CPU UTILIZATION</span>
              {renderSparkline(cpuPercent == null ? [] : [cpuPercent], "#ffb648")}
            </div>
            <div className="metric-main-val">
              <span className="text-bright">{cpuPercent == null ? "—" : `${cpuPercent.toFixed(0)}%`}</span>
              <span className="metric-unit">UTIL</span>
            </div>
            <span className="metric-sub text-xxs text-dim">psutil system CPU</span>
          </div>
        </div>
      </div>

      {/* Section 5: Active Model Information (Bottom-Right) */}
      <div className="metrics-right-column">
        <div className="section-header-wrap flex items-center justify-between">
          <span className="section-title-tag">ACTIVE MODEL INFORMATION</span>
          <span className="badge-emerald text-xxs">{modelStatus === "loaded" ? "ONLINE" : "STANDBY"}</span>
        </div>

        <div className="active-model-card">
          <div className="model-card-top flex items-center justify-between">
            <span className="model-name-title text-bright truncate">
              {model?.model_id ?? "model metadata unavailable"}
            </span>
            <span className="status-badge-pill badge-emerald">Active & Ready</span>
          </div>

          <details><summary>PROVIDER / TOKEN COUNTS / TIMING SOURCES</summary><TelemetryDetails /></details>
          <div className="model-specs-table">
            <div className="spec-row flex items-center justify-between">
              <span className="text-dim">PARAMETERS:</span>
              <span className="text-bright">{model?.num_params ? `~${(model.num_params / 1e6).toFixed(0)}M` : "—"}</span>
            </div>
            <div className="spec-row flex items-center justify-between">
              <span className="text-dim">CONTEXT:</span>
              <span className="text-cyan">{model?.context_length?.toLocaleString() ?? "—"} TOKENS</span>
            </div>
            <div className="spec-row flex items-center justify-between">
              <span className="text-dim">HIDDEN DIM:</span>
              <span className="text-bright">{model?.hidden_size ?? "—"}</span>
            </div>
            <div className="spec-row flex items-center justify-between">
              <span className="text-dim">LAYERS / HEADS:</span>
              <span className="text-bright">{model ? `${model.num_layers} Layers · ${model.num_attention_heads}Q / ${model.num_kv_heads}KV` : "—"}</span>
            </div>
            <div className="spec-row flex items-center justify-between">
              <span className="text-dim">ACCELERATOR:</span>
              <span className="text-emerald">{model?.device ?? "Unavailable"}</span>
            </div>
            <div className="spec-row flex items-center justify-between">
              <span className="text-dim">STATUS:</span>
              <span className="text-emerald font-bold">{modelStatus === "loaded" ? "Active & Ready" : "unavailable"}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
