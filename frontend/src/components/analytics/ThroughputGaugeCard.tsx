import { useBrain } from "../../store/useBrainStore"

export default function ThroughputGaugeCard() {
  const summary = useBrain((s) => s.summary)
  const telemetry = useBrain((s) => s.telemetry)
  const monitoring = useBrain((s) => s.monitoring)
  const hardware = useBrain((s) => s.hardware)
  const decodeTps = telemetry?.performance.decode_tokens_per_second ?? null
  const endToEndTps = telemetry?.performance.end_to_end_tokens_per_second
    ?? (typeof summary?.timings?.tokens_per_second === "number" ? summary.timings.tokens_per_second : null)
  const displayTps = decodeTps ?? endToEndTps

  const ramPercent = monitoring?.ram_percent ?? hardware?.ram?.percent ?? null
  const ramUsedGb = monitoring?.ram_used_gb ?? hardware?.ram?.used_gb ?? null
  const ramTotalGb = monitoring?.ram_total_gb ?? hardware?.ram?.total_gb ?? null

  const dialRadius = 36
  const circumference = 2 * Math.PI * dialRadius
  const dialStrokeDashoffset = circumference - ((ramPercent ?? 0) / 100) * circumference

  return (
    <div className="telemetry-card throughput-gauge-card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <span className="card-title">THROUGHPUT & MEMORY</span>
          <span className="badge-outline">{hardware?.backend ?? "DEVICE UNAVAILABLE"}</span>
        </div>
        <span className="text-xs text-muted mono">TELEMETRY DECK</span>
      </div>

      <div className="throughput-body">
        <div className="throughput-graph-col">
          <div className="tps-headline">
            <span className="tps-val mono">{displayTps == null ? "—" : displayTps.toFixed(1)}</span>
            <span className="tps-unit mono">{decodeTps != null ? "DECODE TOK/S" : "E2E TOK/S"}</span>
          </div>

          <div className="sparkline-container">
            <svg viewBox="0 0 160 40" className="sparkline-svg" preserveAspectRatio="none">
              <defs>
                <linearGradient id="tpsGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00d2ff" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#00d2ff" stopOpacity="0.0" />
                </linearGradient>
              </defs>
              {displayTps != null && <line x1="0" y1="20" x2="160" y2="20" stroke="#00d2ff" strokeWidth="2" />}
            </svg>
          </div>

          <div className="tps-sub-stats mono text-xs">
            <span>DECODE: <b className="text-amber">{decodeTps == null ? "Unavailable" : decodeTps.toFixed(1)}</b></span>
            <span>END-TO-END: <b className="text-cyan">{endToEndTps == null ? "Unavailable" : endToEndTps.toFixed(1)}</b></span>
          </div>
        </div>

        <div className="gauge-dial-col">
          <div className="gauge-svg-wrap">
            <svg viewBox="0 0 96 96" className="gauge-svg">
              <circle
                cx="48"
                cy="48"
                r={dialRadius}
                fill="none"
                stroke="rgba(255, 255, 255, 0.08)"
                strokeWidth="6"
              />
              <circle
                cx="48"
                cy="48"
                r={dialRadius}
                fill="none"
                stroke={ramPercent === null ? "#38516a" : ramPercent > 80 ? "#ef4444" : ramPercent > 65 ? "#ffb648" : "#00d2ff"}
                strokeWidth="6"
                strokeDasharray={circumference}
                strokeDashoffset={dialStrokeDashoffset}
                strokeLinecap="round"
                transform="rotate(-90 48 48)"
                className="gauge-arc"
              />
              <text
                x="48"
                y="45"
                fill="var(--text-bright)"
                fontSize="13"
                fontWeight="700"
                fontFamily="var(--mono)"
                textAnchor="middle"
              >
                {ramPercent === null ? "—" : `${ramPercent.toFixed(0)}%`}
              </text>
              <text
                x="48"
                y="57"
                fill="var(--text-dim)"
                fontSize="7.5"
                fontFamily="var(--mono)"
                textAnchor="middle"
              >
                {hardware?.backend === "MPS" ? "UNIFIED" : "SYSTEM RAM"}
              </text>
            </svg>
          </div>

          <div className="gauge-sub-stats mono text-xs">
            <span>{ramUsedGb === null || ramTotalGb === null ? "memory —" : `${ramUsedGb.toFixed(1)} / ${ramTotalGb.toFixed(0)} GB`}</span>
            <span className="badge-outline">{hardware?.gpu?.vendor ?? "DEVICE"}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
