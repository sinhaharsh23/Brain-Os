import { useMemo } from "react"
import { useBrain } from "../../store/useBrainStore"

export default function ThroughputGaugeCard() {
  const summary = useBrain((s) => s.summary)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const monitoring = useBrain((s) => s.monitoring)
  const hardware = useBrain((s) => s.hardware)
  const running = useBrain((s) => s.running)

  const { currentTps, peakTps, avgTps, sparklinePts } = useMemo(() => {
    let tps = 0
    if (summary?.timings?.tokens_per_second) {
      tps = Number(summary.timings.tokens_per_second)
    }

    const pts: number[] = []
    if (generatedTokens.length > 0) {
      for (const token of generatedTokens.slice(-16)) {
        const dt = token.time_ms / 1000
        if (dt > 0) pts.push(Math.min(120, 1 / dt))
      }
      if (pts.length > 0) {
        tps = pts[pts.length - 1]
      }
    } else if (tps > 0) {
      pts.push(tps * 0.9, tps * 0.95, tps * 1.05, tps)
    }

    const peak = pts.length > 0 ? Math.max(...pts) : tps
    const avg = pts.length > 0 ? pts.reduce((a, b) => a + b, 0) / pts.length : tps

    const maxVal = Math.max(1, peak * 1.15)
    const w = 160
    const h = 40
    const coords = pts.slice(-16).map((val, idx, arr) => {
      const x = (idx / Math.max(1, arr.length - 1)) * w
      const y = h - (val / maxVal) * (h - 8) - 4
      return { x, y }
    })

    let pathD = ""
    let areaD = ""
    if (coords.length > 0) {
      pathD = coords.map((c, i) => `${i === 0 ? "M" : "L"} ${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ")
      areaD = `${pathD} L ${coords[coords.length - 1].x.toFixed(1)},${h} L ${coords[0].x.toFixed(1)},${h} Z`
    }

    return {
      currentTps: tps,
      peakTps: peak,
      avgTps: avg,
      sparklinePts: { pathD, areaD, lastPt: coords[coords.length - 1] },
    }
  }, [summary, generatedTokens, running])

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
          <span className="badge-emerald">{hardware?.backend ?? "ACCELERATOR —"}</span>
        </div>
        <span className="text-xs text-muted mono">TELEMETRY DECK</span>
      </div>

      <div className="throughput-body">
        <div className="throughput-graph-col">
          <div className="tps-headline">
            <span className="tps-val mono">{currentTps > 0 ? currentTps.toFixed(1) : "—"}</span>
            <span className="tps-unit mono">TOK/S</span>
          </div>

          <div className="sparkline-container">
            <svg viewBox="0 0 160 40" className="sparkline-svg" preserveAspectRatio="none">
              <defs>
                <linearGradient id="tpsGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00d2ff" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#00d2ff" stopOpacity="0.0" />
                </linearGradient>
              </defs>
              {sparklinePts.areaD && (
                <path d={sparklinePts.areaD} fill="url(#tpsGradient)" />
              )}
              {sparklinePts.pathD && (
                <path d={sparklinePts.pathD} fill="none" stroke="#00d2ff" strokeWidth="1.75" />
              )}
              {sparklinePts.lastPt && (
                <circle
                  cx={sparklinePts.lastPt.x}
                  cy={sparklinePts.lastPt.y}
                  r="3"
                  fill="#00d2ff"
                  className={running ? "pulse-dot" : ""}
                />
              )}
            </svg>
          </div>

          <div className="tps-sub-stats mono text-xs">
            <span>PEAK: <b className="text-amber">{peakTps > 0 ? peakTps.toFixed(1) : "—"}</b></span>
            <span>AVG: <b className="text-cyan">{avgTps > 0 ? avgTps.toFixed(1) : "—"}</b></span>
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
                UNIFIED
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
