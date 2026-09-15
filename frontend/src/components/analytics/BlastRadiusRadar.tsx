import { useMemo } from "react"
import { useBrain } from "../../store/useBrainStore"

interface AxisInfo {
  label: string
  headIndex: number
  angleDeg: number
  value: number
}

export default function BlastRadiusRadar() {
  const attentionLinks = useBrain((s) => s.attentionLinks)
  const running = useBrain((s) => s.running)
  const selectedHead = useBrain((s) => s.selectedHead)
  const selectHead = useBrain((s) => s.selectHead)
  const currentStep = useBrain((s) => s.currentStep)
  const model = useBrain((s) => s.model)

  // Radar geometry: Center (120, 110), Radius 75
  const cx = 120
  const cy = 110
  const r = 75

  // 6 radial axes representing multi-head attention clusters
  const axes = useMemo<AxisInfo[]>(() => {
    const headWeights = [0, 0, 0, 0, 0, 0]
    
    const linkKeys = Object.keys(attentionLinks).map(Number)
    if (linkKeys.length > 0) {
      const allLinks = linkKeys.flatMap((k) => attentionLinks[k] || [])
      if (allLinks.length > 0) {
        for (let i = 0; i < 6; i++) {
          const matching = allLinks.filter((l) => l.head % 6 === i)
          if (matching.length > 0) {
            const sum = matching.reduce((acc, l) => acc + (l.weight || 0), 0)
            const avg = sum / matching.length
            headWeights[i] = Math.max(0.15, Math.min(1.0, avg * 1.5 + 0.2))
          }
        }
      }
    }

    const headCount = model?.num_attention_heads ?? 0
    const labels = Array.from({ length: 6 }, (_, idx) => {
      if (!headCount) return `H${String(idx).padStart(2, "0")}`
      const start = Math.floor(idx * headCount / 6)
      const end = Math.max(start, Math.floor((idx + 1) * headCount / 6) - 1)
      return `H${String(start).padStart(2, "0")}-${String(end).padStart(2, "0")}`
    })
    return labels.map((label, idx) => {
      const angleDeg = -90 + idx * 60
      return {
        label,
        headIndex: idx * 2,
        angleDeg,
        value: headWeights[idx],
      }
    })
  }, [attentionLinks, running, currentStep, model])

  // Compute polygon points for data
  const dataPoints = axes.map((axis) => {
    const rad = (axis.angleDeg * Math.PI) / 180
    const dist = r * axis.value
    const x = cx + dist * Math.cos(rad)
    const y = cy + dist * Math.sin(rad)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(" ")

  const gridLevels = [0.25, 0.5, 0.75, 1.0]
  const maxVal = Math.max(...axes.map((a) => a.value))
  const topAxis = axes.find((a) => a.value === maxVal)
  const dispersion = (axes.reduce((a, b) => a + Math.abs(b.value - 0.5), 0) / axes.length).toFixed(2)

  return (
    <div className="telemetry-card blast-radius-card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <span className="card-title">BLAST RADIUS</span>
          <span className="badge-amber">ATTN RADAR</span>
        </div>
          <span className="text-xs text-muted mono">{model?.num_attention_heads ?? "—"} HEADS · {Object.keys(attentionLinks).length ? "CAPTURED" : "NO CAPTURE"}</span>
      </div>

      <div className="radar-body">
        <div className="radar-svg-wrap">
          <svg viewBox="0 0 240 220" className="radar-svg">
            {gridLevels.map((lvl) => {
              const pts = [0, 60, 120, 180, 240, 300].map((deg) => {
                const rad = ((deg - 90) * Math.PI) / 180
                const gx = cx + r * lvl * Math.cos(rad)
                const gy = cy + r * lvl * Math.sin(rad)
                return `${gx.toFixed(1)},${gy.toFixed(1)}`
              }).join(" ")
              return (
                <polygon
                  key={lvl}
                  points={pts}
                  fill="none"
                  stroke={lvl === 1.0 ? "rgba(0, 210, 255, 0.35)" : "rgba(255, 255, 255, 0.08)"}
                  strokeWidth={lvl === 1.0 ? 1 : 0.75}
                  strokeDasharray={lvl < 1.0 ? "2,3" : undefined}
                />
              )
            })}

            {axes.map((axis) => {
              const rad = (axis.angleDeg * Math.PI) / 180
              const x2 = cx + r * Math.cos(rad)
              const y2 = cy + r * Math.sin(rad)
              const lx = cx + (r + 16) * Math.cos(rad)
              const ly = cy + (r + 16) * Math.sin(rad)
              const isSelected = selectedHead === axis.headIndex
              return (
                <g
                  key={axis.label}
                  onClick={() => selectHead(isSelected ? null : axis.headIndex)}
                  style={{ cursor: "pointer" }}
                >
                  <line
                    x1={cx}
                    y1={cy}
                    x2={x2}
                    y2={y2}
                    stroke={isSelected ? "#00d2ff" : "rgba(255, 255, 255, 0.12)"}
                    strokeWidth={isSelected ? "1.5" : "1"}
                  />
                  <text
                    x={lx}
                    y={ly + 3}
                    fill={isSelected ? "#00d2ff" : "var(--text-dim)"}
                    fontSize="8.5"
                    fontFamily="var(--mono)"
                    textAnchor="middle"
                    className="radar-label"
                  >
                    {axis.label}
                  </text>
                </g>
              )
            })}

            <polygon
              points={dataPoints}
              fill="rgba(0, 210, 255, 0.22)"
              stroke="#00d2ff"
              strokeWidth="1.75"
              className="radar-poly"
            />

            {axes.map((axis) => {
              const rad = (axis.angleDeg * Math.PI) / 180
              const px = cx + r * axis.value * Math.cos(rad)
              const py = cy + r * axis.value * Math.sin(rad)
              const isTop = axis === topAxis
              const isSelected = selectedHead === axis.headIndex
              return (
                <circle
                  key={axis.label}
                  cx={px}
                  cy={py}
                  r={isSelected ? 4.5 : isTop ? 3.5 : 2.5}
                  fill={isSelected ? "#00d2ff" : isTop ? "#ffb648" : "#00d2ff"}
                  stroke="#080c14"
                  strokeWidth="1"
                  className="radar-dot"
                  onClick={() => selectHead(isSelected ? null : axis.headIndex)}
                  style={{ cursor: "pointer" }}
                />
              )
            })}

            <circle cx={cx} cy={cy} r="2" fill="#00d2ff" opacity="0.6" />
          </svg>
        </div>

        <div className="radar-stats mono">
          <div className="stat-pill">
            <span className="stat-pill-label">PEAK HEAD</span>
            <span className="stat-pill-val text-amber">{topAxis?.label ?? "H04"}</span>
          </div>
          <div className="stat-pill">
            <span className="stat-pill-label">CONCENTRATION</span>
            <span className="stat-pill-val text-cyan">{(maxVal * 100).toFixed(0)}%</span>
          </div>
          <div className="stat-pill">
            <span className="stat-pill-label">DISPERSION</span>
            <span className="stat-pill-val">{dispersion} σ</span>
          </div>
          {!Object.keys(attentionLinks).length && <span className="text-xs text-muted">attention capture unavailable</span>}
        </div>
      </div>
    </div>
  )
}
