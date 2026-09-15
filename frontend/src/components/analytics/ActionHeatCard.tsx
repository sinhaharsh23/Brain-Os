import { useMemo, useState } from "react"
import { useBrain } from "../../store/useBrainStore"

export default function ActionHeatCard() {
  const layers = useBrain((s) => s.layers)
  const mlpTop = useBrain((s) => s.mlpTop)
  const currentStep = useBrain((s) => s.currentStep)
  const running = useBrain((s) => s.running)
  const model = useBrain((s) => s.model)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectLayer = useBrain((s) => s.selectLayer)

  const [hoveredCell, setHoveredCell] = useState<{ layer: number; step: number; val: number } | null>(null)

  const numLayers = model?.num_layers ?? Object.keys(layers).length
  const numCols = 12

  const grid = useMemo(() => {
    const matrix: number[][] = []
    for (let l = 0; l < numLayers; l++) {
      const row: number[] = []
      const lState = layers[l]
      const mlp = mlpTop[l]

      const baseIntensity = lState ? Math.min(1, Math.abs(lState.norm) / 25) : 0
      const topMlpVal = mlp?.top?.[0]?.value ?? 0

      for (let c = 0; c < numCols; c++) {
        const stepOffset = (currentStep % numCols)
        let cellVal = baseIntensity
        if (c === stepOffset && running) {
          cellVal = Math.min(1.0, cellVal + 0.35)
        }
        if (topMlpVal > 2.0 && (l % 4 === 0)) {
          cellVal = Math.min(1.0, cellVal * 1.2)
        }
        row.push(Math.max(0.05, Math.min(1.0, cellVal)))
      }
      matrix.push(row)
    }
    return matrix
  }, [layers, mlpTop, currentStep, running, model])

  const getCellColor = (val: number, isCurrent: boolean) => {
    if (isCurrent) return "rgba(255, 182, 72, 0.9)"
    if (val > 0.8) return "rgba(0, 210, 255, 0.95)"
    if (val > 0.6) return "rgba(0, 160, 230, 0.75)"
    if (val > 0.4) return "rgba(14, 116, 178, 0.55)"
    if (val > 0.2) return "rgba(18, 55, 95, 0.45)"
    return "rgba(14, 28, 48, 0.4)"
  }

  return (
    <div className="telemetry-card action-heat-card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <span className="card-title">ACTION HEAT</span>
          <span className="badge-cyan">{Object.keys(layers).length}/{numLayers || "—"} LAYERS</span>
        </div>
        <div className="heat-legend flex items-center gap-1 mono text-xs">
          <span>0.0</span>
          <div className="heat-ramp" />
          <span>1.0</span>
        </div>
      </div>

      <div className="heat-body">
        <div className="heat-matrix-container">
          <div className="heat-matrix-grid">
            {grid.map((row, lIdx) => {
              const isLayerSelected = selectedLayer === lIdx
              return (
                <div key={lIdx} className={`heat-row ${isLayerSelected ? "selected-row" : ""}`}>
                  <span
                    className="heat-row-label mono"
                    onClick={() => selectLayer(lIdx)}
                    title={`Layer ${lIdx} - Click to select`}
                  >
                    L{String(lIdx).padStart(2, "0")}
                  </span>
                  <div className="heat-cells">
                    {row.map((val, cIdx) => {
                      const isCurrentCol = running && cIdx === (currentStep % numCols)
                      return (
                        <div
                          key={cIdx}
                          className={`heat-cell ${isCurrentCol ? "col-active" : ""}`}
                          style={{
                            backgroundColor: getCellColor(val, isCurrentCol),
                            boxShadow: isCurrentCol ? "0 0 6px rgba(255, 182, 72, 0.6)" : val > 0.8 ? "0 0 4px rgba(0, 210, 255, 0.4)" : "none",
                          }}
                          onMouseEnter={() => setHoveredCell({ layer: lIdx, step: cIdx, val })}
                          onMouseLeave={() => setHoveredCell(null)}
                          onClick={() => selectLayer(lIdx)}
                        />
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="heat-footer mono text-xs">
          {hoveredCell ? (
            <span className="text-cyan">
              LAYER {hoveredCell.layer} · BIN {hoveredCell.step} · INTENSITY {hoveredCell.val.toFixed(3)}
            </span>
          ) : (
            <span className="text-muted">
              LAYER NORMS + MLP CAPTURES · STEP {currentStep} {running ? "· COMPUTING" : "· READY"}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
