import { useMemo, useState } from "react"
import { useBrain } from "../../store/useBrainStore"

export default function ActionHeatCard() {
  const layers = useBrain((s) => s.layers)
  const layersByStep = useBrain((s) => s.layersByStep)
  const running = useBrain((s) => s.running)
  const currentStep = useBrain((s) => s.currentStep)
  const model = useBrain((s) => s.model)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectLayer = useBrain((s) => s.selectLayer)
  const inspectionMode = useBrain((s) => s.inspectionMode)

  const [hoveredCell, setHoveredCell] = useState<{ layer: number; step: number; norm: number; intensity: number } | null>(null)

  const observedSteps = useMemo(() => Object.keys(layersByStep).map(Number).sort((a, b) => a - b).slice(-12), [layersByStep])
  const observedLayerCount = Math.max(0, ...Object.values(layersByStep).flatMap((byLayer) => Object.keys(byLayer).map(Number).map((layer) => layer + 1)))
  const numLayers = model?.num_layers ?? observedLayerCount
  const maxNorm = Math.max(1e-12, ...observedSteps.flatMap((step) => Object.values(layersByStep[step] ?? {}).map((layer) => Math.abs(layer.norm))))

  const grid = useMemo(() => {
    const matrix: number[][] = []
    for (let l = 0; l < numLayers; l++) {
      matrix.push(observedSteps.map((step) => {
        const captured = layersByStep[step]?.[l]
        return captured ? Math.min(1, Math.abs(captured.norm) / maxNorm) : -1
      }))
    }
    return matrix
  }, [layersByStep, observedSteps, numLayers, maxNorm])

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
          <span className="badge-cyan">{inspectionMode === "limited" ? "INTERNAL CAPTURE UNAVAILABLE" : `${Object.keys(layers).length}/${numLayers || "—"} LAYERS`}</span>
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
            {inspectionMode !== "limited" && grid.map((row, lIdx) => {
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
                    {row.map((intensity, cIdx) => {
                      const step = observedSteps[cIdx]
                      const captured = layersByStep[step]?.[lIdx]
                      const isCurrentCol = running && step === currentStep && captured != null
                      return (
                        <div
                          key={cIdx}
                          className={`heat-cell ${isCurrentCol ? "col-active" : ""}`}
                          style={{
                            backgroundColor: intensity < 0 ? "rgba(14, 28, 48, 0.25)" : getCellColor(intensity, isCurrentCol),
                            boxShadow: isCurrentCol ? "0 0 6px rgba(255, 182, 72, 0.6)" : intensity > 0.8 ? "0 0 4px rgba(0, 210, 255, 0.4)" : "none",
                          }}
                          onMouseEnter={() => captured && setHoveredCell({ layer: lIdx, step, norm: captured.norm, intensity })}
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
              LAYER {hoveredCell.layer} · STEP {hoveredCell.step} · OUTPUT NORM {hoveredCell.norm.toFixed(3)}
            </span>
          ) : (
            <span className="text-muted">
              {inspectionMode === "limited" ? "INTERNAL LAYER DATA UNAVAILABLE THROUGH PROVIDER" : `CAPTURED LAYER OUTPUT NORMS · COLOR RELATIVE TO MAX CAPTURED NORM · ${observedSteps.length} REAL STEPS`}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
