import { useMockTrainingMetrics } from "../hooks/useMocks"

export default function TrainingHubView() {
  const { metrics, currentEpoch, totalEpochs } = useMockTrainingMetrics()

  return (
    <div className="section-page-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div>
          <h2 className="section-title">TRAINING HUB (MODEL ADAPTATION & LORA)</h2>
          <span className="section-subtitle text-dim text-xs">
            Fine-tuning checkpoints, loss telemetry, and learning rate scheduling
          </span>
        </div>
        <span className="badge-outline text-xxs">TODO: TRAINING LOSS STREAM</span>
      </div>

      <div className="training-hub-grid">
        <div className="telemetry-card">
          <div className="card-header">
            <span className="card-title">EPOCH TRAINING TELEMETRY (LORA ADAPTER)</span>
            <span className="badge-emerald">EPOCH {currentEpoch} / {totalEpochs}</span>
          </div>

          <table className="mini-data-table">
            <thead>
              <tr>
                <th>EPOCH</th>
                <th>TRAIN LOSS</th>
                <th>VAL LOSS</th>
                <th>PERPLEXITY</th>
                <th>LEARNING RATE</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => (
                <tr key={m.epoch}>
                  <td className="text-dim">#{m.epoch}</td>
                  <td className="text-cyan font-bold">{m.loss.toFixed(3)}</td>
                  <td className="text-amber">{m.valLoss.toFixed(3)}</td>
                  <td className="text-emerald">{m.perplexity.toFixed(1)}</td>
                  <td className="text-dim text-xxs">{m.learningRate.toExponential(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
