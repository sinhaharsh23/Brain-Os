import { useState } from "react"
import { useBrain } from "../../store/useBrainStore"

export default function ResponseHUD() {
  const response = useBrain((s) => s.response)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const running = useBrain((s) => s.running)
  const summary = useBrain((s) => s.summary)
  const currentStep = useBrain((s) => s.currentStep)
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    if (!response) return
    navigator.clipboard.writeText(response)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const getProbColor = (p: number) => {
    if (p >= 0.7) return "prob-high"
    if (p >= 0.4) return "prob-med"
    return "prob-low"
  }

  return (
    <div className="telemetry-card response-hud-card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <span className="card-title">RESPONSE HUD</span>
          <span className={`badge-${running ? "amber" : response ? "emerald" : "outline"}`}>
            {running ? `EMITTING STEP ${currentStep}` : response ? "STREAM COMPLETE" : "AWAITING INPUT"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {summary && (
            <span className="text-xs text-muted mono">
              {summary.num_output_tokens} TOK · {summary.timings?.tokens_per_second ?? "—"} TOK/S · TTFT {summary.timings?.ttft_ms ?? "—"}MS
            </span>
          )}
          {response && (
            <button className="hud-copy-btn mono text-xs" onClick={copyToClipboard} title="Copy response to clipboard">
              {copied ? "✓ COPIED" : "COPY"}
            </button>
          )}
        </div>
      </div>

      {generatedTokens.length > 0 && (
        <div className="hud-token-chips">
          {generatedTokens.slice(-14).map((tok) => (
            <span
              key={tok.step}
              className={`hud-token-chip mono ${getProbColor(tok.probability)}`}
              title={`Step ${tok.step} · id:${tok.token_id} · p=${(tok.probability * 100).toFixed(1)}%`}
            >
              {tok.text === "\n" ? "\\n" : tok.text}
              <span className="chip-p">{(tok.probability * 100).toFixed(0)}%</span>
            </span>
          ))}
          {running && (
            <span className="hud-token-chip chip-streaming mono">
              ▮
            </span>
          )}
        </div>
      )}

      <div className="hud-terminal-body mono">
        {response ? (
          <div className="hud-text-content">
            {response}
            {running && <span className="streaming-cursor">▮</span>}
          </div>
        ) : (
          <div className="hud-placeholder text-muted">
            {running ? "Initializing generation on MPS..." : "— awaiting prompt dispatch from neural console —"}
          </div>
        )}
      </div>
    </div>
  )
}
