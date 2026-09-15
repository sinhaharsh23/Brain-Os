import { useBrain } from "../store/useBrainStore"
import ExternalObservationNotice from "../components/ExternalObservationNotice"

export default function TokenFlowView() {
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const selectedToken = useBrain((s) => s.selectedToken)
  const selectToken = useBrain((s) => s.selectToken)
  const pca = useBrain((s) => s.pca)
  const running = useBrain((s) => s.running)
  const currentStep = useBrain((s) => s.currentStep)
  const inspectionMode = useBrain((s) => s.inspectionMode)

  if (inspectionMode === "limited") return <ExternalObservationNotice />

  return (
    <div style={{ padding: 12, overflowY: "auto", height: "100%" }}>
      <div className="panel-title" style={{ marginTop: 0 }}>
        Tokenization — real tokenizer output ({tokens.length} tokens{pca ? ` · ${pca.pca_method} embedding points ready` : ""})
      </div>
      <div className="token-strip">
        {tokens.map((t) => (
          <span
            key={t.position}
            className={`token-chip ${t.is_special ? "special" : ""} ${selectedToken === t.position ? "selected" : ""}`}
            title={`pos ${t.position} · id ${t.id}`}
            onClick={() => selectToken(selectedToken === t.position ? null : t.position)}
          >
            {t.position}:{t.text}
          </span>
        ))}
      </div>

      <div className="panel-title">Token IDs</div>
      <div className="mono" style={{ fontSize: 11, color: "var(--muted)", wordBreak: "break-all", lineHeight: 1.7 }}>
        [{tokens.map((t) => t.id).join(", ")}{generatedTokens.length ? ", " + generatedTokens.map((t) => t.token_id).join(", ") : ""}]
      </div>

      <div className="panel-title">Generation stream {running ? `· step ${currentStep} ▮` : ""}</div>
      <div className="token-strip">
        {generatedTokens.map((t) => (
          <span
            key={t.step}
            className={`token-chip generated ${selectedToken === t.position ? "selected" : ""}`}
            title={`step ${t.step} · id ${t.token_id} · p=${(t.probability * 100).toFixed(1)}% · rank ${t.rank} · ${t.time_ms}ms`}
            onClick={() => selectToken(selectedToken === t.position ? null : t.position)}
          >
            {t.position}:{t.text}
          </span>
        ))}
      </div>

      <div className="panel-title">Token probabilities</div>
      <div className="kv" style={{ gridTemplateColumns: "60px 1fr 90px 90px" }}>
        {generatedTokens.map((t) => (
          <div className="kv" key={t.step} style={{ gridTemplateColumns: "60px 1fr 90px 90px", gridColumn: "1 / -1" }}>
            <span className="k">step {t.step}</span>
            <span className="v" style={{ textAlign: "left" }}>{t.text}</span>
            <span className="v">p={(t.probability * 100).toFixed(1)}%</span>
            <span className="v">rank {t.rank}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
