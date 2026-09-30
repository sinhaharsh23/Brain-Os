import TelemetryDetails from "./TelemetryDetails"
import { useBrain } from "../store/useBrainStore"

const names: Record<string, string> = {
  ollama: "Ollama",
  gemini: "Google Gemini",
  openai: "OpenAI / ChatGPT",
  anthropic: "Anthropic / Claude",
  google: "Google Gemini",
}

export default function ExternalObservationNotice() {
  const provider = useBrain((s) => s.provider)
  const providerModel = useBrain((s) => s.providerModel)
  const usage = useBrain((s) => s.usage)
  const summary = useBrain((s) => s.summary)

  return (
    <div className="card" style={{ margin: 12, maxWidth: 760 }}>
      <div className="panel-title" style={{ marginTop: 0, color: "var(--warn)" }}>
        Runtime Observation · {names[provider ?? ""] ?? provider ?? "provider"}
      </div>
      <p className="muted">
        Model: <span className="mono">{providerModel || "provider default"}</span>. The provider API exposes response text and metadata only; BrainOS does not invent private tokens, embeddings, hidden states, attention, Q/K/V, logits, or MLP activations.
      </p>
      {provider === "ollama" && <p className="muted">BrainOS uses the installed Ollama GGUF checkpoint with its instrumented llama.cpp runtime to capture real token IDs and transformer signals.</p>}
      <TelemetryDetails />
      {(usage || summary?.timings) && (
        <div className="kv">
          <span className="k">time</span>
          <span className="v">{summary?.timings.total_ms ?? "—"} ms</span>
          <span className="k">first chunk</span>
          <span className="v">{summary?.timings.ttft_ms ?? "—"} ms</span>
          <span className="k">input tokens</span>
          <span className="v">{usage?.input_tokens ?? "not exposed"}</span>
          <span className="k">output tokens</span>
          <span className="v">{usage?.output_tokens ?? "not exposed"}</span>
          <span className="k">total tokens</span>
          <span className="v">{usage?.total_tokens ?? "not exposed"}</span>
        </div>
      )}
    </div>
  )
}
