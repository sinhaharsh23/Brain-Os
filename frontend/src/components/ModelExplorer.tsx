import { useEffect, useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"
import type { HardwareReport, ProviderDescriptor } from "../types"

export default function ModelExplorer() {
  const model = useBrain((s) => s.model)
  const modelStatus = useBrain((s) => s.modelStatus)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const selectedToken = useBrain((s) => s.selectedToken)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectToken = useBrain((s) => s.selectToken)
  const selectLayer = useBrain((s) => s.selectLayer)
  const selectHead = useBrain((s) => s.selectHead)
  const selectedHead = useBrain((s) => s.selectedHead)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  const [hardware, setHardware] = useState<HardwareReport | null>(null)
  const [models, setModels] = useState<{ model_id: string; params_m: number; description: string; verification_status: string; verification_note: string }[]>([])
  const [providers, setProviders] = useState<ProviderDescriptor[]>([])
  const [loadError, setLoadError] = useState("")

  useEffect(() => {
    api.hardware().then(setHardware).catch(() => {})
    api.models()
      .then((m) => setModels(m.supported))
      .catch(() => {})
    api.providers().then(setProviders).catch(() => {})
  }, [])

  const loadModel = async (id: string) => {
    setLoadError("")
    try {
      await api.loadModel(id)
    } catch (e) {
      setLoadError(String(e))
    }
  }

  const allTokens = [
    ...tokens.map((t) => ({ text: t.text, position: t.position, generated: false, id: t.id })),
    ...generatedTokens.map((t) => ({ text: t.text, position: t.position, generated: true, id: t.token_id })),
  ]

  return (
    <div>
      <div className="panel-title">Model</div>
      <div className="card">
        <div className="kv">
          <span className="k">status</span>
          <span className="v">{modelStatus}</span>
          <span className="k">architecture</span>
          <span className="v">{model?.architecture ?? "—"}</span>
          <span className="k">params</span>
          <span className="v">{model ? `${(model.num_params / 1e6).toFixed(1)}M` : "—"}</span>
          <span className="k">layers</span>
          <span className="v">{model?.num_layers ?? "—"}</span>
          <span className="k">heads</span>
          <span className="v">{model?.num_attention_heads ?? "—"}</span>
          <span className="k">head dim</span>
          <span className="v">{model?.head_dim ?? "—"}</span>
          <span className="k">hidden</span>
          <span className="v">{model?.hidden_size ?? "—"}</span>
          <span className="k">MLP dim</span>
          <span className="v">{model?.intermediate_size ?? "—"}</span>
          <span className="k">vocab</span>
          <span className="v">{model?.vocab_size ?? "—"}</span>
          <span className="k">context</span>
          <span className="v">{model?.context_length ?? "—"}</span>
          <span className="k">dtype</span>
          <span className="v">{model?.dtype ?? "—"}</span>
          <span className="k">device</span>
          <span className="v">{model?.device ?? "—"}</span>
          <span className="k">introspection</span>
          <span className="v">{model ? Object.entries(model.capabilities ?? {}).filter(([, supported]) => supported).map(([name]) => name).join(", ") : "—"}</span>
        </div>
      </div>

      <div className="panel-title">Hardware</div>
      <div className="card">
        <div className="kv">
          <span className="k">CPU</span>
          <span className="v">{hardware ? `${hardware.cpu.model} · ${hardware.cpu.count} cores` : "—"}</span>
          <span className="k">backend</span>
          <span className="v">{hardware?.backend ?? "—"}</span>
          <span className="k">model device</span>
          <span className="v">{model?.device ?? hardware?.model_device ?? "—"}</span>
          <span className="k">RAM</span>
          <span className="v">{hardware ? `${hardware.ram.total_gb} GB` : "—"}</span>
          <span className="k">GPU</span>
          <span className="v">{hardware?.gpu ? `${hardware.gpu.vendor} · ${hardware.gpu.name}` : (hardware ? "none detected" : "—")}</span>
          <span className="k">VRAM</span>
          <span className="v">{hardware?.gpu?.vram_total_gb !== null && hardware?.gpu?.vram_total_gb !== undefined ? `${hardware.gpu.vram_total_gb} GB` : "—"}</span>
          <span className="k">PyTorch access</span>
          <span className="v">{hardware ? (hardware.gpu_available ? "available" : "CPU fallback") : "—"}</span>
        </div>
      </div>

      <div className="panel-title">Load model</div>
      <div className="card">
        {models.map((m) => (
          <div key={m.model_id} className="flex spread" style={{ margin: "4px 0" }}>
            <div>
              <div className="mono" style={{ fontSize: 11 }}>
                {m.model_id.split("/")[1]}
              </div>
              <div className="muted" style={{ fontSize: 10 }}>
                {m.params_m}M · {m.description.slice(0, 60)}
              </div>
              <div className="muted" style={{ fontSize: 10, color: m.verification_status === "verified" ? "var(--ok)" : "var(--warn)" }}>
                {m.verification_status} · {m.verification_note}
              </div>
            </div>
            <button className="btn small" onClick={() => loadModel(m.model_id)} disabled={modelStatus === "loading" || model?.model_id === m.model_id}>
              {model?.model_id === m.model_id ? "ACTIVE" : "LOAD"}
            </button>
          </div>
        ))}
        {loadError && <div className="err-box">{loadError}</div>}
      </div>

      <div className="panel-title">Provider observability</div>
      <div className="card">
        {providers.map((provider) => (
          <div key={provider.provider_id} style={{ margin: "5px 0" }}>
            <div className="flex spread">
              <span className="mono" style={{ fontSize: 11 }}>{provider.display_name}</span>
              <span style={{ color: provider.inspection_mode === "deep" ? "var(--ok)" : "var(--warn)", fontSize: 10 }}>
                {provider.inspection_mode === "deep" ? "DEEP" : "LIMITED"} · {provider.availability}
              </span>
            </div>
            <div className="muted" style={{ fontSize: 10 }}>{provider.limitation}</div>
          </div>
        ))}
      </div>

      {model && inspectionMode !== "limited" && (
        <>
          <div className="panel-title">Layers</div>
          <div className="card">
            <div className="flex" style={{ flexWrap: "wrap", gap: 4 }}>
              {Array.from({ length: model.num_layers }, (_, i) => (
                <button
                  key={i}
                  className="btn small"
                  style={{
                    padding: "2px 6px",
                    borderColor: selectedLayer === i ? "var(--accent2)" : "var(--border)",
                    color: selectedLayer === i ? "var(--accent2)" : "var(--muted)",
                  }}
                  onClick={() => {
                    selectLayer(selectedLayer === i ? null : i)
                    selectHead(0)
                  }}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
          <div className="panel-title">Heads</div>
          <div className="card">
            <div className="flex" style={{ flexWrap: "wrap", gap: 4 }}>
              {Array.from({ length: model.num_attention_heads }, (_, i) => (
                <button
                  key={i}
                  className="btn small"
                  style={{
                    padding: "2px 6px",
                    borderColor: selectedHead === i ? "var(--accent)" : "var(--border)",
                    color: selectedHead === i ? "var(--accent)" : "var(--muted)",
                  }}
                  onClick={() => selectHead(selectedHead === i ? null : i)}
                >
                  H{i}
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      <div className="panel-title">Tokens ({allTokens.length})</div>
      <div className="card">
        <div className="token-strip" style={{ maxHeight: 220, overflowY: "auto" }}>
          {allTokens.map((t) => (
            <span
              key={t.position}
              className={`token-chip ${t.generated ? "generated" : ""} ${selectedToken === t.position ? "selected" : ""}`}
              title={`pos ${t.position} id ${t.id}`}
              onClick={() => selectToken(selectedToken === t.position ? null : t.position)}
            >
              {t.text === "\n" ? "\\n" : t.text}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
