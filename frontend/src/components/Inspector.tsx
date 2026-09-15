import { useEffect, useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"
import type { AttentionResponse, TensorVectorResponse } from "../types"
import ExternalObservationNotice from "./ExternalObservationNotice"

interface Loaded {
  embedding?: TensorVectorResponse
  qkv?: { q?: TensorVectorResponse; k?: TensorVectorResponse; v?: TensorVectorResponse }
  mlp?: TensorVectorResponse
  attention?: AttentionResponse
  hidden?: TensorVectorResponse
  error?: string
}

export default function Inspector() {
  const selectedToken = useBrain((s) => s.selectedToken)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectedHead = useBrain((s) => s.selectedHead)
  const sessionId = useBrain((s) => s.sessionId)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const layers = useBrain((s) => s.layers)
  const qkvStats = useBrain((s) => s.qkvStats)
  const mlpTop = useBrain((s) => s.mlpTop)
  const candidates = useBrain((s) => s.candidates)
  const selectedNeuron = useBrain((s) => s.selectedNeuron)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  const [loaded, setLoaded] = useState<Loaded | null>(null)
  const [neuronSearch, setNeuronSearch] = useState("")
  const [loading, setLoading] = useState(false)
  const [neuronValue, setNeuronValue] = useState<{ index: number; value: number } | null>(null)

  const tokenAt = (pos: number | null) => {
    if (pos === null) return null
    if (pos < tokens.length) return tokens[pos]
    const g = generatedTokens.find((t) => t.position === pos)
    return g ?? null
  }

  useEffect(() => {
    setLoaded(null)
    if (!sessionId || selectedToken === null) return
    setLoading(true)
    const layer = selectedLayer ?? 0
    let cancelled = false
    Promise.all([
      api.embedding(sessionId, selectedToken).catch(() => undefined),
      api.qkv(sessionId, layer, "q", selectedToken).catch(() => undefined),
      api.qkv(sessionId, layer, "k", selectedToken).catch(() => undefined),
      api.qkv(sessionId, layer, "v", selectedToken).catch(() => undefined),
      api.mlp(sessionId, layer, selectedToken, 12).catch(() => undefined),
      api.attention(sessionId, layer, selectedHead ?? 0, selectedToken).catch(() => undefined),
      api.hidden(sessionId, layer, selectedToken).catch(() => undefined),
    ]).then(([embedding, q, k, v, mlp, attention, hidden]) => {
      if (cancelled) return
      setLoaded({ embedding, qkv: { q, k, v }, mlp, attention, hidden })
      setLoading(false)
    })
    return () => {
      cancelled = true
    }
  }, [selectedToken, selectedLayer, selectedHead, sessionId])

  useEffect(() => {
    setNeuronValue(null)
    if (!sessionId || !selectedNeuron) return
    api.mlp(sessionId, selectedNeuron.layer, selectedToken ?? 0, 8, selectedNeuron.index)
      .then((result) => setNeuronValue(result.neuron ?? null))
      .catch(() => setNeuronValue(null))
  }, [sessionId, selectedNeuron, selectedToken])

  const layer = selectedLayer ?? 0
  const token = tokenAt(selectedToken)
  void token

  if (inspectionMode === "limited") return <ExternalObservationNotice />

  const showNeuron = () => {
    if (!sessionId) return
    const mlp = mlpTop[layer]
    if (!mlp) return
    const idx = Number(neuronSearch || selectedNeuron?.index)
    const found = mlp.top.find((t) => t.index === idx)
    const row = found
      ? { label: `neuron ${idx} @ token`, value: `${token?.text ?? "—"} → ${found.value.toFixed(4)} (rank ${found.rank})` }
      : { label: "search", value: "not in top-k — full tensor available via Dev mode" }
    useBrain.getState().setInspector({
      kind: "neuron",
      title: `Neuron ${neuronSearch || "?"} · Layer ${layer + 1}`,
      rows: [
        { label: "layer", value: `${layer + 1} / ${useBrain.getState().model?.num_layers ?? "?"}` },
        row,
        { label: "top-k values", value: mlp.top.map((t) => `${t.index}:${t.value.toFixed(3)}`).join("  ") },
        { label: "stats", value: `max ${mlp.stats.max.toFixed(3)} · mean ${mlp.stats.mean.toFixed(4)} · std ${mlp.stats.std.toFixed(4)}` },
      ],
    })
  }

  const statsRows = (s?: { n: number; min: number; max: number; mean: number; std: number; l2_norm: number }) =>
    s
      ? [
          { label: "dim", value: String(s.n) },
          { label: "min / max", value: `${s.min.toFixed(3)} / ${s.max.toFixed(3)}` },
          { label: "mean ± std", value: `${s.mean.toFixed(4)} ± ${s.std.toFixed(4)}` },
          { label: "L2 norm", value: s.l2_norm.toFixed(3) },
        ]
      : []

  return (
    <div>
      <div className="panel-title">Inspector</div>

      {token && selectedToken !== null && (
        <div className="card inspector-section">
          <h4>Token · pos {token.position}</h4>
          <div className="kv">
            <span className="k">text</span>
            <span className="v" style={{ color: token.text.includes("\n") ? "var(--warn)" : undefined }}>
              {JSON.stringify(token.text)}
            </span>
            <span className="k">id</span>
            <span className="v">{"token_id" in token ? token.token_id : token.id}</span>
            <span className="k">type</span>
            <span className="v">{"generated" in token ? "generated" : "prompt"}</span>
          </div>
          {loading && <div className="muted">loading tensors…</div>}
          {loaded?.embedding && (
            <div className="inspector-section">
              <h4>Embedding (PCA space)</h4>
              <div className="kv">
                <span className="k">norm</span>
                <span className="v">{loaded.embedding.stats.l2_norm.toFixed(2)}</span>
                <span className="k">pca3</span>
                <span className="v">
                  {loaded.embedding.pca3 ? `[${loaded.embedding.pca3.map((x) => x.toFixed(2)).join(", ")}]` : "—"}
                </span>
              </div>
            </div>
          )}
          {loaded?.hidden && (
            <div className="inspector-section">
              <h4>Hidden state · layer {layer + 1}</h4>
              {statsRows(loaded.hidden.stats).map((r) => (
                <div className="kv" key={r.label}>
                  <span className="k">{r.label}</span>
                  <span className="v">{r.value}</span>
                </div>
              ))}
            </div>
          )}
          {loaded?.qkv?.q && (
            <div className="inspector-section">
              <h4>Q / K / V · layer {layer + 1}</h4>
              {(["q", "k", "v"] as const).map((n) => {
                const t = loaded.qkv?.[n]
                return (
                  <div className="kv" key={n}>
                    <span className="k">{n.toUpperCase()}</span>
                    <span className="v">{t ? `L2 ${t.stats.l2_norm.toFixed(2)} · μ ${t.stats.mean.toFixed(3)}` : "unavailable"}</span>
                  </div>
                )
              })}
            </div>
          )}
          {loaded?.mlp && (
            <div className="inspector-section">
              <h4>MLP activations · layer {layer + 1}</h4>
              <div className="kv">
                <span className="k">top</span>
                <span className="v">{loaded.mlp.top?.slice(0, 6).map((t) => `#${t.index}=${t.value.toFixed(2)}`).join(" · ")}</span>
                <span className="k">max</span>
                <span className="v">{loaded.mlp.stats.max.toFixed(3)}</span>
                <span className="k">nonzero</span>
                <span className="v">{loaded.mlp.stats.n} dims</span>
              </div>
            </div>
          )}
          {loaded?.attention && (
            <div className="inspector-section">
              <h4>
                Attention · L{layer + 1} H{selectedHead ?? 0}
              </h4>
              <div className="kv">
                <span className="k">top targets</span>
                <span className="v">
                  {loaded.attention.weights
                    .slice()
                    .sort((a, b) => b.weight - a.weight)
                    .slice(0, 5)
                    .map((w) => `pos${w.token_index}:${(w.weight * 100).toFixed(1)}%`)
                    .join(" · ")}
                </span>
                <span className="k">entropy</span>
                <span className="v">{loaded.attention.stats.std.toFixed(4)}</span>
              </div>
            </div>
          )}
          {loaded?.error && <div className="err-box">{loaded.error}</div>}
        </div>
      )}

      {selectedToken === null && selectedLayer !== null && (
        <div className="card inspector-section">
          <h4>Layer {layer + 1}</h4>
          <div className="kv">
            <span className="k">hidden norm</span>
            <span className="v">{layers[layer]?.norm.toFixed(3) ?? "—"}</span>
            <span className="k">last step</span>
            <span className="v">{layers[layer]?.step ?? "—"}</span>
            <span className="k">q/k/v L2</span>
            <span className="v">
              {qkvStats
                .filter((s) => s.layer === layer)
                .map((s) => `${s.name.toUpperCase()}:${s.stats.l2_norm.toFixed(1)}`)
                .join(" ")}
            </span>
            <span className="k">MLP max</span>
            <span className="v">{mlpTop[layer] ? mlpTop[layer].stats.max.toFixed(3) : "—"}</span>
            {selectedNeuron && (
              <>
                <span className="k">selected unit</span>
                <span className="v">#{selectedNeuron.index} = {neuronValue ? neuronValue.value.toFixed(5) : "loading…"}</span>
              </>
            )}
          </div>
          <div className="inspector-section">
            <h4>Neuron search</h4>
            <div className="flex">
              <input
                type="text"
                placeholder="neuron index, e.g. 1432"
                value={neuronSearch}
                onChange={(e) => setNeuronSearch(e.target.value)}
                style={{ width: 120 }}
              />
              <button className="btn small" onClick={showNeuron}>
                inspect
              </button>
            </div>
          </div>
        </div>
      )}

      {selectedToken === null && selectedLayer === null && candidates && (
        <div className="card inspector-section">
          <h4>Next-token distribution</h4>
          {candidates.slice(0, 8).map((c) => (
            <div className="kv" key={c.token_id}>
              <span className="k">#{c.rank}</span>
              <span className="v">
                {c.text} · {(c.probability * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {selectedToken === null && selectedLayer === null && !candidates && (
        <div className="inspector-empty">
          Select a token, layer or head to inspect real model tensors.
          <br />
          <br />
          Run an inference to populate data.
        </div>
      )}
    </div>
  )
}
