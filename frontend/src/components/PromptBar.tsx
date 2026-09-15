import { useEffect, useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { send } from "../ws/client"
import { api } from "../api/client"
import type { ProviderDescriptor } from "../types"

const FALLBACK_PROVIDERS: ProviderDescriptor[] = [
  { provider_id: "qwen-local", display_name: "Qwen local", kind: "local", inspection_mode: "deep", availability: "available", limitation: "", capabilities: {}, models: ["Qwen/Qwen2.5-0.5B-Instruct"] },
  { provider_id: "openai", display_name: "OpenAI / ChatGPT", kind: "external", inspection_mode: "limited", availability: "unknown", limitation: "", capabilities: {}, models: ["gpt-4o-mini"] },
  { provider_id: "anthropic", display_name: "Anthropic / Claude", kind: "external", inspection_mode: "limited", availability: "unknown", limitation: "", capabilities: {}, models: ["claude-3-5-haiku-latest"] },
  { provider_id: "google", display_name: "Google Gemini", kind: "external", inspection_mode: "limited", availability: "unknown", limitation: "", capabilities: {}, models: ["gemini-2.0-flash"] },
]

const optionLabels: Record<string, string> = {
  "qwen-local": "Qwen · DEEP",
  openai: "OpenAI · LIMITED",
  anthropic: "Claude · LIMITED",
  google: "Gemini · LIMITED",
}

export default function PromptBar() {
  const [prompt, setPrompt] = useState("")
  const running = useBrain((s) => s.running)
  const paused = useBrain((s) => s.paused)
  const connected = useBrain((s) => s.connected)
  const modelStatus = useBrain((s) => s.modelStatus)
  const runStatus = useBrain((s) => s.runStatus)
  const runId = useBrain((s) => s.runId)
  const queuePosition = useBrain((s) => s.queuePosition)
  const [maxNew, setMaxNew] = useState(64)
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [topK, setTopK] = useState(40)
  const [provider, setProvider] = useState("qwen-local")
  const [providerModel, setProviderModel] = useState("")
  const [providers, setProviders] = useState<ProviderDescriptor[]>(FALLBACK_PROVIDERS)

  useEffect(() => {
    api.providers().then(setProviders).catch(() => {})
  }, [])

  const providerInfo = providers.find((item) => item.provider_id === provider)

  const changeProvider = (value: string) => {
    setProvider(value)
    setProviderModel(providers.find((item) => item.provider_id === value)?.models[0] ?? "")
  }

  const canRun = connected && !running && (provider !== "qwen-local" || modelStatus === "loaded")

  const run = () => {
    const text = prompt.trim()
    if (!text || !canRun) return
    send("run", {
      provider,
      prompt: text,
      params: {
        max_new_tokens: maxNew,
        temperature,
        top_p: topP,
        top_k: topK,
        use_chat_template: true,
        ...(providerModel ? { model: providerModel } : {}),
      },
    })
  }

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      run()
    }
  }

  return (
    <div className="prompt-bar">
      <input
        type="text"
        placeholder="Ask the model a question — e.g. Explain how artificial intelligence works"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={onKey}
        disabled={running}
      />
      <div className="prompt-params">
        <select value={provider} onChange={(e) => changeProvider(e.target.value)} disabled={running} title="Provider capability mode">
          {providers.map((item) => <option key={item.provider_id} value={item.provider_id}>{optionLabels[item.provider_id] ?? `${item.display_name} · ${item.inspection_mode === "deep" ? "DEEP" : "LIMITED"}`}</option>)}
        </select>
        {provider !== "qwen-local" && providerInfo?.models.length ? (
          <select value={providerModel || providerInfo.models[0]} onChange={(e) => setProviderModel(e.target.value)} disabled={running} title="Provider model">
            {providerInfo.models.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        ) : null}
        {providerInfo && <span className={providerInfo.inspection_mode === "deep" ? "ok" : "warn"}>{providerInfo.inspection_mode === "deep" ? "Deep Inspection" : `External Observation · ${providerInfo.availability}`}</span>}
        <span>tokens</span>
        <input type="number" min={1} max={512} value={maxNew} onChange={(e) => setMaxNew(Number(e.target.value))} disabled={running} />
        <span>temp</span>
        <input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} disabled={running} />
        <span>top-p</span>
        <input type="number" min={0.05} max={1} step={0.05} value={topP} onChange={(e) => setTopP(Number(e.target.value))} disabled={running} />
        <span>top-k</span>
        <input type="number" min={0} max={200} value={topK} onChange={(e) => setTopK(Number(e.target.value))} disabled={running} />
      </div>
      {running ? (
        <div className="flex">
          {runStatus === "QUEUED" ? <span className="text-amber mono text-xs">QUEUED · POSITION {queuePosition ?? "—"}</span> : (
            <button className="btn" onClick={() => { send(paused ? "resume_inference" : "pause_inference") }}>
              {paused ? "RESUME" : "PAUSE"}
            </button>
          )}
          <button className="btn danger" onClick={() => { send("cancel", runId ? { run_id: runId } : {}) }}>
            CANCEL
          </button>
        </div>
      ) : (
        <button className="btn primary" onClick={run} disabled={!canRun}>
          RUN
        </button>
      )}
    </div>
  )
}
