import { useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { send } from "../ws/client"

export default function PromptBar() {
  const [prompt, setPrompt] = useState("")
  const running = useBrain((s) => s.running)
  const connected = useBrain((s) => s.connected)
  const modelStatus = useBrain((s) => s.modelStatus)
  const [maxNew, setMaxNew] = useState(64)
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [topK, setTopK] = useState(40)
  const [paused, setPaused] = useState(false)

  const canRun = connected && !running && modelStatus === "loaded"

  const run = () => {
    const text = prompt.trim()
    if (!text || !canRun) return
    send("run", {
      prompt: text,
      params: {
        max_new_tokens: maxNew,
        temperature,
        top_p: topP,
        top_k: topK,
        use_chat_template: true,
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
          <button className="btn" onClick={() => { send(paused ? "resume_inference" : "pause_inference"); setPaused(!paused) }}>
            {paused ? "RESUME" : "PAUSE"}
          </button>
          <button className="btn danger" onClick={() => { send("cancel"); setPaused(false) }}>
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
