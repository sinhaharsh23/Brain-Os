import { useEffect, useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"

export default function DevView() {
  const devLog = useBrain((s) => s.devLog)
  const timeline = useBrain((s) => s.timeline)
  const sessions = useBrain((s) => s.sessions)
  const model = useBrain((s) => s.model)
  const [events, setEvents] = useState<Record<string, unknown>[]>([])
  const [selectedSession, setSelectedSession] = useState("")
  const [tensorResults, setTensorResults] = useState<string[]>([])

  const loadEvents = async (sid: string) => {
    setSelectedSession(sid)
    try {
      setEvents(await api.sessionEvents(sid))
    } catch (e) {
      setEvents([])
      setTensorResults((r) => [...r, `load events: ${e}`])
    }
  }

  const probeTensors = async () => {
    const sid = useBrain.getState().sessionId
    if (!sid) return
    const out: string[] = []
    const checks: [string, () => Promise<unknown>][] = [
      ["embedding/0", () => api.embedding(sid, 0)],
      ["qkv l0 q pos0", () => api.qkv(sid, 0, "q", 0)],
      ["mlp l0 pos0", () => api.mlp(sid, 0, 0, 8)],
      ["attention l0h0 pos0", () => api.attention(sid, 0, 0, 0)],
      ["logits step0", () => api.logits(sid, 0, 8)],
    ]
    for (const [name, fn] of checks) {
      try {
        const r = (await fn()) as { stats?: { n: number; l2_norm: number }; shape?: number[] }
        out.push(`✓ ${name}: n=${r.stats?.n} l2=${r.stats?.l2_norm.toFixed(2)} ${r.shape ? "shape=" + JSON.stringify(r.shape) : ""}`)
      } catch (e) {
        out.push(`✗ ${name}: ${e}`)
      }
    }
    setTensorResults(out)
  }

  useEffect(() => {
    api.sessions().then((s) => useBrain.getState().set({ sessions: s })).catch(() => {})
  }, [])

  return (
    <div style={{ padding: 12, overflowY: "auto", height: "100%", fontFamily: "var(--mono)", fontSize: 11 }}>
      <div className="panel-title" style={{ marginTop: 0 }}>
        Developer mode — raw events & tensor probes
      </div>

      <div className="flex" style={{ marginBottom: 8, flexWrap: "wrap" }}>
        <select value={selectedSession} onChange={(e) => loadEvents(e.target.value)} style={{ maxWidth: 340 }}>
          <option value="">select recorded session…</option>
          {sessions.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.session_id} · {s.prompt.slice(0, 40)}
            </option>
          ))}
        </select>
        <button className="btn small" onClick={probeTensors}>
          probe current session tensors
        </button>
        <span className="muted">{model ? `hooks: eager attention + q/k/v + mlp registered on ${model.architecture}` : ""}</span>
      </div>

      {tensorResults.map((r, i) => (
        <div key={i} className={r.startsWith("✓") ? "info" : "error"}>
          {r}
        </div>
      ))}

      {events.length > 0 && (
        <>
          <div className="panel-title">Session event stream ({events.length} events)</div>
          <div style={{ maxHeight: 260, overflowY: "auto" }}>
            {events.map((e, i) => (
              <div key={i} className="info">
                <span className="t">{(Number(e.ts) - Number(events[0].ts)).toFixed(2)}s</span>
                {String(e.type)} {JSON.stringify(e.data).slice(0, 180)}
              </div>
            ))}
          </div>
        </>
      )}

      <div className="panel-title">Timeline ({timeline.length})</div>
      <div style={{ maxHeight: 200, overflowY: "auto" }}>
        {timeline.map((e, i) => (
          <div key={i} className="info">
            <span className="t">{(e.ts / 1000).toFixed(1)}s</span>
            [{e.type}] {e.label}
          </div>
        ))}
      </div>

      <div className="panel-title">Dev log ({devLog.length})</div>
      <div style={{ maxHeight: 200, overflowY: "auto" }}>
        {devLog.map((e, i) => (
          <div key={i} className={e.level}>
            <span className="t">[{new Date(e.ts).toLocaleTimeString()}]</span>
            {e.message}
          </div>
        ))}
      </div>
    </div>
  )
}
