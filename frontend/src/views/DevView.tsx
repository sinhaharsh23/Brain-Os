import TelemetryDetails from "../components/TelemetryDetails"
import { useEffect, useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"
import ExternalObservationNotice from "../components/ExternalObservationNotice"

export default function DevView() {
  const devLog = useBrain((s) => s.devLog)
  const timeline = useBrain((s) => s.timeline)
  const sessions = useBrain((s) => s.sessions)
  const model = useBrain((s) => s.model)
  const inspectionMode = useBrain((s) => s.inspectionMode)
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
    const session = useBrain.getState()
    const sid = session.sessionId
    if (!sid) return
    const latestGenerated = session.generatedTokens[session.generatedTokens.length - 1]
    // A generated token is predicted from the preceding input position. The
    // active forward capture therefore belongs to position - 1; the latest
    // output token itself has not necessarily been fed back through a pass.
    const position = latestGenerated ? Math.max(0, latestGenerated.position - 1) : session.tokens[session.tokens.length - 1]?.position ?? 0
    const step = latestGenerated?.step ?? 0
    const out: string[] = []
    const checks: [string, () => Promise<unknown>][] = [
      [`embedding/${position}`, () => api.embedding(sid, position)],
      [`qkv l0 q pos${position}`, () => api.qkv(sid, 0, "q", position)],
      [`mlp l0 pos${position}`, () => api.mlp(sid, 0, position, 8)],
      [`attention l0h0 pos${position}`, () => api.attention(sid, 0, 0, position)],
      [`logits step${step}`, () => api.logits(sid, step, 8)],
    ]
    for (const [name, fn] of checks) {
      try {
        const r = (await fn()) as { stats?: { n: number; l2_norm: number } | null; shape?: number[]; candidates?: { text: string; probability: number }[]; capture_kind?: string }
        const detail = r.stats ? `n=${r.stats.n} l2=${r.stats.l2_norm.toFixed(2)}` : r.candidates ? `top candidates=${r.candidates.length}` : "captured"
        out.push(`✓ ${name}: ${detail} ${r.shape ? "shape=" + JSON.stringify(r.shape) : r.capture_kind ? `capture=${r.capture_kind}` : ""}`)
      } catch (e) {
        out.push(`✗ ${name}: ${e}`)
      }
    }
    setTensorResults(out)
  }

  useEffect(() => {
    api.sessions().then((s) => useBrain.getState().set({ sessions: s })).catch(() => {})
  }, [])

  if (inspectionMode === "limited") return <ExternalObservationNotice />

  return (
    <div style={{ padding: 12, overflowY: "auto", height: "100%", fontFamily: "var(--mono)", fontSize: 11 }}>
      <TelemetryDetails />
      <p className="muted">Prefill processes the chat-formatted input to initialize model state. Each autoregressive decode step traverses the transformer stack again, until a stop token or the output limit.</p>
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
            <span className="t">{Math.max(0, e.ts - (timeline[0]?.ts ?? e.ts)).toFixed(2)}s</span>
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
