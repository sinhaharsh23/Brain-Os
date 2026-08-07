import { useState } from "react"
import { useBrain } from "../store/useBrainStore"
import { send } from "../ws/client"
import { api } from "../api/client"

type Tab = "output" | "probability" | "system" | "timeline" | "replay" | "devlog"

function OutputTab() {
  const response = useBrain((s) => s.response)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const running = useBrain((s) => s.running)
  const summary = useBrain((s) => s.summary)
  return (
    <div>
      <div className="token-strip" style={{ marginBottom: 6 }}>
        {generatedTokens.map((t) => (
          <span key={t.step} className="token-chip generated" title={`step ${t.step} · p=${(t.probability * 100).toFixed(1)}%`}>
            {t.text === "\n" ? "\\n" : t.text}
          </span>
        ))}
        {running && <span className="token-chip" style={{ borderStyle: "dashed", color: "var(--warn)" }}>▮</span>}
      </div>
      <div style={{ whiteSpace: "pre-wrap", fontFamily: "var(--mono)", fontSize: 12.5, lineHeight: 1.6, maxHeight: 110, overflowY: "auto" }}>
        {response || <span className="muted">— waiting for output —</span>}
      </div>
      {summary && (
        <div className="muted" style={{ fontSize: 10.5, marginTop: 4 }}>
          {summary.num_output_tokens} tokens · {JSON.stringify(summary.timings.tokens_per_second)} tok/s · total {summary.timings.total_ms}ms
        </div>
      )}
    </div>
  )
}

function ProbabilityTab() {
  const candidates = useBrain((s) => s.candidates)
  const lastLogits = useBrain((s) => s.lastLogits)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const selected = generatedTokens[generatedTokens.length - 1]
  const maxP = candidates?.[0]?.probability ?? 1

  return (
    <div>
      {lastLogits && (
        <div className="muted" style={{ fontFamily: "var(--mono)", fontSize: 10.5, marginBottom: 4 }}>
          step {lastLogits.step} · temperature {lastLogits.temperature} · top-p {lastLogits.top_p} · top-k {lastLogits.top_k}
        </div>
      )}
      {candidates ? (
        candidates.map((c) => (
          <div key={c.token_id} className={`candidate-row ${selected && c.token_id === selected.token_id ? "selected" : ""}`}>
            <span className="rank">#{c.rank}</span>
            <span className="tok">{c.text}</span>
            <div className="bar-wrap">
              <div className="bar" style={{ width: `${(c.probability / maxP) * 100}%` }} />
            </div>
            <span className="pct">{(c.probability * 100).toFixed(1)}% · {c.logit.toFixed(1)}</span>
          </div>
        ))
      ) : (
        <div className="muted">waiting for logits…</div>
      )}
    </div>
  )
}

function SystemTab() {
  const mon = useBrain((s) => s.monitoring)
  if (!mon)
    return (
      <div className="muted">
        <div className="kv">
          <span className="k">CPU</span>
          <span className="v">—</span>
          <span className="k">RAM</span>
          <span className="v">—</span>
        </div>
      </div>
    )
  return (
    <div className="kv" style={{ gridTemplateColumns: "auto 1fr auto 1fr" }}>
      <span className="k">CPU</span>
      <span className="v">{mon.cpu_percent.toFixed(1)}%</span>
      <span className="k">RAM</span>
      <span className="v">{mon.ram_used_gb.toFixed(1)} / {mon.ram_total_gb.toFixed(0)} GB ({mon.ram_percent.toFixed(0)}%)</span>
      <span className="k">process</span>
      <span className="v">{mon.process_ram_gb.toFixed(2)} GB</span>
      <span className="k">GPU</span>
      <span className="v">{mon.gpu_percent !== null ? `${mon.gpu_percent.toFixed(0)}%` : "no GPU (CPU mode)"}</span>
      <span className="k">VRAM</span>
      <span className="v">{mon.vram_used_gb !== null ? `${mon.vram_used_gb.toFixed(2)} / ${mon.vram_total_gb?.toFixed(1)} GB` : "—"}</span>
      <span className="k">model</span>
      <span className="v">{mon.model_loaded ? "loaded" : "not loaded"}</span>
    </div>
  )
}

function TimelineTab() {
  const timeline = useBrain((s) => s.timeline)
  const running = useBrain((s) => s.running)
  return (
    <div className="dev-log">
      {timeline.length === 0 && <div className="muted">no events yet — run an inference</div>}
      {timeline.map((e, i) => (
        <div key={i} className="info">
          <span className="t">{(e.ts / 1000).toFixed(1)}s</span>
          <span style={{ color: e.type === "layer" ? "var(--accent2)" : e.type === "token" ? "var(--generated)" : "var(--text)" }}>
            [{e.type}]
          </span>{" "}
          {e.label}
          {e.detail && <span className="muted"> ({e.detail})</span>}
          {running && i === timeline.length - 1 && <span className="muted"> ▮</span>}
        </div>
      ))}
    </div>
  )
}

function ReplayTab() {
  const sessions = useBrain((s) => s.sessions)
  const replay = useBrain((s) => s.replay)
  const running = useBrain((s) => s.running)
  const [speed, setSpeed] = useState(2)

  const refresh = () => api.sessions().then((s) => useBrain.getState().set({ sessions: s })).catch(() => {})

  return (
    <div>
      <div className="flex" style={{ marginBottom: 6 }}>
        <button className="btn small" onClick={refresh}>
          refresh
        </button>
        <span className="muted" style={{ fontSize: 11 }}>
          {replay.sessionId ? `replaying ${replay.sessionId} · ${replay.status}` : "no session loaded"}
        </span>
        <span className="muted" style={{ fontSize: 11 }}>
          speed
        </span>
        <input type="number" min={0.25} max={32} step={0.25} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} style={{ width: 60 }} />
        <button className="btn small" disabled={!replay.sessionId} onClick={() => send("replay_step", { direction: -1 })}>◀ step</button>
        <button className="btn small" disabled={!replay.sessionId} onClick={() => send("replay_step", { direction: 1 })}>step ▶</button>
      </div>
      {replay.count > 0 && (
        <div className="flex" style={{ marginBottom: 6 }}>
          <span className="muted mono" style={{ fontSize: 10 }}>event {replay.index + 1} / {replay.count}</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, replay.count - 1)}
            value={Math.min(replay.index, Math.max(0, replay.count - 1))}
            onChange={(e) => send("replay_seek", { index: Number(e.target.value) })}
            style={{ flex: 1 }}
          />
        </div>
      )}
      {sessions.length === 0 && <div className="muted">no recorded sessions yet</div>}
      {sessions.map((s) => (
        <div key={s.session_id} className="flex spread" style={{ margin: "3px 0" }}>
          <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "60%" }}>
            <span className="mono" style={{ fontSize: 11 }}>
              {s.prompt.slice(0, 44)}
            </span>
            <span className="muted" style={{ fontSize: 10 }}>
              {" "}
              · {s.num_output_tokens} tok · {JSON.stringify(s.timings.tokens_per_second) ?? "?"} t/s
            </span>
          </div>
          <div className="flex">
            <button className="btn small" disabled={running} onClick={() => send("replay", { session_id: s.session_id, speed })}>
              ▶
            </button>
            <button
              className="btn small"
              disabled={!replay.sessionId}
              onClick={() => send("replay_pause", { paused: !replay.paused })}
            >
              {replay.paused ? "⏵" : "⏸"}
            </button>
            <button className="btn small" disabled={!replay.sessionId} onClick={() => send("replay_stop")}>
              ⏹
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

function DevLogTab() {
  const devLog = useBrain((s) => s.devLog)
  return (
    <div className="dev-log">
      {devLog.map((e, i) => (
        <div key={i} className={e.level}>
          <span className="t">[{new Date(e.ts).toLocaleTimeString()}]</span>
          {e.message}
        </div>
      ))}
    </div>
  )
}

export default function BottomPanel() {
  const [tab, setTab] = useState<Tab>("output")
  const running = useBrain((s) => s.running)
  return (
    <div className="bottom-panel">
      <div className="bottom-tabs">
        {(["output", "probability", "system", "timeline", "replay", "devlog"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t}
            {t === "timeline" && running ? " ▮" : ""}
          </button>
        ))}
      </div>
      <div className="bottom-content">
        {tab === "output" && <OutputTab />}
        {tab === "probability" && <ProbabilityTab />}
        {tab === "system" && <SystemTab />}
        {tab === "timeline" && <TimelineTab />}
        {tab === "replay" && <ReplayTab />}
        {tab === "devlog" && <DevLogTab />}
      </div>
    </div>
  )
}
