import { useState, useRef, useEffect } from "react"
import { useBrain } from "../../store/useBrainStore"

export default function RunLogCard() {
  const devLog = useBrain((s) => s.devLog)
  const running = useBrain((s) => s.running)
  const [filter, setFilter] = useState<"all" | "info" | "warn" | "error">("all")
  const scrollRef = useRef<HTMLDivElement | null>(null)

  const filteredLogs = devLog.filter((l) => (filter === "all" ? true : l.level === filter))

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [devLog])

  return (
    <div className="telemetry-card run-log-card">
      <div className="card-header">
        <span className="card-title">RUN LOG</span>
        <div className="log-filters">
          <button className={filter === "all" ? "active" : ""} onClick={() => setFilter("all")}>ALL</button>
          <button className={filter === "info" ? "active" : ""} onClick={() => setFilter("info")}>INFO</button>
          <button className={filter === "warn" ? "active" : ""} onClick={() => setFilter("warn")}>WARN</button>
          <button className={filter === "error" ? "active" : ""} onClick={() => setFilter("error")}>ERR</button>
        </div>
      </div>

      <div className="log-entries-scroll" ref={scrollRef}>
        {filteredLogs.length === 0 ? (
          <div className="empty-log-msg mono text-dim text-xs">standby - awaiting inference dispatch</div>
        ) : (
          filteredLogs.map((log, idx) => {
            const timeStr = new Date(log.ts).toLocaleTimeString("en-GB", { hour12: false })
            return (
              <div key={idx} className={`log-row mono ${log.level}`}>
                <span className="log-time">{timeStr}</span>
                <span className="log-tag">[{log.level.toUpperCase()}]</span>
                <span className="log-msg">{log.message}</span>
              </div>
            )
          })
        )}
        {running && (
          <div className="log-row mono info pulse-row">
            <span className="log-time">--:--:--</span>
            <span className="log-tag">[STREAM]</span>
            <span className="log-msg">generating next token...</span>
          </div>
        )}
      </div>
    </div>
  )
}
