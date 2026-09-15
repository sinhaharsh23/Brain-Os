import { useBrain } from "../store/useBrainStore"
import ThroughputGaugeCard from "../components/analytics/ThroughputGaugeCard"
import PipelineStatusCard from "../components/analytics/PipelineStatusCard"

export default function SystemMonitorView() {
  const hardware = useBrain((s) => s.hardware)
  const monitoring = useBrain((s) => s.monitoring)

  return (
    <div className="section-page-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div>
          <h2 className="section-title">SYSTEM MONITOR (PERFORMANCE & TELEMETRY)</h2>
          <span className="section-subtitle text-dim text-xs">
            Real-time Apple Silicon Metal Performance Shaders and memory utilization
          </span>
        </div>
        <span className={hardware?.backend === "MPS" ? "badge-emerald" : "badge-outline"}>
          {hardware ? `${hardware.backend} ${hardware.gpu_available ? "ONLINE" : "FALLBACK"}` : "HARDWARE PROBE PENDING"}
        </span>
      </div>

      <div className="system-monitor-grid">
        <ThroughputGaugeCard />
        <PipelineStatusCard />

        <div className="telemetry-card hw-raw-card">
          <div className="card-header">
            <span className="card-title">HARDWARE PROBE TELEMETRY</span>
          </div>
          <div className="hw-grid-details text-xs">
            <div className="hw-col">
              <div className="hw-row"><span className="text-dim">CPU CHIP:</span> <span className="text-bright">{hardware?.cpu?.model ?? "unavailable"}</span></div>
              <div className="hw-row"><span className="text-dim">CORES:</span> <span className="text-bright">{hardware ? `${hardware.cpu.count} Cores` : "unavailable"}</span></div>
              <div className="hw-row"><span className="text-dim">CPU LOAD:</span> <span className="text-cyan">{monitoring ? `${monitoring.cpu_percent.toFixed(1)}%` : "unavailable"}</span></div>
            </div>
            <div className="hw-col">
              <div className="hw-row"><span className="text-dim">RAM USED:</span> <span className="text-bright">{monitoring ? `${monitoring.ram_used_gb.toFixed(2)} GB` : "unavailable"}</span></div>
              <div className="hw-row"><span className="text-dim">RAM TOTAL:</span> <span className="text-bright">{monitoring ? `${monitoring.ram_total_gb.toFixed(0)} GB` : "unavailable"}</span></div>
              <div className="hw-row"><span className="text-dim">PROCESS RSS:</span> <span className="text-amber">{monitoring ? `${monitoring.process_ram_gb.toFixed(2)} GB` : "unavailable"}</span></div>
            </div>
            <div className="hw-col">
              <div className="hw-row"><span className="text-dim">PYTORCH:</span> <span className="text-bright">{hardware?.torch_version ?? "unavailable"}</span></div>
              <div className="hw-row"><span className="text-dim">ACCELERATOR:</span> <span className="text-emerald">{hardware?.backend ?? "unavailable"}</span></div>
              <div className="hw-row"><span className="text-dim">ARCH:</span> <span className="text-bright">{hardware ? (hardware.backend === "MPS" ? "arm64 (Apple Silicon)" : "reported by runtime") : "unavailable"}</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
