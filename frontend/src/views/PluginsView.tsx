import { useMockPlugins } from "../hooks/useMocks"

export default function PluginsView() {
  const { plugins, activeCount } = useMockPlugins()

  return (
    <div className="section-page-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div>
          <h2 className="section-title">PLUGINS & EXTENSIONS</h2>
          <span className="section-subtitle text-dim text-xs">
            Ecosystem extensions, execution sidecars, and tool call hooks
          </span>
        </div>
        <span className="badge-emerald">{activeCount} ACTIVE EXTENSIONS</span>
      </div>

      <div className="plugins-grid">
        {plugins.map((p) => (
          <div key={p.id} className="telemetry-card plugin-card">
            <div className="card-header">
              <span className="card-title">{p.name}</span>
              <span className="badge-emerald">{p.status}</span>
            </div>
            <p className="plugin-desc text-dim text-xs py-2">{p.description}</p>
            <div className="plugin-meta-row flex items-center justify-between text-xxs">
              <span className="text-dim">CATEGORY: <b className="text-bright">{p.category}</b></span>
              <span className="text-cyan">v{p.version}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
