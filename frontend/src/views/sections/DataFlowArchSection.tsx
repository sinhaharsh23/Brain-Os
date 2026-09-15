import { useBrain } from "../../store/useBrainStore"
import { useMockKnowledgeBase } from "../../hooks/useMocks"

export default function DataFlowArchSection() {
  const running = useBrain((s) => s.running)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const { totalDocs } = useMockKnowledgeBase()

  const totalTokens = tokens.length + generatedTokens.length || 24

  return (
    <div className="data-flow-arch-container mono">
      <div className="section-header-wrap flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="section-title-tag">DATA FLOW ARCHITECTURE</span>
          <span className="badge-outline text-xxs">END-TO-END TENSOR PIPELINE</span>
        </div>
        <span className="text-xxs text-dim">BIDIRECTIONAL CONTEXT BUS</span>
      </div>

      <div className="arch-diagram-wrapper">
        {/* Top Floating Tier: Context & Memory Matrix */}
        <div className="arch-context-tier">
          <div className="arch-storage-node kb-node">
            <span className="storage-icon">📚</span>
            <div className="storage-text">
              <span className="storage-title">KNOWLEDGE BASE / RAG</span>
              <span className="storage-sub text-xxs text-dim">{totalDocs} Docs Indexed</span>
            </div>
          </div>

          <div className="arch-bidirectional-hub">
            <span className="hub-label">⇅ INGEST & RETRIEVE ⇅</span>
          </div>

          <div className="arch-storage-node memory-matrix-node">
            <span className="storage-icon">🗄️</span>
            <div className="storage-text">
              <span className="storage-title">MEMORY MATRIX (KV CACHE)</span>
              <span className="storage-sub text-xxs text-dim">{totalTokens} Tokens Cached</span>
            </div>
          </div>
        </div>

        {/* Center Main Horizontal Pipeline */}
        <div className="arch-main-pipeline">
          <div className="pipeline-node node-input">
            <span className="node-badge">01</span>
            <span className="node-title">User Input</span>
            <span className="node-detail text-xxs text-dim">Prompt string</span>
          </div>

          <div className="pipeline-connector">➔</div>

          <div className="pipeline-node node-token">
            <span className="node-badge">02</span>
            <span className="node-title">Token Engine</span>
            <span className="node-detail text-xxs text-dim">BPE Vocab</span>
          </div>

          <div className="pipeline-connector">➔</div>

          <div className="pipeline-node node-embed">
            <span className="node-badge">03</span>
            <span className="node-title">Embedding Engine</span>
            <span className="node-detail text-xxs text-dim">896d Tensor</span>
          </div>

          <div className="pipeline-connector">➔</div>

          <div className={`pipeline-node node-transformer ${running ? "computing-node" : ""}`}>
            <span className="node-badge">04</span>
            <span className="node-title">Neural Network</span>
            <span className="node-detail text-xxs text-dim">24 Layers</span>
          </div>

          <div className="pipeline-connector">➔</div>

          <div className="pipeline-node node-decode">
            <span className="node-badge">05</span>
            <span className="node-title">Decoding Engine</span>
            <span className="node-detail text-xxs text-dim">Softmax Sampler</span>
          </div>

          <div className="pipeline-connector">➔</div>

          <div className="pipeline-node node-output">
            <span className="node-badge">06</span>
            <span className="node-title">Response Output</span>
            <span className="node-detail text-xxs text-dim">Client Stream</span>
          </div>
        </div>
      </div>
    </div>
  )
}
