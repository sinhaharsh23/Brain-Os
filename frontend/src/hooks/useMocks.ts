import { useState, useEffect } from "react"

// TODO: Backend integration - Replace with real RAG / Knowledge Base storage endpoints
export interface KnowledgeDocument {
  id: string
  title: string
  tokens: number
  relevance: number
  category: string
}

export function useMockKnowledgeBase() {
  const [docs] = useState<KnowledgeDocument[]>([
    { id: "kb-01", title: "BrainOS Architecture Whitepaper", tokens: 1420, relevance: 0.96, category: "Core Spec" },
    { id: "kb-02", title: "Apple Silicon Metal Performance Shaders", tokens: 890, relevance: 0.88, category: "Hardware" },
    { id: "kb-03", title: "Qwen2.5 GQA & RoPE Parameterization", tokens: 1150, relevance: 0.94, category: "Model" },
    { id: "kb-04", title: "KV Cache Dynamic Compaction Protocol", tokens: 670, relevance: 0.82, category: "Memory" },
  ])
  return { docs, totalDocs: docs.length, totalTokens: 4130 }
}

// TODO: Backend integration - Replace with local fine-tuning / training loss telemetry WebSocket stream
export interface TrainingEpochMetric {
  epoch: number
  loss: number
  valLoss: number
  learningRate: number
  perplexity: number
}

export function useMockTrainingMetrics() {
  const [metrics] = useState<TrainingEpochMetric[]>([
    { epoch: 1, loss: 2.84, valLoss: 2.91, learningRate: 2e-4, perplexity: 17.1 },
    { epoch: 2, loss: 2.15, valLoss: 2.24, learningRate: 1.8e-4, perplexity: 8.6 },
    { epoch: 3, loss: 1.72, valLoss: 1.83, learningRate: 1.5e-4, perplexity: 5.6 },
    { epoch: 4, loss: 1.38, valLoss: 1.52, learningRate: 1.1e-4, perplexity: 4.0 },
    { epoch: 5, loss: 1.14, valLoss: 1.31, learningRate: 7e-5, perplexity: 3.1 },
    { epoch: 6, loss: 0.98, valLoss: 1.18, learningRate: 3e-5, perplexity: 2.7 },
  ])
  return { metrics, isTraining: false, currentEpoch: 6, totalEpochs: 10 }
}

// TODO: Backend integration - Replace with dynamic plugin & sidecar execution hooks
export interface BrainPlugin {
  id: string
  name: string
  version: string
  author: string
  status: "ACTIVE" | "STANDBY" | "DISABLED"
  description: string
  category: string
}

export function useMockPlugins() {
  const [plugins] = useState<BrainPlugin[]>([
    { id: "p-calc", name: "Python REPL Sidecar", version: "1.2.0", author: "BrainOS Core", status: "ACTIVE", description: "Direct sandbox execution for Python code blocks", category: "Execution" },
    { id: "p-rag", name: "Vector Context Retriever", version: "2.1.0", author: "BrainOS Memory", status: "ACTIVE", description: "Cosine similarity top-k search over embedded knowledge chunks", category: "Retrieval" },
    { id: "p-web", name: "FastAPI Telemetry Streamer", version: "3.0.4", author: "System", status: "ACTIVE", description: "PyTorch forward hook event dispatcher over WebSocket", category: "Telemetry" },
    { id: "p-viz", name: "3D Synapse ThreeJS Renderer", version: "1.0.0", author: "Deepmind Team", status: "ACTIVE", description: "WebGL GPU-accelerated spatial token and layer visualizer", category: "Visualization" },
  ])
  return { plugins, activeCount: plugins.filter((p) => p.status === "ACTIVE").length }
}

// Battery / Power status hook (uses standard browser battery API with graceful fallback)
export function useMockBattery() {
  const [battery, setBattery] = useState<{ level: number; charging: boolean }>({ level: 100, charging: true })

  useEffect(() => {
    // Check if browser supports battery API
    const nav = navigator as unknown as { getBattery?: () => Promise<{ level: number; charging: boolean; addEventListener: (t: string, fn: () => void) => void }> }
    if (nav.getBattery) {
      nav.getBattery().then((b) => {
        setBattery({ level: Math.round(b.level * 100), charging: b.charging })
      }).catch(() => {})
    }
  }, [])

  return battery
}
