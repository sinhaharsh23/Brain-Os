import { Suspense, useEffect, useMemo, useRef, useState } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { Grid, Line, OrbitControls, Stars, Text } from "@react-three/drei"
import * as THREE from "three"
import { useBrain, type LayerState } from "../store/useBrainStore"
import type { AttentionLink, GenToken, PcaToken, TokenInfo } from "../types"

type Vec3 = [number, number, number]

const NODE_COUNT = 9
const FIELD_LEFT = -8.7
const FIELD_RIGHT = 8.7
const TOKEN_X = -10.7
const OUTPUT_X = 10.7

function layerX(layer: number, numLayers: number) {
  return FIELD_LEFT + (layer / Math.max(1, numLayers - 1)) * (FIELD_RIGHT - FIELD_LEFT)
}

function nodePosition(layer: number, node: number, numLayers: number): Vec3 {
  const angle = (node / NODE_COUNT) * Math.PI * 2
  const wobble = Math.sin((layer + 1) * 0.61 + node * 1.7) * 0.08
  return [layerX(layer, numLayers), Math.sin(angle) * (1.55 + wobble), Math.cos(angle) * (1.45 + wobble)]
}

function tokenPosition(index: number, count: number): Vec3 {
  const visible = Math.max(1, Math.min(count, 14))
  const angle = ((index % visible) / visible) * Math.PI * 2
  return [TOKEN_X, Math.sin(angle) * 2.2, Math.cos(angle) * 2.1]
}

function FlowPulse({ start, end, phase, running, color, size = 0.065 }: { start: Vec3; end: Vec3; phase: number; running: boolean; color: string; size?: number }) {
  const ref = useRef<THREE.Mesh>(null)
  const startVector = useMemo(() => new THREE.Vector3(...start), [start])
  const endVector = useMemo(() => new THREE.Vector3(...end), [end])

  useFrame(({ clock }) => {
    if (!ref.current) return
    const progress = (clock.getElapsedTime() * (running ? 0.72 : 0.12) + phase) % 1
    ref.current.position.lerpVectors(startVector, endVector, progress)
    ;(ref.current.material as THREE.MeshStandardMaterial).emissiveIntensity = running ? 3.2 : 0.85
  })

  return (
    <mesh ref={ref} position={start}>
      <sphereGeometry args={[size, 8, 8]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.8} toneMapped={false} transparent opacity={0.95} />
    </mesh>
  )
}

function NeuralLine({ start, end, color, opacity, active, phase, running, pulse = true }: { start: Vec3; end: Vec3; color: string; opacity: number; active: boolean; phase: number; running: boolean; pulse?: boolean }) {
  return (
    <group>
      <Line points={[start, end]} color={color} transparent opacity={opacity} lineWidth={active ? 1.7 : 0.65} />
      {pulse && <FlowPulse start={start} end={end} phase={phase} running={running} color={color} size={active ? 0.085 : 0.055} />}
    </group>
  )
}

function NeuralLayer({ layer, numLayers, state, selected, active, activeModule, showNeurons, onSelect }: { layer: number; numLayers: number; state?: LayerState; selected: boolean; active: boolean; activeModule: string | null; showNeurons: boolean; onSelect: (layer: number) => void }) {
  const group = useRef<THREE.Group>(null)
  // Keep each captured layer visible briefly even when a full forward pass
  // finishes between browser frames. This decay is driven by real events.
  useFrame(() => {
    const pulse = state ? Math.max(0, 1 - (Date.now() - state.at) / 350) : 0
    group.current?.traverse((object) => {
      if (object instanceof THREE.Mesh && object.material instanceof THREE.MeshStandardMaterial && object.geometry.type !== "BoxGeometry") {
        object.material.emissiveIntensity = selected || active ? 2.8 : 0.35 + pulse * 2.4
      }
    })
  })
  const hot = selected || active
  const layerColor = activeModule === "mlp" && active ? "#ffb648" : hot ? "#39d9ff" : "#2470ac"
  const opacity = selected ? 0.95 : active ? 0.88 : 0.48

  return (
    <group ref={group} position={[layerX(layer, numLayers), 0, 0]} onClick={(event) => { event.stopPropagation(); onSelect(layer) }}>
      <mesh rotation={[0, Math.PI / 2, 0]}>
        <torusGeometry args={[1.9, selected ? 0.055 : 0.026, 8, 36]} />
        <meshStandardMaterial color={layerColor} emissive={layerColor} emissiveIntensity={hot ? 2.2 : 0.35} transparent opacity={opacity} toneMapped={false} />
      </mesh>
      <mesh>
        <boxGeometry args={[0.08, 3.45, 3.2]} />
        <meshStandardMaterial color="#122b52" emissive={layerColor} emissiveIntensity={hot ? 0.8 : 0.12} transparent opacity={selected ? 0.22 : 0.08} depthWrite={false} />
      </mesh>
      {showNeurons && Array.from({ length: NODE_COUNT }, (_, node) => {
        const [, y, z] = nodePosition(layer, node, numLayers)
        const nodeHot = hot || state?.status === "processing-attention" || state?.status === "processing-mlp"
        return (
          <mesh key={node} position={[0, y, z]}>
            <sphereGeometry args={[nodeHot ? 0.12 : 0.075, 8, 8]} />
            <meshStandardMaterial color={layerColor} emissive={layerColor} emissiveIntensity={nodeHot ? 2.8 : 0.4} transparent opacity={nodeHot ? 0.98 : 0.7} toneMapped={false} />
          </mesh>
        )
      })}
      {hot && <pointLight color={layerColor} intensity={active ? 2.8 : 1.2} distance={4.2} decay={2} />}
      <Text position={[0, -2.7, 0]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.19} color={selected ? "#c9f6ff" : "#6388ab"} anchorX="center" anchorY="middle">
        {`L${String(layer).padStart(2, "0")}`}
      </Text>
    </group>
  )
}

function InputTokens({ tokens, generatedTokens, visible, selectedToken, onSelect }: { tokens: TokenInfo[]; generatedTokens: GenToken[]; visible: boolean; selectedToken: number | null; onSelect: (position: number) => void }) {
  if (!visible) return null
  const allTokens = [...tokens, ...generatedTokens.map((token) => ({ text: token.text, position: token.position } as TokenInfo))]
  const count = allTokens.length
  return (
    <group>
      {allTokens.slice(0, 14).map((token, index) => {
        const position = tokenPosition(index, count)
        const selected = selectedToken === token.position
        return (
          <mesh key={`${token.position}-${index}`} position={position} onClick={(event) => { event.stopPropagation(); onSelect(token.position) }}>
            <sphereGeometry args={[selected ? 0.18 : 0.11, 8, 8]} />
            <meshStandardMaterial color={selected ? "#fff0a8" : "#40d9ff"} emissive={selected ? "#ffb648" : "#00baff"} emissiveIntensity={selected ? 3.5 : 1.4} toneMapped={false} />
          </mesh>
        )
      })}
      <Text position={[TOKEN_X, -3.1, 0]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.23} color="#57d7ff" anchorX="center" anchorY="middle">INPUT TOKENS</Text>
    </group>
  )
}

function AttentionPaths({ layer, links, tokenCount, numLayers, visible, running, selectedOnly, selectedToken }: { layer: number; links: AttentionLink[]; tokenCount: number; numLayers: number; visible: boolean; running: boolean; selectedOnly: boolean; selectedToken: number | null }) {
  if (!visible || links.length === 0 || tokenCount === 0) return null
  const source = nodePosition(layer, 0, numLayers)
  const visibleLinks = selectedOnly && selectedToken !== null ? links.filter((link) => link.token_index === selectedToken) : links
  return (
    <group>
      {visibleLinks.slice(0, 18).map((link, index) => {
        const target = tokenPosition(Math.min(link.token_index, 13), tokenCount)
        const mid: Vec3 = [(source[0] + target[0]) / 2, Math.max(source[1], target[1]) + 0.9 + link.weight, (source[2] + target[2]) / 2]
        return <NeuralLine key={`${link.head}-${link.token_index}-${index}`} start={target} end={mid} color="#b57cff" opacity={Math.max(0.12, Math.min(0.78, link.weight * 1.8))} active={running} phase={index * 0.13} running={running} />
      })}
    </group>
  )
}

function EmbeddingMap({ pca, visible }: { pca: { tokens: PcaToken[] } | null; visible: boolean }) {
  if (!visible || !pca?.tokens?.length) return null
  return (
    <group position={[-2.4, 0, -1.8]}>
      {pca.tokens.slice(0, 80).map((token) => {
        const point: Vec3 = [Math.tanh(token.pca3[0] / 3) * 2.5, Math.tanh(token.pca3[1] / 3) * 2.2, Math.tanh(token.pca3[2] / 3) * 2.2]
        return <mesh key={token.position} position={point}><sphereGeometry args={[0.035 + Math.min(0.06, token.norm / 250), 6, 6]} /><meshBasicMaterial color="#9d6bff" transparent opacity={0.5} toneMapped={false} /></mesh>
      })}
    </group>
  )
}

function OutputCandidates({ candidates, running }: { candidates: { text: string; probability: number; token_id: number }[] | null; running: boolean }) {
  return (
    <group>
      <mesh position={[OUTPUT_X, 0, 0]}>
        <sphereGeometry args={[0.32 + (running ? 0.12 : 0), 12, 12]} />
        <meshStandardMaterial color="#ffb648" emissive="#ff7a35" emissiveIntensity={running ? 4 : 1.1} toneMapped={false} />
      </mesh>
      {(candidates?.slice(0, 5) ?? []).map((candidate, index) => (
        <mesh key={`${candidate.token_id}-${index}`} position={[OUTPUT_X + 0.2, (index - 2) * 0.42, 0]}>
          <sphereGeometry args={[0.05 + Math.min(0.13, candidate.probability * 0.2), 7, 7]} />
          <meshBasicMaterial color="#ffb648" transparent opacity={0.45 + candidate.probability * 0.5} toneMapped={false} />
        </mesh>
      ))}
      <Text position={[OUTPUT_X, -3.1, 0]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.23} color="#ffcf73" anchorX="center" anchorY="middle">OUTPUT</Text>
    </group>
  )
}

function NeuralField({ numLayers, showAttention, selectedAttentionOnly, showCloud, showTokens, showNeurons, running, response }: { numLayers: number; showAttention: boolean; selectedAttentionOnly: boolean; showCloud: boolean; showTokens: boolean; showNeurons: boolean; running: boolean; response: string }) {
  const model = useBrain((s) => s.model)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const layers = useBrain((s) => s.layers)
  const activeLayer = useBrain((s) => s.activeLayer)
  const activeModule = useBrain((s) => s.activeModule)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectedToken = useBrain((s) => s.selectedToken)
  const attentionLinks = useBrain((s) => s.attentionLinks)
  const pca = useBrain((s) => s.pca)
  const candidates = useBrain((s) => s.candidates)
  const selectLayer = useBrain((s) => s.selectLayer)
  const selectToken = useBrain((s) => s.selectToken)
  const tokenCount = tokens.length + generatedTokens.length
  const focusLayer = selectedLayer ?? activeLayer ?? Math.min(numLayers - 1, Math.floor(numLayers / 2))

  const connections = useMemo(() => {
    const result: { start: Vec3; end: Vec3; layer: number; node: number; phase: number; kind: "intra" | "inter" | "skip"; pulse: boolean }[] = []
    for (let layer = 0; layer < numLayers - 1; layer += 1) {
      for (let node = 0; node < NODE_COUNT; node += 1) {
        const start = nodePosition(layer, node, numLayers)
        for (const hop of [0, 1]) {
          result.push({ start, end: nodePosition(layer + 1, (node + hop) % NODE_COUNT, numLayers), layer, node, phase: (layer * 0.19 + node * 0.11 + hop * 0.07) % 1, kind: "inter", pulse: hop === 0 || node % 3 === 0 })
        }
      }
      for (let node = 0; node < NODE_COUNT; node += 1) {
        result.push({ start: nodePosition(layer, node, numLayers), end: nodePosition(layer, (node + 1) % NODE_COUNT, numLayers), layer, node, phase: (layer * 0.23 + node * 0.17) % 1, kind: "intra", pulse: false })
      }
    }
    for (let layer = 0; layer < numLayers - 2; layer += 2) {
      for (const node of [0, 3, 6]) {
        result.push({ start: nodePosition(layer, node, numLayers), end: nodePosition(layer + 2, (node + 2) % NODE_COUNT, numLayers), layer, node, phase: (layer * 0.31 + node * 0.09) % 1, kind: "skip", pulse: true })
      }
    }
    return result
  }, [numLayers])

  const inputStart: Vec3 = [TOKEN_X + 0.4, 0, 0]
  const inputEnd = nodePosition(0, 0, numLayers)
  const outputStart = nodePosition(numLayers - 1, 0, numLayers)
  const outputEnd: Vec3 = [OUTPUT_X - 0.4, 0, 0]

  return (
    <group>
      <Text position={[TOKEN_X, 2.9, 0]} fontSize={0.22} color="#6d9abb" anchorX="center" anchorY="middle">HUMAN INPUT</Text>
      <Text position={[0, 3.25, 0]} fontSize={0.28} color="#d7efff" anchorX="center" anchorY="middle">DENSE NEURAL NETWORK</Text>
      <Text position={[0, 2.86, 0]} fontSize={0.16} color="#6d9abb" anchorX="center" anchorY="middle">{`${model?.num_layers ?? numLayers} REAL LAYERS · ${model?.num_attention_heads ?? "—"} HEADS`}</Text>
      <NeuralLine start={inputStart} end={inputEnd} color="#40d9ff" opacity={0.65} active={running} phase={0.08} running={running} />
      <NeuralLine start={outputStart} end={outputEnd} color="#ffb648" opacity={0.72} active={running} phase={0.62} running={running} />
      {connections.map((connection) => {
        const hot = connection.layer === activeLayer || connection.layer + 1 === activeLayer || connection.layer === selectedLayer || connection.layer + 1 === selectedLayer
        const layerLinks = attentionLinks[connection.layer]?.length ?? 0
        const baseOpacity = connection.kind === "intra" ? 0.16 : connection.kind === "skip" ? 0.13 : 0.2
        const opacity = Math.min(0.72, baseOpacity + layerLinks * 0.008 + (hot ? 0.3 : 0))
        const color = activeLayer !== null && activeModule === "mlp" ? "#ffb648" : connection.kind === "skip" ? "#a875ff" : hot ? "#46dcff" : connection.kind === "intra" ? "#3477b4" : "#1b6594"
        return <NeuralLine key={`${connection.kind}-${connection.layer}-${connection.node}-${connection.phase}`} start={connection.start} end={connection.end} color={color} opacity={opacity} active={hot} phase={connection.phase} running={running} pulse={connection.pulse} />
      })}
      {Array.from({ length: numLayers }, (_, layer) => (
        <NeuralLayer key={layer} layer={layer} numLayers={numLayers} state={layers[layer]} selected={selectedLayer === layer} active={activeLayer === layer} activeModule={activeModule} showNeurons={showNeurons} onSelect={selectLayer} />
      ))}
      <InputTokens tokens={tokens} generatedTokens={generatedTokens} visible={showTokens} selectedToken={selectedToken} onSelect={selectToken} />
      <AttentionPaths layer={focusLayer} links={attentionLinks[focusLayer] ?? []} tokenCount={tokenCount} numLayers={numLayers} visible={showAttention} running={running} selectedOnly={selectedAttentionOnly} selectedToken={selectedToken} />
      <EmbeddingMap pca={pca} visible={showCloud} />
      <OutputCandidates candidates={candidates} running={running} />
      {selectedToken !== null && <pointLight position={[TOKEN_X, 0, 0]} color="#fff0a8" intensity={3.5} distance={5} decay={2} />}
      {response && <pointLight position={[OUTPUT_X, 0, 0]} color="#ff9c52" intensity={2.5} distance={5} decay={2} />}
    </group>
  )
}

export default function BrainScene() {
  const model = useBrain((s) => s.model)
  const [showAttention, setShowAttention] = useState(true)
  const [selectedAttentionOnly, setSelectedAttentionOnly] = useState(false)
  const [showCloud, setShowCloud] = useState(false)
  const [showTokens, setShowTokens] = useState(true)
  const [showNeurons, setShowNeurons] = useState(true)
  const numLayers = model?.num_layers ?? 0
  const running = useBrain((s) => s.running)
  const response = useBrain((s) => s.response)
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const attentionLinks = useBrain((s) => s.attentionLinks)
  const benchmarkMode = import.meta.env.DEV || new URLSearchParams(window.location.search).has("benchmark")
  const [fps, setFps] = useState<number | null>(null)
  const [minFps, setMinFps] = useState<number | null>(null)

  useEffect(() => {
    if (!benchmarkMode) return
    let animationId = 0
    let frames = 0
    let minimum = Number.POSITIVE_INFINITY
    let windowStart = performance.now()
    const sample = (now: number) => {
      frames += 1
      const elapsed = now - windowStart
      if (elapsed >= 1000) {
        const current = (frames * 1000) / elapsed
        minimum = Math.min(minimum, current)
        setFps(Math.round(current * 10) / 10)
        setMinFps(Math.round(minimum * 10) / 10)
        frames = 0
        windowStart = now
      }
      animationId = requestAnimationFrame(sample)
    }
    animationId = requestAnimationFrame(sample)
    return () => cancelAnimationFrame(animationId)
  }, [benchmarkMode])

  const renderedNodes = tokens.length + generatedTokens.length + numLayers * NODE_COUNT
  const renderedEdges = Object.values(attentionLinks).reduce((count, links) => count + links.length, 0) + Math.max(0, numLayers - 1) * NODE_COUNT * 3

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas camera={{ position: [0, 5.8, 18], fov: 48 }} dpr={[1, 1.5]}>
        <color attach="background" args={["#040914"]} />
        <fog attach="fog" args={["#040914", 18, 34]} />
        <ambientLight intensity={0.28} />
        <hemisphereLight args={["#79dfff", "#080b1c", 0.62]} />
        <pointLight position={[-9, 2, 4]} color="#29cfff" intensity={10} distance={13} decay={2} />
        <pointLight position={[0, -1, 6]} color="#6e61ff" intensity={8} distance={16} decay={2} />
        <pointLight position={[9, 2, 4]} color="#ff9a4c" intensity={10} distance={13} decay={2} />
        <Suspense fallback={null}>
          <NeuralField numLayers={numLayers} showAttention={showAttention} selectedAttentionOnly={selectedAttentionOnly} showCloud={showCloud} showTokens={showTokens} showNeurons={showNeurons} running={running} response={response} />
          <Stars radius={36} depth={18} count={500} factor={1.8} fade speed={0.15} />
          <Grid position={[0, -3.25, 0]} args={[30, 20]} cellSize={0.65} cellColor="#122b4c" sectionSize={3} sectionColor="#20426b" fadeDistance={27} fadeStrength={2} />
        </Suspense>
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={7} maxDistance={28} target={[0, 0, 0]} />
      </Canvas>

      <div style={{ position: "absolute", top: 10, left: 12, display: "flex", gap: 8, zIndex: 5, flexWrap: "wrap" }}>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11, fontFamily: "var(--mono)" }}><input type="checkbox" checked={showAttention} onChange={(e) => setShowAttention(e.target.checked)} /> neuro lines</label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11, fontFamily: "var(--mono)" }}><input type="checkbox" checked={selectedAttentionOnly} onChange={(e) => setSelectedAttentionOnly(e.target.checked)} /> selected path</label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11, fontFamily: "var(--mono)" }}><input type="checkbox" checked={showTokens} onChange={(e) => setShowTokens(e.target.checked)} /> token lights</label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11, fontFamily: "var(--mono)" }}><input type="checkbox" checked={showCloud} onChange={(e) => setShowCloud(e.target.checked)} /> embedding map</label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11, fontFamily: "var(--mono)" }}><input type="checkbox" checked={showNeurons} onChange={(e) => setShowNeurons(e.target.checked)} /> layer neurons</label>
      </div>

      <div style={{ position: "absolute", bottom: 10, left: 12, right: 12, zIndex: 5, pointerEvents: "none", fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 12 }}>
        <span>{running ? "LIVE SIGNAL FLOW · processing captured events" : "signal flow standby · drag to orbit · scroll to zoom · click a layer or token"}</span>
        <span>response text · bottom Output panel</span>
      </div>

      {benchmarkMode && <div className="neural-fps-probe mono" data-testid="neural-fps-probe"><span>FRAME PROBE · {running ? "ACTIVE" : "IDLE"}</span><strong>{fps === null ? "—" : `${fps} FPS`}</strong><span>MIN {minFps === null ? "—" : `${minFps} FPS`}</span><span>NODES {renderedNodes}</span><span>EDGES {renderedEdges}</span></div>}
    </div>
  )
}
