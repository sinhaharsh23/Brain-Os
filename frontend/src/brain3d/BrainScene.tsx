import { Suspense, useState } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Grid, Stars } from "@react-three/drei"
import { useBrain } from "../store/useBrainStore"
import TransformerStack from "./TransformerStack"
import TokenRing from "./TokenRing"
import AttentionLinks from "./AttentionLinks"
import FlowParticle from "./FlowParticle"
import EmbeddingCloud from "./EmbeddingCloud"
import { bottomOfStack, inputY, layerY, outputY } from "./layout"

function OutputBeam({ numLayers }: { numLayers: number }) {
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const running = useBrain((s) => s.running)
  const count = generatedTokens.length
  const intensity = Math.min(1, count / 20) * (running ? 0.9 : 0.35)
  const y = outputY(numLayers)
  return (
    <group position={[0, y, 0]}>
      <mesh>
        <cylinderGeometry args={[0.45, 0.45, 0.7, 12]} />
        <meshStandardMaterial color="#3d2a05" emissive="#ffb648" emissiveIntensity={0.2 + intensity} transparent opacity={0.85} />
      </mesh>
      <mesh position={[0, -1.3, 0]}>
        <sphereGeometry args={[0.34 + intensity * 0.25, 12, 12]} />
        <meshBasicMaterial color="#ffb648" transparent opacity={0.25 + intensity * 0.5} toneMapped={false} />
      </mesh>
    </group>
  )
}

function LayerLabels({ numLayers }: { numLayers: number }) {
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectLayer = useBrain((s) => s.selectLayer)
  if (numLayers > 16) return null
  return (
    <group>
      {Array.from({ length: numLayers }, (_, i) => (
        <sprite
          key={i}
          position={[3.0, layerY(i, numLayers), 0]}
          onClick={(e) => {
            e.stopPropagation()
            selectLayer(selectedLayer === i ? null : i)
          }}
        >
          <spriteMaterial
            color={selectedLayer === i ? "#9d6bff" : "#5a6c96"}
            transparent
            opacity={selectedLayer === i ? 1 : 0.7}
            depthWrite={false}
          />
        </sprite>
      ))}
    </group>
  )
}

export default function BrainScene() {
  const model = useBrain((s) => s.model)
  const [showAttention, setShowAttention] = useState(true)
  const [showCloud, setShowCloud] = useState(false)
  const [showTokens, setShowTokens] = useState(true)
  const numLayers = model?.num_layers ?? 24
  const running = useBrain((s) => s.running)
  const response = useBrain((s) => s.response)

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas camera={{ position: [8.5, 1.5, 9.5], fov: 50 }} dpr={[1, 1.5]}>
        <color attach="background" args={["#080c16"]} />
        <fog attach="fog" args={["#080c16", 24, 40]} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[6, 10, 6]} intensity={1.2} />
        <pointLight position={[0, inputY(numLayers), 0]} intensity={14} color="#4cc2ff" distance={12} />
        <pointLight position={[0, outputY(numLayers), 0]} intensity={16} color="#ffb648" distance={12} />
        <Suspense fallback={null}>
          <TransformerStack numLayers={numLayers} />
          {showTokens && <TokenRing numLayers={numLayers} />}
          <AttentionLinks numLayers={numLayers} visible={showAttention} />
          <FlowParticle numLayers={numLayers} />
          <OutputBeam numLayers={numLayers} />
          <EmbeddingCloud visible={showCloud} />
          <LayerLabels numLayers={numLayers} />
          <Stars radius={40} depth={20} count={900} factor={2.4} fade speed={0.4} />
          <Grid
            position={[0, bottomOfStack(numLayers) - 2.2, 0]}
            args={[30, 30]}
            cellSize={0.6}
            cellColor="#14203c"
            sectionSize={3}
            sectionColor="#1d2f5a"
            fadeDistance={28}
            fadeStrength={2}
          />
        </Suspense>
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={3} maxDistance={26} />
      </Canvas>

      <div style={{ position: "absolute", top: 10, left: 12, display: "flex", gap: 8, zIndex: 5 }}>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11, fontFamily: "var(--mono)" }}>
          <input type="checkbox" checked={showAttention} onChange={(e) => setShowAttention(e.target.checked)} />
          attention links
        </label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11, fontFamily: "var(--mono)" }}>
          <input type="checkbox" checked={showTokens} onChange={(e) => setShowTokens(e.target.checked)} />
          tokens
        </label>
        <label className="muted" style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11, fontFamily: "var(--mono)" }}>
          <input type="checkbox" checked={showCloud} onChange={(e) => setShowCloud(e.target.checked)} />
          embedding PCA cloud
        </label>
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 10,
          left: 12,
          right: 12,
          zIndex: 5,
          pointerEvents: "none",
          fontFamily: "var(--mono)",
          fontSize: 11,
          color: "var(--muted)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
          gap: 12,
        }}
      >
        <span>
          {running ? "processing step…" : "idle"} · drag to orbit · scroll to zoom · click a token node to inspect
        </span>
        <span style={{ maxWidth: "55%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {response ? `▶ ${response.slice(0, 120)}${response.length > 120 ? "…" : ""}` : "output appears here"}
        </span>
      </div>
    </div>
  )
}
