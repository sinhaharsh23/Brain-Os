import { useEffect, useRef } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Stars } from "@react-three/drei"
import * as THREE from "three"
import { useBrain } from "../store/useBrainStore"
import { Text } from "@react-three/drei"
import ExternalObservationNotice from "../components/ExternalObservationNotice"

function PcaPoints() {
  const pca = useBrain((s) => s.pca)
  const selectedToken = useBrain((s) => s.selectedToken)
  const selectToken = useBrain((s) => s.selectToken)
  const tokens = pca?.tokens ?? []

  const positions = new Float32Array(tokens.length * 3)
  const colors = new Float32Array(tokens.length * 3)
  let maxNorm = 1
  for (const t of tokens) maxNorm = Math.max(maxNorm, t.norm)
  const geometry = new THREE.BufferGeometry()
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]
    positions[i * 3] = t.pca3[0]
    positions[i * 3 + 1] = t.pca3[1]
    positions[i * 3 + 2] = t.pca3[2]
    const n = t.norm / maxNorm
    const c = new THREE.Color().setHSL(0.55 - n * 0.35, 0.9, 0.55)
    colors[i * 3] = c.r
    colors[i * 3 + 1] = c.g
    colors[i * 3 + 2] = c.b
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3))

  return (
    <group>
      <points geometry={geometry} onClick={(e) => selectToken(e.instanceId ?? null)}>
        <pointsMaterial vertexColors size={0.14} sizeAttenuation />
      </points>
      {tokens.map((t, i) => (
        <Text
          key={i}
          position={[t.pca3[0], t.pca3[1] - 0.22, t.pca3[2]]}
          fontSize={0.16}
          color={selectedToken === i ? "#4cc2ff" : "#7c89a8"}
          anchorX="center"
          onClick={(e) => {
            e.stopPropagation()
            selectToken(i)
          }}
        >
          {t.text.length > 10 ? t.text.slice(0, 9) + "…" : t.text}
        </Text>
      ))}
    </group>
  )
}

function Scatter2D() {
  const pca = useBrain((s) => s.pca)
  const selectedToken = useBrain((s) => s.selectedToken)
  const setSelected = useBrain((s) => s.selectToken)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tokens = pca?.tokens ?? []

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const w = 520
    const h = 420
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`
    ctx.scale(dpr, dpr)
    ctx.fillStyle = "#0f1524"
    ctx.fillRect(0, 0, w, h)

    if (!pca || tokens.length < 2) {
      ctx.fillStyle = "#7c89a8"
      ctx.font = "11px monospace"
      ctx.fillText("no embedding data — run an inference", 20, 30)
      return
    }
    const xs = tokens.map((t) => t.pca3[0])
    const ys = tokens.map((t) => t.pca3[1])
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const pad = 60
    const sx = (v: number) => pad + ((v - minX) / Math.max(maxX - minX, 1e-6)) * (w - pad * 2)
    const sy = (v: number) => h - pad - ((v - minY) / Math.max(maxY - minY, 1e-6)) * (h - pad * 2)

    for (const t of tokens) {
      ctx.beginPath()
      ctx.arc(sx(t.pca3[0]), sy(t.pca3[1]), selectedToken === t.position ? 6 : 3.5, 0, Math.PI * 2)
      ctx.fillStyle = selectedToken === t.position ? "#4cc2ff" : "#9d6bff"
      ctx.fill()
    }
    ctx.fillStyle = "#7c89a8"
    ctx.font = "9px monospace"
    const step = Math.max(1, Math.floor(tokens.length / 26))
    tokens.forEach((t, i) => {
      if (i % step !== 0 && selectedToken !== t.position) return
      ctx.fillStyle = selectedToken === t.position ? "#4cc2ff" : "#8fa3c8"
      ctx.fillText(t.text.length > 8 ? t.text.slice(0, 7) + "…" : t.text, sx(t.pca3[0]) + 5, sy(t.pca3[1]) - 4)
    })
    ctx.strokeStyle = "#1e2a4a"
    ctx.strokeRect(0, 0, w, h)
  }, [pca, tokens, selectedToken])

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const px = (e.clientX - rect.left) * (canvas.width / rect.width)
    const py = (e.clientY - rect.top) * (canvas.height / rect.height)
    if (!pca || tokens.length < 2) return
    const xs = tokens.map((t) => t.pca3[0])
    const ys = tokens.map((t) => t.pca3[1])
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const pad = 60
    const sx = (v: number) => pad + ((v - minX) / Math.max(maxX - minX, 1e-6)) * (520 - pad * 2)
    const sy = (v: number) => 420 - pad - ((v - minY) / Math.max(maxY - minY, 1e-6)) * (420 - pad * 2)
    let best = -1
    let bestD = Infinity
    tokens.forEach((t, i) => {
      const d = (sx(t.pca3[0]) - px) ** 2 + (sy(t.pca3[1]) - py) ** 2
      if (d < bestD) {
        bestD = d
        best = i
      }
    })
    if (best >= 0 && bestD < 400) setSelected(best)
  }

  return <canvas ref={canvasRef} className="heatmap" onClick={onClick} style={{ cursor: "crosshair" }} />
}

export default function EmbeddingView() {
  const pca = useBrain((s) => s.pca)
  const inspectionMode = useBrain((s) => s.inspectionMode)
  if (inspectionMode === "limited") return <ExternalObservationNotice />
  return (
    <div style={{ display: "flex", gap: 10, padding: 10, height: "100%", overflow: "auto" }}>
      <div style={{ flex: "1 1 0", minWidth: 0 }}>
        <div className="muted" style={{ fontFamily: "var(--mono)", fontSize: 11, marginBottom: 4 }}>
          3D — PCA of real token embeddings {pca ? `· explained variance ${pca.explained_variance.slice(0, 2).map((v) => (v * 100).toFixed(0) + "%").join(", ")}` : ""}
        </div>
        <Canvas camera={{ position: [4, 3, 4], fov: 55 }} dpr={[1, 1.5]}>
          <color attach="background" args={["#0f1524"]} />
          <ambientLight intensity={0.8} />
          <PcaPoints />
          <Stars radius={30} depth={15} count={400} factor={2} fade />
          <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
        </Canvas>
      </div>
      <div style={{ flex: "0 0 540px" }}>
        <div className="muted" style={{ fontFamily: "var(--mono)", fontSize: 11, marginBottom: 4 }}>2D — PCA dims 1×2 (click point to inspect token)</div>
        <Scatter2D />
      </div>
    </div>
  )
}
