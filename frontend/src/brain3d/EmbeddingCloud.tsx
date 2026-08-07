import { useMemo, useRef } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import { useBrain } from "../store/useBrainStore"

interface EmbeddingCloudProps {
  visible: boolean
}

const SCALE = 1.35

export default function EmbeddingCloud({ visible }: EmbeddingCloudProps) {
  const pca = useBrain((s) => s.pca)
  const ref = useRef<THREE.Points>(null)

  const { positions, colors } = useMemo(() => {
    const tokens = pca?.tokens ?? []
    const positions = new Float32Array(tokens.length * 3)
    const colors = new Float32Array(tokens.length * 3)
    let maxNorm = 1
    for (const t of tokens) maxNorm = Math.max(maxNorm, t.norm)
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i]
      positions[i * 3] = t.pca3[0] * SCALE
      positions[i * 3 + 1] = t.pca3[1] * SCALE
      positions[i * 3 + 2] = t.pca3[2] * SCALE
      const n = t.norm / maxNorm
      const c = new THREE.Color().setHSL(0.55 - n * 0.35, 0.9, 0.5 + n * 0.25)
      colors[i * 3] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b
    }
    return { positions, colors }
  }, [pca])

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3))
    return geo
  }, [positions, colors])

  useFrame(() => {
    if (ref.current) {
      ref.current.rotation.y += 0.0008
    }
  })

  if (!visible || !pca) return null

  return (
    <group position={[0, 0.2, 0]}>
      <points ref={ref} geometry={geometry}>
        <pointsMaterial vertexColors size={0.11} sizeAttenuation transparent opacity={0.9} depthWrite={false} />
      </points>
    </group>
  )
}
