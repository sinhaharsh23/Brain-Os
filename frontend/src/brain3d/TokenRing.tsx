import { useMemo, useRef, useState } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import { Text } from "@react-three/drei"
import { useBrain } from "../store/useBrainStore"
import { RING_RADIUS, inputY, tokenAngle } from "./layout"

interface TokenRingProps {
  numLayers: number
}

interface TokenItem {
  position: number
  text: string
  generated: boolean
  special: boolean
}

function useTokenItems(): { items: TokenItem[]; numLayers: number; selectedToken: number | null } {
  const tokens = useBrain((s) => s.tokens)
  const generatedTokens = useBrain((s) => s.generatedTokens)
  const model = useBrain((s) => s.model)
  const selectedToken = useBrain((s) => s.selectedToken)
  const items = useMemo<TokenItem[]>(() => {
    const prompt: TokenItem[] = tokens.map((t) => ({ position: t.position, text: t.text, generated: false, special: t.is_special }))
    const gen: TokenItem[] = generatedTokens.map((t) => ({ position: t.position, text: t.text, generated: true, special: false }))
    return [...prompt, ...gen]
  }, [tokens, generatedTokens])
  return { items, numLayers: model?.num_layers ?? 24, selectedToken }
}

export default function TokenRing({ numLayers }: TokenRingProps) {
  const { items, selectedToken } = useTokenItems()
  const ref = useRef<THREE.InstancedMesh>(null)
  const [hovered, setHovered] = useState<number | null>(null)
  const selectToken = useBrain((s) => s.selectToken)
  const total = items.length

  const colors = useMemo(() => {
    const c = new Float32Array(Math.max(total, 1) * 3)
    for (let i = 0; i < total; i++) {
      const it = items[i]
      let col: [number, number, number]
      if (it.generated) col = [1.0, 0.72, 0.28]
      else if (it.special) col = [0.62, 0.42, 1.0]
      else col = [0.3, 0.76, 1.0]
      c[i * 3] = col[0]
      c[i * 3 + 1] = col[1]
      c[i * 3 + 2] = col[2]
    }
    return c
  }, [items])

  const dummy = useMemo(() => new THREE.Object3D(), [])
  const colorObj = useMemo(() => new THREE.Color(), [])

  useFrame(() => {
    if (!ref.current) return
    const mesh = ref.current
    for (let i = 0; i < Math.max(total, 1); i++) {
      const a = tokenAngle(i, Math.max(total, 1))
      const y = inputY(numLayers) + Math.sin(a * 2) * 0.12
      dummy.position.set(Math.cos(a) * RING_RADIUS, y, Math.sin(a) * RING_RADIUS)
      const isSel = selectedToken === i
      const isHov = hovered === i
      const scale = (isSel ? 1.6 : isHov ? 1.35 : 1.0) * (items[i]?.generated ? 0.85 : 0.7)
      dummy.scale.set(scale, scale, scale)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
      mesh.setColorAt(i, colorObj.setRGB(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]))
    }
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  })

  const showLabels = total <= 90
  const labelStep = showLabels ? 1 : Math.ceil(total / 90)
  const selected = items[selectedToken ?? -1]

  return (
    <group>
      <instancedMesh
        ref={ref}
        args={[undefined, undefined, Math.max(total, 1)]}
        onClick={(e) => {
          e.stopPropagation()
          selectToken(e.instanceId ?? null)
        }}
        onPointerMove={(e) => {
          if (e.instanceId !== undefined) setHovered(e.instanceId)
        }}
        onPointerOut={() => setHovered(null)}
      >
        <sphereGeometry args={[0.16, 12, 12]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      {total > 0 &&
        items.map((it, i) => {
          const show = (i % labelStep === 0 && !showLabels) || showLabels
          if (!show && selectedToken !== i) return null
          const a = tokenAngle(i, total)
          const y = inputY(numLayers) + Math.sin(a * 2) * 0.12
          const pos: [number, number, number] = [Math.cos(a) * RING_RADIUS, y, Math.sin(a) * RING_RADIUS]
          return (
            <Text
              key={it.position}
              position={[pos[0], pos[1] - 0.34, pos[2]]}
              fontSize={0.16}
              color={it.generated ? "#ffb648" : it.special ? "#9d6bff" : "#8fd8ff"}
              anchorX="center"
              anchorY="top"
              maxWidth={1.6}
            >
              {it.text.length > 14 ? it.text.slice(0, 12) + "…" : it.text === "\n" ? "\\n" : it.text}
            </Text>
          )
        })}
      {selected && (
        <Text position={[0, inputY(numLayers) + 0.75, 0]} fontSize={0.22} color="#4cc2ff" anchorX="center">
          ▶ {selected.text.length > 24 ? selected.text.slice(0, 22) + "…" : selected.text} (pos {selected.position})
        </Text>
      )}
    </group>
  )
}
