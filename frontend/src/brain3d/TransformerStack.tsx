import { useMemo, useRef } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import { useBrain } from "../store/useBrainStore"
import { LAYER_GAP, layerY } from "./layout"

interface SlabProps {
  numLayers: number
}

function useSlabMatrices(numLayers: number) {
  const layers = useBrain((s) => s.layers)
  const currentStep = useBrain((s) => s.currentStep)
  const mlpTop = useBrain((s) => s.mlpTop)

  const intensity = useMemo(() => new Float32Array(numLayers), [numLayers])

  const baseIntensity = useMemo(() => {
    const arr = new Float32Array(numLayers)
    for (let i = 0; i < numLayers; i++) {
      const t = i / Math.max(numLayers - 1, 1)
      arr[i] = 0.06 + 0.1 * Math.sin(t * Math.PI)
    }
    return arr
  }, [numLayers])

  return { layers, currentStep, mlpTop, intensity, baseIntensity }
}

export default function TransformerStack({ numLayers }: SlabProps) {
  const meshRefs = useRef<(THREE.Mesh | null)[]>([])
  const glowRefs = useRef<(THREE.Mesh | null)[]>([])
  const { layers, currentStep, mlpTop, baseIntensity } = useSlabMatrices(numLayers)

  const materials = useMemo(() => {
    return Array.from({ length: numLayers }, (_, _i) => {
      const base = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x1b2a52),
        emissive: new THREE.Color(0x0a1430),
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0.75,
        roughness: 0.35,
        metalness: 0.15,
      })
      base.userData.baseEmissive = new THREE.Color(0x0a1430)
      base.userData.baseColor = new THREE.Color(0x1b2a52)
      return base
    })
  }, [numLayers])

  useFrame((_, dt) => {
    const now = Date.now()
    for (let i = 0; i < numLayers; i++) {
      const mat = materials[i]
      const l = layers[i]
      let target = baseIntensity[i]
      if (l && l.at) {
        const age = (now - l.at) / 1000
        const decay = Math.exp(-age * 2.2)
        const normScale = Math.min(1.2, l.norm / 40)
        target = 0.15 + decay * 1.6 * normScale
      }
      const mlp = mlpTop[i]
      if (mlp) {
        const peak = Math.min(2.0, mlp.stats.max / 6)
        target = Math.max(target, 0.1 + peak * 0.5)
      }
      const e = mat.emissive
      e.lerp(new THREE.Color(0x2a5fff).multiplyScalar(target * 0.7), Math.min(1, dt * 8))
      const c = mat.color
      c.lerp(new THREE.Color(0x24366b).multiplyScalar(0.6 + target), Math.min(1, dt * 8))
      mat.emissiveIntensity = 0.4 + target * 2.2
    }
    void currentStep
  })

  return (
    <group>
      {Array.from({ length: numLayers }, (_, i) => {
        const y = layerY(i, numLayers)
        return (
          <group key={i}>
            <mesh
              ref={(el) => {
                meshRefs.current[i] = el
              }}
              position={[0, y, 0]}
              material={materials[i]}
            >
              <boxGeometry args={[5.2, 0.3, 5.2]} />
            </mesh>
            <mesh ref={(el) => (glowRefs.current[i] = el)} position={[0, y, 0]}>
              <boxGeometry args={[5.0, 0.24, 5.0]} />
              <meshBasicMaterial color="#2a5fff" transparent opacity={0.05} blending={THREE.AdditiveBlending} depthWrite={false} />
            </mesh>
          </group>
        )
      })}
      <group position={[0, layerY(0, numLayers) + LAYER_GAP / 2 + 0.35, 0]}>
        <mesh>
          <boxGeometry args={[1.6, 0.14, 1.6]} />
          <meshStandardMaterial color="#123b6e" emissive="#123b6e" emissiveIntensity={0.5} transparent opacity={0.85} />
        </mesh>
        <mesh position={[0, -0.05, 0]}>
          <boxGeometry args={[0.9, 0.06, 0.9]} />
          <meshStandardMaterial color="#0d2c52" emissive="#2a5fff" emissiveIntensity={0.25} />
        </mesh>
      </group>
    </group>
  )
}
