import { useMemo, useRef } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import { useBrain } from "../store/useBrainStore"
import { inputY, layerY, outputY } from "./layout"

export default function FlowParticle({ numLayers }: { numLayers: number }) {
  const ref = useRef<THREE.Mesh>(null)
  const trail = useRef<THREE.Points>(null)

  const y = useMemo(() => {
    const obj = { value: inputY(numLayers) }
    return obj
  }, [numLayers])

  const points = useMemo(() => {
    const arr = new Float32Array(80 * 3)
    const geo = new THREE.BufferGeometry()
    geo.setAttribute("position", new THREE.BufferAttribute(arr, 3))
    return geo
  }, [])

  useFrame(() => {
    const st = useBrain.getState()
    const layers = st.layers
    let target = inputY(numLayers)
    let found = -1
    const step = st.currentStep
    if (st.running) {
      let maxLayer = -1
      for (const [k, v] of Object.entries(layers)) {
        if (v.step === step) maxLayer = Math.max(maxLayer, Number(k))
      }
      if (maxLayer >= 0) {
        target = layerY(maxLayer, numLayers)
        found = maxLayer
      }
    } else {
      target = outputY(numLayers)
    }
    y.value += (target - y.value) * Math.min(1, 0.12)
    if (ref.current) {
      ref.current.position.y = y.value
      const t = (y.value - outputY(numLayers)) / (inputY(numLayers) - outputY(numLayers))
      const hue = 0.55 + t * 0.15
      ;(ref.current.material as THREE.MeshBasicMaterial).color.setHSL(hue, 1, 0.55)
    }
    if (trail.current) {
      const pos = trail.current.geometry.attributes.position as THREE.BufferAttribute
      const arr = pos.array as Float32Array
      for (let i = arr.length / 3 - 1; i > 0; i--) {
        arr[i * 3] = arr[(i - 1) * 3]
        arr[i * 3 + 1] = arr[(i - 1) * 3 + 1]
        arr[i * 3 + 2] = arr[(i - 1) * 3 + 2]
      }
      arr[0] = 0
      arr[1] = y.value
      arr[2] = 0
      pos.needsUpdate = true
      void found
    }
  })

  return (
    <group>
      <mesh ref={ref} position={[0, y.value, 0]}>
        <sphereGeometry args={[0.22, 12, 12]} />
        <meshBasicMaterial color="#4cc2ff" toneMapped={false} />
      </mesh>
      <points ref={trail} geometry={points}>
        <pointsMaterial color="#4cc2ff" size={0.09} transparent opacity={0.5} sizeAttenuation depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
    </group>
  )
}
