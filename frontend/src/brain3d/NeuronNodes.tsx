import { useMemo } from "react"
import { Text } from "@react-three/drei"
import { useBrain } from "../store/useBrainStore"
import { SLAB_W, layerY } from "./layout"

export default function NeuronNodes({ numLayers, visible }: { numLayers: number; visible: boolean }) {
  const mlpTop = useBrain((s) => s.mlpTop)
  const model = useBrain((s) => s.model)
  const selectedLayer = useBrain((s) => s.selectedLayer)
  const selectedNeuron = useBrain((s) => s.selectedNeuron)
  const selectNeuron = useBrain((s) => s.selectNeuron)
  const intermediate = model?.intermediate_size ?? 1

  const nodes = useMemo(() => {
    const result: { layer: number; index: number; value: number; rank: number; position: [number, number, number] }[] = []
    for (const [layerText, payload] of Object.entries(mlpTop)) {
      const layer = Number(layerText)
      for (let i = 0; i < payload.top.length; i++) {
        const neuron = payload.top[i]
        const column = (neuron.index / Math.max(intermediate - 1, 1)) * (SLAB_W - 0.55) - (SLAB_W - 0.55) / 2
        const row = (i % 4) * 0.15 - 0.225
        result.push({
          layer,
          index: neuron.index,
          value: neuron.value,
          rank: neuron.rank,
          position: [column, layerY(layer, numLayers) + 0.29 + Math.abs(row), row * 4],
        })
      }
    }
    return result
  }, [mlpTop, intermediate, numLayers])

  if (!visible) return null

  return (
    <group>
      {nodes.map((node) => {
        const selected = selectedNeuron?.layer === node.layer && selectedNeuron.index === node.index
        const layerVisible = selectedLayer === null || selectedLayer === node.layer
        const magnitude = Math.min(1.8, Math.abs(node.value) / 4)
        const color = node.value >= 0 ? "#ffb648" : "#c278ff"
        return (
          <group key={`${node.layer}-${node.index}`} position={node.position} visible={layerVisible}>
            <mesh
              scale={selected ? 1.8 : 0.7 + magnitude}
              onClick={(event) => {
                event.stopPropagation()
                selectNeuron(node.layer, node.index)
              }}
            >
              <sphereGeometry args={[0.075, 8, 8]} />
              <meshBasicMaterial color={selected ? "#ffffff" : color} toneMapped={false} />
            </mesh>
            {(selected || selectedLayer === node.layer) && (
              <Text position={[0, 0.16, 0]} fontSize={0.075} color={selected ? "#ffffff" : color} anchorX="center">
                #{node.index} {node.value.toFixed(2)}
              </Text>
            )}
          </group>
        )
      })}
      {nodes.length > 0 && (
        <Text position={[0, layerY(numLayers - 1, numLayers) - 0.45, 0]} fontSize={0.11} color="#ffb648" anchorX="center">
          real top-K MLP units · click a unit to inspect
        </Text>
      )}
    </group>
  )
}
