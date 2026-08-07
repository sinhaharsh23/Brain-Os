import { useMemo } from "react"
import * as THREE from "three"
import { useBrain } from "../store/useBrainStore"
import { RING_RADIUS, inputY, layerY, tokenAngle } from "./layout"

interface AttentionLinksProps {
  numLayers: number
  visible: boolean
}

export default function AttentionLinks({ numLayers, visible }: AttentionLinksProps) {
  const links = useBrain((s) => s.attentionLinks)
  const totalTokens = useBrain((s) => s.tokens.length + s.generatedTokens.length)
  const activePosition = useBrain((s) => s.currentStep)
  const selectedToken = useBrain((s) => s.selectedToken)
  const selectedHead = useBrain((s) => s.selectedHead)
  const model = useBrain((s) => s.model)

  const geometry = useMemo(() => {
    const positions: number[] = []
    const colors: number[] = []
    const allLayerLinks = Object.entries(links)
      .map(([layerStr, ls]) => ({ layer: Number(layerStr), ls }))
      .sort((a, b) => a.layer - b.layer)
    for (const { layer, ls } of allLayerLinks) {
      const fromY = layerY(layer, numLayers)
      for (const l of ls) {
        if (selectedHead !== null && l.head !== selectedHead) continue
        const toA = tokenAngle(l.token_index, Math.max(totalTokens, 1))
        const toY = inputY(numLayers)
        positions.push(0, fromY, 0)
        positions.push(Math.cos(toA) * RING_RADIUS, toY, Math.sin(toA) * RING_RADIUS)
        const hue = l.head / Math.max(model?.num_attention_heads ?? 14, 1)
        const c = new THREE.Color().setHSL(hue, 0.85, 0.55)
        const b = 0.35 + Math.min(1, l.weight * 3) * 0.65
        colors.push(c.r * b, c.g * b, c.b * b, c.r * b, c.g * b, c.b * b)
      }
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3))
    return geo
  }, [links, numLayers, totalTokens, selectedHead, model])

  void activePosition
  void selectedToken

  if (!visible || geometry.attributes.position.count === 0) return null

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial vertexColors transparent opacity={0.75} depthWrite={false} />
    </lineSegments>
  )
}
