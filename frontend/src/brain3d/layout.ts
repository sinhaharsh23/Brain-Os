export const LAYER_GAP = 0.6
export const SLAB_W = 5.2
export const SLAB_D = 5.2
export const SLAB_H = 0.3
export const RING_RADIUS = 3.6
export const RING_Y_OFFSET = 1.6
export const OUTPUT_Y_OFFSET = -1.6
export const STACK_CENTER_Y = 0.2

export function layerY(layer: number, numLayers: number): number {
  const mid = (numLayers - 1) / 2
  return STACK_CENTER_Y + (mid - layer) * LAYER_GAP
}

export function topOfStack(numLayers: number): number {
  return layerY(0, numLayers) + SLAB_H / 2
}

export function bottomOfStack(numLayers: number): number {
  return layerY(numLayers - 1, numLayers) - SLAB_H / 2
}

export function inputY(numLayers: number): number {
  return topOfStack(numLayers) + RING_Y_OFFSET
}

export function outputY(numLayers: number): number {
  return bottomOfStack(numLayers) + OUTPUT_Y_OFFSET
}

export function tokenAngle(index: number, total: number): number {
  return (index / Math.max(total, 1)) * Math.PI * 2 - Math.PI / 2
}

export function tokenPosition(index: number, total: number): [number, number, number] {
  const a = tokenAngle(index, total)
  return [Math.cos(a) * RING_RADIUS, inputY(useNumLayers()), Math.sin(a) * RING_RADIUS]
}

export function useNumLayers(): number {
  return 24
}
