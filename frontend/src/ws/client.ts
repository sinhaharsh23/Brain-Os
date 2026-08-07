import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"

let ws: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let reconnectDelay = 500

export function connect(): void {
  const proto = window.location.protocol === "https:" ? "wss" : "ws"
  const url = `${proto}://${window.location.hostname}:8765/ws`
  try {
    ws = new WebSocket(url)
  } catch {
    scheduleReconnect()
    return
  }
  ws.onopen = () => {
    reconnectDelay = 500
    useBrain.getState().set({ connected: true })
    api.sessions().then((s) => useBrain.getState().set({ sessions: s })).catch(() => {})
  }
  ws.onmessage = (msg) => {
    try {
      const ev = JSON.parse(msg.data as string)
      if (ev.type === "ack") return
      useBrain.getState().dispatch(ev)
    } catch {
      /* malformed message */
    }
  }
  ws.onclose = () => {
    useBrain.getState().set({ connected: false })
    scheduleReconnect()
  }
  ws.onerror = () => {
    try {
      ws?.close()
    } catch {
      /* ignore */
    }
  }
}

function scheduleReconnect(): void {
  if (reconnectTimer) return
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null
    connect()
  }, reconnectDelay)
  reconnectDelay = Math.min(reconnectDelay * 2, 10000)
}

export function send(action: string, payload: Record<string, unknown> = {}): void {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action, ...payload }))
  }
}

export function disconnect(): void {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  ws?.close()
  ws = null
}
