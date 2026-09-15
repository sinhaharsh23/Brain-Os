import { useBrain } from "../store/useBrainStore"
import { api } from "../api/client"

let ws: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let reconnectDelay = 500
let shouldReconnect = true

export function connect(): void {
  shouldReconnect = true
  const proto = window.location.protocol === "https:" ? "wss" : "ws"
  const token = localStorage.getItem("brainos_auth_token") ?? (import.meta.env.VITE_BRAINOS_TOKEN as string | undefined)
  const configured = import.meta.env.VITE_BRAINOS_WS_URL as string | undefined
  const baseUrl = configured ?? `${proto}://${window.location.host}/ws`
  const url = `${baseUrl}${token ? `${baseUrl.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}` : ""}`
  let socket: WebSocket
  try {
    socket = new WebSocket(url)
  } catch {
    if (shouldReconnect) scheduleReconnect()
    return
  }
  ws = socket
  socket.onopen = () => {
    reconnectDelay = 500
    useBrain.getState().set({ connected: true })
    api.sessions().then((s) => useBrain.getState().set({ sessions: s })).catch(() => {})
  }
  socket.onmessage = (msg) => {
    try {
      const ev = JSON.parse(msg.data as string)
      if (ev.type === "ack") {
        const data = ev.data ?? {}
        useBrain.getState().set({ runId: data.run_id ?? null, requestId: data.request_id ?? null, sessionId: data.session_id ?? null, queuePosition: data.queue_position ?? null, runStatus: "QUEUED" })
        return
      }
      if (ev.type === "error") {
        const data = ev.data ?? {}
        useBrain.getState().set({ inferenceError: String(data.message ?? "WebSocket error"), runStatus: String(data.code ?? "ERROR") })
        return
      }
      useBrain.getState().dispatch(ev)
    } catch {
      /* malformed message */
    }
  }
  socket.onclose = () => {
    if (ws !== socket) return
    useBrain.getState().set({ connected: false })
    if (shouldReconnect) scheduleReconnect()
  }
  socket.onerror = () => {
    try {
      socket.close()
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
  shouldReconnect = false
  if (reconnectTimer) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  const socket = ws
  ws = null
  socket?.close()
}
