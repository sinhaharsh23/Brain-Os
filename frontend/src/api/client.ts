import type {
  AttentionResponse,
  HardwareReport,
  ModelMetadata,
  ProviderDescriptor,
  SessionSummary,
  TensorVectorResponse,
  TokenInfo,
} from "../types"

const BASE = "/api"
const envToken = import.meta.env.VITE_BRAINOS_TOKEN as string | undefined

export class ApiError extends Error {
  public status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

function token(): string | undefined {
  return localStorage.getItem("brainos_auth_token") ?? envToken
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const current = token()
  return current ? { ...extra, "X-BrainOS-Token": current } : extra
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { headers: headers() })
  if (!r.ok) throw new ApiError(r.status, `${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new ApiError(r.status, `${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}

export const api = {
  health: () => get<Record<string, unknown>>("/health"),
  hardware: () => get<HardwareReport>("/hardware"),
  models: () => get<{ supported: { model_id: string; params_m: number; description: string; recommended: boolean; verification_status: string; verification_note: string }[]; recommended: { model_id: string; reason: string } }>("/models"),
  providers: () => get<ProviderDescriptor[]>("/providers"),
  model: () => get<ModelMetadata>("/model"),
  loadModel: (model_id: string) => post<{ status: string; model_id: string }>("/model/load", { model_id }),
  tokenize: (text: string, useChatTemplate = true) =>
    post<{ tokens: TokenInfo[]; count: number; time_ms: number }>("/tokenize", { text, use_chat_template: useChatTemplate }),
  sessions: () => get<SessionSummary[]>("/sessions"),
  session: (id: string) => get<Record<string, unknown>>(`/sessions/${id}`),
  sessionEvents: (id: string) => get<Record<string, unknown>[]>(`/sessions/${id}/events`),
  embedding: (sid: string, position: number) => get<TensorVectorResponse>(`/sessions/${sid}/embedding/${position}`),
  attention: (sid: string, layer: number, head: number, position: number) =>
    get<AttentionResponse>(`/sessions/${sid}/attention?layer=${layer}&head=${head}&position=${position}`),
  qkv: (sid: string, layer: number, name: string, position: number) =>
    get<TensorVectorResponse>(`/sessions/${sid}/qkv?layer=${layer}&name=${name}&position=${position}`),
  mlp: (sid: string, layer: number, position: number, topk = 24, neuron?: number) =>
    get<TensorVectorResponse & { neuron?: { index: number; value: number } }>(`/sessions/${sid}/mlp?layer=${layer}&position=${position}&topk=${topk}${neuron === undefined ? "" : `&neuron=${neuron}`}`),
  hidden: (sid: string, layer: number, position: number) =>
    get<TensorVectorResponse>(`/sessions/${sid}/hidden?layer=${layer}&position=${position}`),
  logits: (sid: string, step: number, k = 16) =>
    get<{ candidates: { token_id: number; text: string; logit: number; probability: number; rank: number }[] }>(`/sessions/${sid}/logits?step=${step}&k=${k}`),
  monitoringHistory: () => get<{ history: unknown[] }>("/monitoring/history"),
  authMe: () => get<{ mode: "local" | "multi_user"; user: import("../types").AuthUser | null }>("/auth/me"),
  authRegister: (username: string, password: string) => post<{ user: import("../types").AuthUser; token: string }>("/auth/register", { username, password }),
  authLogin: (username: string, password: string) => post<{ user: import("../types").AuthUser; token: string }>("/auth/login", { username, password }),
  authLogout: () => post<{ status: string }>("/auth/logout", {}),
  scheduler: () => get<{ queue_size: number; queue_limit: number; active: number; active_limit: number; runs: number }>("/scheduler"),
}
