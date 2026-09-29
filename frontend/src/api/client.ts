import type {
  AttentionResponse,
  HardwareReport,
  ModelMetadata,
  ProviderDescriptor,
  ModelDescriptor,
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
  ollamaModels: () => get<{ models: string[]; status: string }>("/providers/ollama/models"),
  providerModelInfo: (provider: string, model: string) => get<import("../types").Telemetry>(`/providers/${provider}/model-info?model=${encodeURIComponent(model)}`),
  health: () => get<Record<string, unknown>>("/health"),
  hardware: () => get<HardwareReport>("/hardware"),
  models: () => get<{ supported: { model_id: string; params_m: number; description: string; recommended: boolean; verification_status: string; verification_note: string }[]; recommended: { model_id: string; reason: string } }>("/models"),
  architecture: (maxDepth = 3) => get<{ model_id: string; architecture: string; tree: Record<string, unknown> }>(`/model/architecture?max_depth=${maxDepth}`),
  providers: () => get<ProviderDescriptor[]>("/providers"),
  providerRegistry: () => get<{ providers: ProviderDescriptor[]; models: ModelDescriptor[] }>("/provider-registry"),
  validateProvider: (provider_id: string, testConnection = false) => post<{
    provider: string
    configured: boolean
    available: boolean
    configuration_status: string
    connection_status: string
    error_code: string | null
    error: string | null
  }>(`/providers/${encodeURIComponent(provider_id)}/validate`, { test_connection: testConnection }),
  model: () => get<ModelMetadata>("/model"),
  loadModel: (model_id: string) => post<{ status: string; model_id: string }>("/model/load", { model_id }),
  tokenize: (text: string, useChatTemplate = true) =>
    post<{ tokens: TokenInfo[]; count: number; time_ms: number }>("/tokenize", { text, use_chat_template: useChatTemplate }),
  sessions: () => get<SessionSummary[]>("/sessions"),
  compareSessions: (sessionA: string, sessionB: string) => get<{ session_a: SessionSummary; session_b: SessionSummary; token_similarity: number; steps_comparison: { step: number; token_a: { text?: string; token_id?: number; probability?: number } | null; token_b: { text?: string; token_id?: number; probability?: number } | null; match: boolean; prob_diff: number | null }[]; performance: Record<string, number | null> }>(`/sessions/compare?session_a=${encodeURIComponent(sessionA)}&session_b=${encodeURIComponent(sessionB)}`),
  session: (id: string) => get<Record<string, unknown>>(`/sessions/${id}`),
  sessionEvents: (id: string) => get<Record<string, unknown>[]>(`/sessions/${id}/events`),
  embedding: (sid: string, position: number) => get<TensorVectorResponse>(`/sessions/${sid}/embedding/${position}`),
  attention: (sid: string, layer: number, head: number, position: number, full = false) =>
    get<AttentionResponse>(`/sessions/${sid}/attention?layer=${layer}&head=${head}&position=${position}&full=${full ? "true" : "false"}`),
  qkv: (sid: string, layer: number, name: string, position: number, limit = 512, head?: number) =>
    get<TensorVectorResponse>(`/sessions/${sid}/qkv?layer=${layer}&name=${name}&position=${position}&limit=${limit}${head === undefined ? "" : `&head=${head}`}`),
  mlp: (sid: string, layer: number, position: number, topk = 24, neuron?: number) =>
    get<TensorVectorResponse & { neuron?: { index: number; value: number } }>(`/sessions/${sid}/mlp?layer=${layer}&position=${position}&topk=${topk}${neuron === undefined ? "" : `&neuron=${neuron}`}`),
  residual: (sid: string, layer: number, position: number) =>
    get<{ session_id: string; layer: number; position: number; metrics: Record<string, number>; all_layers: (Record<string, number> & { layer: number })[] }>(`/sessions/${sid}/residual?layer=${layer}&position=${position}`),
  logitLens: (sid: string, step: number, layer?: number, k = 5) =>
    get<{ session_id: string; step: number; layer?: number; candidates?: { token_id: number; text: string; logit: number; probability: number; rank: number }[]; layers?: { layer: number; candidates: { token_id: number; text: string; logit: number; probability: number; rank: number }[] }[] }>(`/sessions/${sid}/logit-lens?step=${step}${layer === undefined ? "" : `&layer=${layer}`}&k=${k}`),
  kvCache: (sid: string, step: number) =>
    get<{ session_id: string; step: number; seq_length: number; total_bytes: number; total_kb: number; total_mb: number; num_layers: number; layers: { layer: number; key_shape: number[]; value_shape: number[]; bytes: number; kb: number }[] }>(`/sessions/${sid}/kv-cache?step=${step}`),
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
