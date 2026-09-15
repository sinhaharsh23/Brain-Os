export interface ModelMetadata {
  model_id: string
  architecture: string
  num_params: number
  num_layers: number
  hidden_size: number
  num_attention_heads: number
  num_kv_heads: number
  head_dim: number
  intermediate_size: number
  vocab_size: number
  context_length: number
  max_position_embeddings: number
  activation_function: string
  dtype: string
  quantization: string
  device: string
  tokenizer_name: string
  extra: Record<string, unknown>
  capabilities: Record<string, boolean>
}

export interface HardwareReport {
  cpu: { count: number; percent: number; model: string }
  ram: { total_gb: number; available_gb: number; used_gb: number; percent: number }
  gpu: { vendor: string; name: string; vram_total_gb: number | null; vram_used_gb: number | null; compute_available: boolean; pci_slot?: string | null } | null
  gpu_vendor: string | null
  gpu_name: string | null
  backend: "CPU" | "MPS" | "CUDA" | "ROCm"
  device: string
  model_device: string
  requested_device: "auto" | "cpu" | "mps" | "cuda"
  gpu_available: boolean
  cuda_available: boolean
  rocm_available: boolean
  torch_version: string
  cuda_version: string | null
  hip_version: string | null
}

export interface ProviderDescriptor {
  provider_id: string
  display_name: string
  kind: string
  inspection_mode: "deep" | "limited"
  availability: string
  limitation: string
  capabilities: Record<string, boolean>
  models: string[]
}

export interface AuthUser {
  id: string
  username: string
  created_at: string
}

export interface QueueState {
  runId: string | null
  requestId: string | null
  runStatus: string
  queuePosition: number | null
  queueLimit: number | null
}

export interface ExternalUsage {
  input_tokens?: number
  output_tokens?: number
  total_tokens?: number
}

export interface TokenInfo {
  text: string
  id: number
  position: number
  is_special: boolean
}

export interface AttentionLink {
  head: number
  token_index: number
  weight: number
}

export interface LayerLinks {
  layer: number
  links: AttentionLink[]
  active_position: number
}

export interface LayerComplete {
  layer: number
  step: number
  is_first: boolean
  hidden_norm: number
  hidden_mean: number
  hidden_std: number
  input_shape: number[]
  output_shape: number[]
}

export interface Candidate {
  token_id: number
  text: string
  logit: number
  probability: number
  rank: number
}

export interface GenToken {
  step: number
  token_id: number
  text: string
  probability: number
  rank: number | null
  position: number
  time_ms: number
  pca3: [number, number, number] | null
}

export interface PcaToken {
  position: number
  text: string
  id: number
  norm: number
  pca3: [number, number, number]
}

export interface EmbeddingsComplete {
  tokens: PcaToken[]
  count: number
  explained_variance: number[]
  pca_method: string
  embedding_dim: number
}

export interface QkvStat {
  layer: number
  name: string
  stats: TensorStats
}

export interface MlpTop {
  layer: number
  top: { index: number; value: number; rank: number }[]
  stats: TensorStats
}

export interface TensorStats {
  n: number
  min: number
  max: number
  mean: number
  std: number
  l2_norm: number
}

export interface MonitorSnapshot {
  cpu_percent: number
  ram_total_gb: number
  ram_used_gb: number
  ram_available_gb: number
  ram_percent: number
  process_ram_gb: number
  gpu_percent: number | null
  vram_used_gb: number | null
  vram_total_gb: number | null
  model_loaded: boolean
  ts: number
}

export interface Summary {
  session_id: string
  prompt: string
  model_id: string
  status: string
  created_at: number
  num_tokens: number
  num_output_tokens: number | null
  response: string
  timings: Record<string, number | boolean | null>
  errors: string[]
  provider?: string
  inspection_mode?: "deep" | "limited"
  usage?: ExternalUsage | null
}

export interface SessionSummary {
  session_id: string
  prompt: string
  model_id: string
  created_at: number
  status: string
  num_output_tokens: number | null
  response: string
  timings: Record<string, number | boolean>
}

export interface TensorVectorResponse {
  layer?: number
  name?: string
  position: number
  stats: TensorStats
  values?: { index: number; value: number }[]
  top?: { index: number; value: number; rank: number }[]
  vector?: number[]
  pca3?: number[] | null
  dimension?: number
  shape?: number[]
  dtype?: string
  token?: TokenInfo | GenToken
  neuron?: { index: number; value: number }
}

export interface AttentionResponse {
  layer: number
  head: number
  position: number
  total_tokens: number
  weights: { token_index: number; weight: number }[]
  stats: TensorStats
  is_full_matrix: boolean
}

export interface DevEntry {
  level: string
  message: string
  ts: number
}

export interface TimelineEntry {
  type: string
  label: string
  ts: number
  step?: number
  layer?: number
  detail?: string
}

export interface ReplayState {
  sessionId: string | null
  playing: boolean
  paused: boolean
  speed: number
  status: string
  index: number
  count: number
}
