import type { Telemetry } from "../types"
import { create } from "zustand"
import type {
  AttentionLink,
  AuthUser,
  Candidate,
  ChatMessage,
  DevEntry,
  EmbeddingsComplete,
  ExternalUsage,
  GenToken,
  HardwareReport,
  LayerLinks,
  LayerComplete,
  MlpTop,
  ModelMetadata,
  ModelDescriptor,
  ProviderDescriptor,
  MonitorSnapshot,
  KvCacheSnapshot,
  ResidualSummary,
  QkvStat,
  ReplayState,
  SessionSummary,
  Summary,
  TimelineEntry,
  TokenInfo,
} from "../types"

export type ViewMode =
  | "command-center"
  | "hologram"
  | "neural-interface"
  | "neuro-core"
  | "brain"
  | "architecture"
  | "attention"
  | "embedding"
  | "tokenflow"
  | "introspection"
  | "memory-matrix"
  | "data-streams"
  | "model-hub"
  | "training-hub"
  | "plugins"
  | "system-monitor"
  | "settings"
  | "dev"

export interface LayerState {
  norm: number
  mean: number
  std: number
  step: number
  timeMs: number
  active: boolean
  at: number
  status: "idle" | "waiting" | "processing" | "processing-attention" | "processing-mlp" | "completed" | "selected" | "error"
}

export type TransformerViewMode = "architecture" | "flow" | "residual" | "attention" | "activation" | "cache" | "performance"

export interface InspectorData {
  kind: "token" | "layer" | "attention" | "neuron" | "embedding" | "qkv" | "logits"
  title: string
  rows: { label: string; value: string }[]
  table?: { headers: string[]; rows: (string | number)[][] }
}

export interface BrainState {
  telemetry: Telemetry | null
  providerLogprobs: { token: string; logprob: number; top_logprobs?: { token: string; logprob: number }[] }[]
  nativeModel: ModelMetadata | null
  switchProvider: (provider: string, model: string) => void
  connected: boolean
  modelStatus: string
  provider: string | null
  inspectionMode: "deep" | "limited" | null
  providerModel: string | null
  providerCatalog: ProviderDescriptor[]
  modelCatalog: ModelDescriptor[]
  chatMessages: ChatMessage[]
  usage: ExternalUsage | null
  model: ModelMetadata | null
  hardware: HardwareReport | null
  sessionId: string | null
  running: boolean
  paused: boolean
  currentStep: number
  prompt: string
  chatTemplate: {
    available: boolean
    name: string
    serialized: string
    message_count: number
    has_system_message: boolean
    generation_prompt: boolean
    error?: string
    reason?: string
  } | null
  tokens: TokenInfo[]
  generatedTokens: GenToken[]
  response: string
  layers: Record<number, LayerState>
  layersByStep: Record<number, Record<number, LayerState>>
  attentionLinks: Record<number, AttentionLink[]>
  attentionLinksByStep: Record<number, Record<number, AttentionLink[]>>
  qkvStats: QkvStat[]
  qkvStatsByStep: Record<number, QkvStat[]>
  mlpTop: Record<number, MlpTop>
  mlpTopByStep: Record<number, Record<number, MlpTop>>
  candidates: Candidate[] | null
  candidatesByStep: Record<number, Candidate[]>
  lastLogits: { step: number; temperature: number; top_p: number; top_k: number } | null
  lastLogitsByStep: Record<number, { step: number; temperature: number; top_p: number; top_k: number }>
  pca: EmbeddingsComplete | null
  monitoring: MonitorSnapshot | null
  kvCache: Record<number, KvCacheSnapshot>
  residuals: Record<number, ResidualSummary[]>
  logitLens: Record<number, { layer: number; candidates: Candidate[] }[]>
  devLog: DevEntry[]
  timeline: TimelineEntry[]
  sessions: SessionSummary[]
  replay: ReplayState
  inspector: InspectorData | null
  selectedToken: number | null
  selectedStep: number | null
  selectedLayer: number | null
  activeLayer: number | null
  activeModule: string | null
  transformerViewMode: TransformerViewMode
  selectedHead: number | null
  selectedNeuron: { layer: number; index: number } | null
  view: ViewMode
  summary: Summary | null
  inferenceError: string | null
  user: AuthUser | null
  authMode: "local" | "multi_user"
  authRequired: boolean
  runId: string | null
  requestId: string | null
  runStatus: string
  queuePosition: number | null
  queueLimit: number | null

  dispatch: (ev: { type: string; data: Record<string, unknown>; ts: number; session_id: string | null }) => void
  set: (patch: Partial<BrainState>) => void
  selectToken: (pos: number | null) => void
  selectStep: (step: number | null) => void
  selectLayer: (layer: number | null) => void
  selectHead: (head: number | null) => void
  selectNeuron: (layer: number, index: number) => void
  setInspector: (data: InspectorData | null) => void
  setView: (v: ViewMode) => void
  setTransformerViewMode: (mode: TransformerViewMode) => void
  addChatMessage: (message: ChatMessage) => void
}

function tokenLabel(text: string): string {
  return text.replace(/Ġ/g, " ").replace(/Ċ/g, "\n").replace(/\n/g, "\\n")
}

function pushLog(entries: DevEntry[], message: string, level = "info", ts = Date.now()): DevEntry[] {
  const browserTs = ts < 100000000000 ? ts * 1000 : ts
  return [...entries.slice(-499), { level, message, ts: browserTs }]
}

function pushTimeline(entries: TimelineEntry[], e: TimelineEntry): TimelineEntry[] {
  return [...entries.slice(-499), e]
}

function traceReset(): Partial<BrainState> {
  return {
    telemetry: null,
    providerLogprobs: [],
    usage: null,
    replay: { sessionId: null, playing: false, paused: false, speed: 1, status: "idle", index: 0, count: 0 },
    sessionId: null,    running: false,
    paused: false,
    currentStep: 0,
    prompt: "",
    chatTemplate: null,
    tokens: [],
    generatedTokens: [],
    response: "",
    layers: {},
    layersByStep: {},
    attentionLinks: {},
    attentionLinksByStep: {},
    qkvStats: [],
    qkvStatsByStep: {},
    mlpTop: {},
    mlpTopByStep: {},
    candidates: null,
    candidatesByStep: {},
    lastLogits: null,
    lastLogitsByStep: {},
    pca: null,
    kvCache: {},
    residuals: {},
    logitLens: {},
    summary: null,
    inspector: null,
    selectedToken: null,
    selectedStep: null,
    selectedLayer: null,
    activeLayer: null,
    activeModule: null,
    transformerViewMode: "architecture",
    selectedHead: null,
    selectedNeuron: null,
    inferenceError: null,
    runId: null,
    requestId: null,
    runStatus: "IDLE",
    queuePosition: null,
    queueLimit: null,
    timeline: [],
  }
}

const retiredSessions = new Set<string>()

export const useBrain = create<BrainState>((set, get) => ({
  telemetry: null,
  providerLogprobs: [],
  nativeModel: null,
  switchProvider: (provider, modelId) => {
    const previous = get()
    if (previous.sessionId) retiredSessions.add(previous.sessionId)
    set({ ...traceReset(), provider, providerModel: modelId || null, model: provider === "qwen-local" ? previous.nativeModel : null,
      inspectionMode: provider === "qwen-local" ? "deep" : "limited", devLog: [],
      modelStatus: provider === "qwen-local" && previous.nativeModel ? "loaded" : "not_loaded" })
  },
  connected: false,
  modelStatus: "not_loaded",
  provider: null,
  inspectionMode: null,
    providerModel: null,
    providerCatalog: [],
    modelCatalog: [],
    chatMessages: [],
  usage: null,
  model: null,
  hardware: null,
  sessionId: null,
  running: false,
  paused: false,
  currentStep: 0,
  prompt: "",
  chatTemplate: null,
  tokens: [],
  generatedTokens: [],
  response: "",
  layers: {},
  layersByStep: {},
  attentionLinks: {},
  attentionLinksByStep: {},
  qkvStats: [],
  qkvStatsByStep: {},
  mlpTop: {},
  mlpTopByStep: {},
  candidates: null,
  candidatesByStep: {},
  lastLogits: null,
  lastLogitsByStep: {},
  pca: null,
  monitoring: null,
  kvCache: {},
  residuals: {},
  logitLens: {},
  devLog: [],
  timeline: [],
  sessions: [],
  replay: { sessionId: null, playing: false, paused: false, speed: 1, status: "idle", index: 0, count: 0 },
  inspector: null,
  selectedToken: null,
  selectedStep: null,
  selectedLayer: null,
  activeLayer: null,
  activeModule: null,
  transformerViewMode: "architecture",
  selectedHead: null,
  selectedNeuron: null,
  // Observatory is the single primary shell. Inspectors render inside it
  // when a module is selected from its navigation.
  view: "command-center",
  summary: null,
  inferenceError: null,
  user: null,
  authMode: "local",
  authRequired: false,
  runId: null,
  requestId: null,
  runStatus: "IDLE",
  queuePosition: null,
  queueLimit: null,

  set: (patch) => set(patch),

  dispatch: (ev) => {
    const { data, ts } = ev
    const st = get()
    if (ev.type === "replay.loaded" && ev.session_id) retiredSessions.delete(ev.session_id)
    else if (ev.session_id && retiredSessions.has(ev.session_id)) return

    const cloudEvent = ev.type.startsWith("generation.") || ev.type.startsWith("provider.") || ev.type.startsWith("response.") || ev.type.startsWith("external.") || ev.type === "input.prepared"
    const localTraceEvent = ev.type.startsWith("inference.") || ev.type.startsWith("token") || ev.type.startsWith("step.") || ev.type.startsWith("layer.") || ev.type.endsWith(".captured") || ev.type === "logits.ready" || ev.type === "embeddings.complete"
    if ((st.provider === "qwen-local" && cloudEvent) || (st.inspectionMode === "limited" && localTraceEvent)) return

    // A cancelled run can finish delivering an event after the next run has
    // already been queued. Ignore that stale session instead of allowing it
    // to reset the current inspection context.
    if (
      ev.session_id &&
      st.sessionId &&
      ev.session_id !== st.sessionId &&
      !["inference.queued", "inference.started", "external.started", "replay.loaded"].includes(ev.type)
    ) {
      return
    }

    switch (ev.type) {
      case "telemetry.updated":
        set({ telemetry: data as unknown as Telemetry })
        break
      case "provider.logprobs":
        set({ providerLogprobs: [...st.providerLogprobs, ...(data.tokens as BrainState["providerLogprobs"])] })
        break
      case "provider.first_content":
      case "provider.started":
      case "provider.connected":
      case "provider.request_submitted":
      case "provider.metrics_finalized":
      case "input.prepared":
      case "provider.usage":
        set({ ...(data.usage ? { usage: data.usage as ExternalUsage } : {}), devLog: pushLog(st.devLog, `${st.provider}: ${ev.type} ${JSON.stringify(data)}`) })
        break
      case "system.ready":
        set({ devLog: pushLog(st.devLog, `connected to backend (${(data as { ws_url: string }).ws_url})`) })
        break
      case "system.error":
        set({
          devLog: pushLog(st.devLog, `ERROR: ${String(data.message)}`, "error"),
          inferenceError: String(data.message),
          modelStatus: data.stage === "model_load" ? "error" : st.modelStatus,
          running: data.stage === "external_provider" || data.stage === "model_load" ? false : st.running,
          paused: data.stage === "external_provider" || data.stage === "model_load" ? false : st.paused,
          runStatus: data.code === "queue_full" ? "QUEUE_FULL" : st.runStatus,
        })
        break
      case "inference.queued":
        set({
          running: true,
          runId: String(data.run_id ?? st.runId ?? ""),
          requestId: String(data.request_id ?? st.requestId ?? ""),
          sessionId: String(data.session_id ?? ev.session_id ?? ""),
          runStatus: "QUEUED",
          queuePosition: Number(data.queue_position ?? 0),
          queueLimit: Number(data.queue_limit ?? 0),
          devLog: pushLog(st.devLog, `run queued at position ${String(data.queue_position ?? "?")}`),
        })
        break
      case "inference.timeout":
        set({ running: false, paused: false, runStatus: "TIMED_OUT", inferenceError: `inference timed out after ${String(data.timeout_s)}s` })
        break
      case "inference.failed":
        set({ running: false, paused: false, runStatus: "FAILED", inferenceError: String(data.message ?? "inference failed"), devLog: pushLog(st.devLog, `inference failed: ${String(data.message ?? "unknown error")}`, "error"), chatMessages: st.chatMessages.map((message, index, messages) => index === messages.map((item) => item.role).lastIndexOf("assistant") ? { ...message, status: "error", metadata: { ...(message.metadata ?? {}), error: String(data.message ?? "inference failed") } } : message) })
        break
      case "generation.started":
      case "external.started":
        set({
          telemetry: null, providerLogprobs: [],
          running: true,
          paused: false,
          inferenceError: null,
          provider: String(data.provider ?? "external"),
          inspectionMode: "limited",
          providerModel: String(data.model ?? ""),
          model: null,
          usage: null,
          sessionId: (ev.session_id as string) ?? null,
          prompt: String(data.prompt ?? ""),
          chatTemplate: null,
          tokens: [],
          generatedTokens: [],
          response: "",
          layers: {},
          layersByStep: {},
          attentionLinks: {},
          attentionLinksByStep: {},
          qkvStats: [],
          qkvStatsByStep: {},
          mlpTop: {},
          mlpTopByStep: {},
          candidates: null,
          candidatesByStep: {},
          lastLogits: null,
          lastLogitsByStep: {},
          pca: null,
          kvCache: {},
          residuals: {},
          logitLens: {},
          summary: null,
          selectedToken: null,
          selectedStep: null,
          selectedLayer: null,
          selectedHead: null,
          selectedNeuron: null,
          activeLayer: null,
          activeModule: null,
          runStatus: "RUNNING",
          timeline: pushTimeline(st.timeline, { type: "external", label: `${String(data.provider)} external observation started`, ts }),
          devLog: pushLog(st.devLog, `${String(data.provider)} runtime: provider-visible events only`),
          chatMessages: st.chatMessages.map((message, index, messages) => index === messages.map((item) => item.role).lastIndexOf("assistant") ? { ...message, provider: String(data.provider ?? "external"), model: String(data.model ?? ""), mode: "cloud", status: "streaming" } : message),
        })
        break
      case "model.metadata":
        // Provider-reported metadata does not imply direct access to the
        // provider's internal transformer graph. Keep Ollama in limited mode.
        set({ model: data.metadata as ModelMetadata, provider: String(data.provider ?? st.provider ?? "ollama"), providerModel: String(data.model ?? st.providerModel ?? ""), inspectionMode: "limited" })
        break
      case "inference.paused":
        set({ paused: true, devLog: pushLog(st.devLog, `inference paused after step ${String(data.step)}`), timeline: pushTimeline(st.timeline, { type: "pause", label: `paused after step ${String(data.step)}`, step: data.step as number, ts }) })
        break
      case "inference.resumed":
        set({ paused: false, devLog: pushLog(st.devLog, `inference resumed at step ${String(data.step)}`), timeline: pushTimeline(st.timeline, { type: "resume", label: `resumed at step ${String(data.step)}`, step: data.step as number, ts }) })
        break
      case "external.chunk":
        set({
          response: `${st.response}${String(data.text ?? "")}`,
          timeline: pushTimeline(st.timeline, { type: "external", label: `external response chunk`, ts }),
        })
        break
      case "response.text.delta":
        set({
          response: `${st.response}${String(data.text ?? "")}`,
          chatMessages: st.chatMessages.map((message, index, messages) => index === messages.map((item) => item.role).lastIndexOf("assistant") ? { ...message, content: `${message.content}${String(data.text ?? "")}`, status: "streaming" } : message),
          timeline: pushTimeline(st.timeline, { type: "external", label: "provider stream delta", ts }),
        })
        break
      case "external.usage":
        set({ usage: data.usage as ExternalUsage })
        break
      case "generation.completed":
        set({
          running: false,
          paused: false,
          runStatus: "COMPLETED",
          summary: data.summary as Summary,
          usage: (data.usage as ExternalUsage | null | undefined) ?? ((data.summary as Summary | undefined)?.usage ?? null),
          chatMessages: st.chatMessages.map((message, index, messages) => index === messages.map((item) => item.role).lastIndexOf("assistant") ? { ...message, content: String(data.response ?? message.content), status: "complete", responseId: data.response_id == null ? null : String(data.response_id), usage: data.usage as ExternalUsage | null } : message),
          timeline: pushTimeline(st.timeline, { type: "complete", label: "cloud response complete", ts }),
        })
        break
      case "generation.cancelled":
        set({ running: false, paused: false, runStatus: "CANCELLED", chatMessages: st.chatMessages.map((message, index, messages) => index === messages.map((item) => item.role).lastIndexOf("assistant") ? { ...message, status: "cancelled" } : message) })
        break
      case "generation.error":
        set({ running: false, paused: false, runStatus: "FAILED", inferenceError: String(data.message ?? "provider request failed"), chatMessages: st.chatMessages.map((message, index, messages) => index === messages.map((item) => item.role).lastIndexOf("assistant") ? { ...message, status: "error", metadata: { ...(message.metadata ?? {}), error: String(data.message ?? "provider request failed") } } : message) })
        break
      case "model.ready":
        set({ nativeModel: data.metadata as ModelMetadata, ...(!st.provider || st.provider === "qwen-local" ? { model: data.metadata as ModelMetadata, modelStatus: "loaded", provider: "qwen-local", inspectionMode: "deep" as const } : {}) })
        break
      case "monitoring.tick":
        set({ monitoring: data as unknown as MonitorSnapshot })
        break
      case "inference.started":
        set({
          telemetry: null, providerLogprobs: [],
          running: true,
          inferenceError: null,
          provider: String(data.provider ?? "qwen-local"),
          inspectionMode: data.inspection_mode === "deep" || !data.inspection_mode ? "deep" : "limited",
          usage: null,
          sessionId: (ev.session_id as string) ?? null,
          runId: typeof data.run_id === "string" ? data.run_id : st.runId,
          requestId: typeof data.request_id === "string" ? data.request_id : st.requestId,
          runStatus: data.phase === "scheduler" ? "STARTING" : "RUNNING",
          queuePosition: null,
          prompt: String(data.prompt ?? ""),
          tokens: [],
          generatedTokens: [],
          response: "",
          layers: {},
          layersByStep: {},
          attentionLinks: {},
          attentionLinksByStep: {},
          qkvStats: [],
          qkvStatsByStep: {},
          mlpTop: {},
          mlpTopByStep: {},
          candidates: null,
          candidatesByStep: {},
          lastLogits: null,
          lastLogitsByStep: {},
          pca: null,
          kvCache: {},
          residuals: {},
          logitLens: {},
          summary: null,
          timeline: [],
          selectedToken: null,
          selectedStep: null,
          devLog: pushLog(st.devLog, `inference started: "${String(data.prompt).slice(0, 80)}"`),
          selectedLayer: null,
          selectedHead: null,
          selectedNeuron: null,
          activeLayer: null,
          activeModule: null,
          chatTemplate: null,
        })
        break
      case "tokenization.complete": {
        const tokens = data.tokens as TokenInfo[]
        set({
          tokens,
          chatTemplate: data.chat_template as BrainState["chatTemplate"],
          devLog: pushLog(st.devLog, `tokenization: ${tokens.length} tokens in ${String(data.time_ms)}ms`),
          timeline: pushTimeline(st.timeline, { type: "tokenization", label: `tokenized ${tokens.length} tokens`, ts }),
        })
        break
      }
      case "embeddings.complete":
        set({
          pca: data as unknown as EmbeddingsComplete,
          devLog: pushLog(st.devLog, `embeddings: ${String(data.count)} vectors, dim ${String(data.embedding_dim)} (PCA ${String(data.pca_method)})`),
          timeline: pushTimeline(st.timeline, { type: "embeddings", label: `embedded ${String(data.count)} tokens`, ts }),
        })
        break
      case "step.started":
        set({
          currentStep: data.step as number,
          selectedStep: data.step as number,
          layers: st.layersByStep[Number(data.step)] ?? {},
          attentionLinks: st.attentionLinksByStep[Number(data.step)] ?? {},
          qkvStats: st.qkvStatsByStep[Number(data.step)] ?? [],
          mlpTop: st.mlpTopByStep[Number(data.step)] ?? {},
          candidates: st.candidatesByStep[Number(data.step)] ?? null,
          lastLogits: st.lastLogitsByStep[Number(data.step)] ?? null,
          timeline: pushTimeline(st.timeline, {
            type: "step",
            label: `step ${String(data.step)}: ${String(data.new_token ?? "(prompt)")}`,
            step: data.step as number,
            ts,
          }),
        })
        break
      case "layer.started":
      case "attention.started":
      case "mlp.started": {
        const layer = Number(data.layer)
        const step = Number(data.step ?? st.currentStep)
        const previous = st.layersByStep[step]?.[layer]
        const state = previous ?? { norm: 0, mean: 0, std: 0, step, timeMs: 0, active: true, at: Date.now(), status: "waiting" as const }
        const module = String(data.module ?? "layer")
        const nextStatus: LayerState["status"] = module === "attention" ? "processing-attention" : module === "mlp" ? "processing-mlp" : "processing"
        const stepLayers = { ...(st.layersByStep[step] ?? {}), [layer]: { ...state, active: true, status: nextStatus, at: Date.now() } }
        set({ activeLayer: layer, activeModule: module, layers: step === st.selectedStep ? stepLayers : st.layers, layersByStep: { ...st.layersByStep, [step]: stepLayers }, timeline: pushTimeline(st.timeline, { type: "layer", label: `${module} ${layer} started`, step, layer, ts }) })
        break
      }
      case "layer.completed":
      case "attention.completed":
      case "mlp.completed":
        // The subsequent summary event supplies the real norms. Keep the
        // completion marker here so playback can distinguish it from waiting.
        set({ activeLayer: Number(data.layer), activeModule: String(data.module ?? "layer"), devLog: pushLog(st.devLog, `${String(data.module ?? "layer")} ${String(data.layer)} completed at step ${String(data.step ?? st.currentStep)}`) })
        break
      case "layer.complete": {
        const d = data as unknown as LayerComplete
        const layerState = { norm: d.hidden_norm, mean: d.hidden_mean, std: d.hidden_std, step: d.step, timeMs: typeof (d as any).duration_ms === "number" ? Number((d as any).duration_ms) : 0, active: false, at: Date.now(), status: "completed" as const }
        const stepLayers = { ...(st.layersByStep[d.step] ?? {}), [d.layer]: layerState }
        set({
          layers: d.step === st.selectedStep ? stepLayers : st.layers,
          layersByStep: { ...st.layersByStep, [d.step]: stepLayers },
          timeline: pushTimeline(st.timeline, {
            type: "layer",
            label: `layer ${d.layer + 1} |norm|=${d.hidden_norm.toFixed(2)}`,
            step: d.step,
            layer: d.layer,
            detail: d.hidden_norm.toFixed(4),
            ts,
          }),
        })
        break
      }
      case "attention.captured": {
        const layers = data.layers as LayerLinks[]
        const links: Record<number, AttentionLink[]> = {}
        for (const l of layers) links[l.layer] = l.links
        const step = Number(data.step)
        set({ attentionLinks: step === st.selectedStep ? links : st.attentionLinks, attentionLinksByStep: { ...st.attentionLinksByStep, [step]: links } })
        break
      }
      case "qkv.captured": {
        const step = Number(data.step)
        const qkvStats = data.layers as QkvStat[]
        set({ qkvStats: step === st.selectedStep ? qkvStats : st.qkvStats, qkvStatsByStep: { ...st.qkvStatsByStep, [step]: qkvStats } })
        break
      }
      case "residual.captured":
        set({ residuals: { ...st.residuals, [Number(data.step)]: data.layers as ResidualSummary[] } })
        break
      case "kv_cache.captured":
        set({ kvCache: { ...st.kvCache, [Number(data.step)]: data as unknown as KvCacheSnapshot } })
        break
      case "logit_lens.captured":
        set({ logitLens: { ...st.logitLens, [Number(data.step)]: data.layers as { layer: number; candidates: Candidate[] }[] } })
        break
      case "mlp.captured": {
        const layers = data.layers as MlpTop[]
        const top: Record<number, MlpTop> = {}
        for (const l of layers) top[l.layer] = l
        const step = Number(data.step)
        set({ mlpTop: step === st.selectedStep ? top : st.mlpTop, mlpTopByStep: { ...st.mlpTopByStep, [step]: top } })
        break
      }
      case "logits.ready": {
        const step = data.step as number
        const candidates = data.candidates as Candidate[]
        const lastLogits = { step, temperature: data.temperature as number, top_p: data.top_p as number, top_k: data.top_k as number }
        set({ candidates: step === st.selectedStep ? candidates : st.candidates, candidatesByStep: { ...st.candidatesByStep, [step]: candidates }, lastLogits: step === st.selectedStep ? lastLogits : st.lastLogits, lastLogitsByStep: { ...st.lastLogitsByStep, [step]: lastLogits } })
        break
      }
      case "token.selected":
        if (typeof data.step === "number") set({ currentStep: data.step, selectedStep: data.step })
        break
      case "token.embedding": {
        const position = Number(data.position)
        const pca3 = data.pca3 as [number, number, number]
        const token = { position, id: Number(data.token_id), text: String(data.text), norm: Number(data.norm), pca3 }
        set({
          generatedTokens: st.generatedTokens.map((item) => item.position === position ? { ...item, pca3 } : item),
          pca: st.pca ? { ...st.pca, tokens: [...st.pca.tokens.filter((item) => item.position !== position), token] } : null,
        })
        break
      }
      case "token.generated": {
        const d = data as unknown as {
          step: number
          token_id: number
          text: string
          probability: number
          rank: number | null
          position?: number
          output: string
          time_ms: number
          entropy?: number | null
          sampling_method?: string | null
          embedding: { norm: number; pca3: [number, number, number] }
        }
        const tok: GenToken = {
          step: d.step,
          token_id: d.token_id,
          text: tokenLabel(d.text),
          probability: d.probability,
          rank: d.rank,
          position: typeof d.position === "number" ? d.position : st.tokens.length + d.step,
          time_ms: d.time_ms,
          entropy: d.entropy ?? undefined,
          sampling_method: d.sampling_method ?? undefined,
          pca3: d.embedding?.pca3 ?? null,
        }
        const stepLayers = st.layersByStep[d.step] ?? st.layers
        const stepLinks = st.attentionLinksByStep[d.step] ?? st.attentionLinks
        const stepQkv = st.qkvStatsByStep[d.step] ?? st.qkvStats
        const stepMlp = st.mlpTopByStep[d.step] ?? st.mlpTop
        const stepCandidates = st.candidatesByStep[d.step] ?? st.candidates
        const stepLogits = st.lastLogitsByStep[d.step] ?? st.lastLogits
        set({
          generatedTokens: [...st.generatedTokens, tok],
          telemetry: st.telemetry ? { ...st.telemetry, tokens: { ...st.telemetry.tokens, generated_tokens: st.generatedTokens.length + 1, sequence_tokens: st.tokens.length + st.generatedTokens.length + 1 } } : null,
          response: d.output,
          layers: stepLayers,
          attentionLinks: stepLinks,
          qkvStats: stepQkv,
          mlpTop: stepMlp,
          candidates: stepCandidates,
          lastLogits: stepLogits,
          currentStep: d.step,
          selectedStep: d.step,
          selectedToken: tok.position,
          devLog: pushLog(st.devLog, `generated [${d.step}] "${tokenLabel(d.text)}" p=${d.probability.toFixed(3)} ${d.time_ms}ms`),
          timeline: pushTimeline(st.timeline, {
            type: "token",
            label: `token ${d.step}: "${tokenLabel(d.text)}"`,
            step: d.step,
            detail: `${(d.probability * 100).toFixed(1)}%`,
            ts,
          }),
        })
        break
      }
      case "inference.complete":
        set({
          running: false,
          paused: false,
          runStatus: "COMPLETED",
          queuePosition: null,
          summary: data.summary as Summary,
          usage: (data.usage as ExternalUsage | null | undefined) ?? ((data.summary as Summary | undefined)?.usage ?? null),
          devLog: pushLog(st.devLog, `inference complete: ${String(data.num_output_tokens)} tokens, ${JSON.stringify((data.timings as Record<string, number>).tokens_per_second ?? 0)} tok/s`),
          timeline: pushTimeline(st.timeline, { type: "complete", label: "inference complete", ts }),
          activeLayer: null,
          activeModule: null,
          chatMessages: st.chatMessages.map((message, index, messages) => index === messages.map((item) => item.role).lastIndexOf("assistant") ? { ...message, content: String(data.response ?? message.content), status: "complete", mode: st.provider === "ollama" || st.provider === "qwen-local" ? "local" : "cloud", provider: st.provider ?? "qwen-local", model: st.providerModel ?? st.model?.model_id ?? null } : message),
        })
        break
      case "inference.cancelled":
        set({
          running: false,
          paused: false,
          runStatus: String(data.status ?? "CANCELLED"),
          queuePosition: null,
          summary: data.summary as Summary,
          devLog: pushLog(st.devLog, `inference cancelled after ${String(data.num_output_tokens)} tokens`, "warn"),
          timeline: pushTimeline(st.timeline, { type: "cancelled", label: "inference cancelled", ts }),
          chatMessages: st.chatMessages.map((message, index, messages) => index === messages.map((item) => item.role).lastIndexOf("assistant") ? { ...message, status: "cancelled" } : message),
        })
        break
      case "replay.loaded":
        set({ ...traceReset(), provider: String(data.provider ?? "qwen-local"), inspectionMode: data.inspection_mode === "limited" ? "limited" : "deep", model: (data.model_metadata as ModelMetadata | null) ?? null, providerModel: String(data.model_id ?? ""), replay: { ...st.replay, sessionId: data.session_id as string, playing: true, paused: false, status: "playing", index: Number(data.index ?? 0), count: Number(data.count ?? 0), speed: Number(data.speed ?? st.replay.speed) } })
        break
      case "replay.play":
        set({ replay: { ...st.replay, playing: data.status !== "done", paused: data.status === "paused", status: String(data.status), index: data.status === "done" ? Math.max(0, st.replay.count - 1) : st.replay.index, speed: data.speed == null ? st.replay.speed : Number(data.speed) } })
        break
      case "replay.seek":
        set({ replay: { ...st.replay, paused: true, playing: true, index: Number(data.index ?? 0), count: Number(data.count ?? st.replay.count), status: "paused" } })
        break
      case "dev.log":
        set({ devLog: pushLog(st.devLog, String(data.message), String(data.level ?? "info")) })
        break
    }
  },

  selectToken: (pos) => {
    const state = get()
    const generated = pos === null ? undefined : state.generatedTokens.find((token) => token.position === pos)
    const step = generated?.step ?? (pos !== null && state.tokens.some((token) => token.position === pos) ? 0 : null)
    set({
      selectedToken: pos,
      selectedStep: step,
      currentStep: step ?? state.currentStep,
      layers: step === null ? {} : state.layersByStep[step] ?? {},
      attentionLinks: step === null ? {} : state.attentionLinksByStep[step] ?? {},
      qkvStats: step === null ? [] : state.qkvStatsByStep[step] ?? [],
      mlpTop: step === null ? {} : state.mlpTopByStep[step] ?? {},
      candidates: step === null ? null : state.candidatesByStep[step] ?? null,
      lastLogits: step === null ? null : state.lastLogitsByStep[step] ?? null,
    })
  },
  selectStep: (step) => {
    const state = get()
    set({
      selectedStep: step,
      currentStep: step ?? state.currentStep,
      layers: step === null ? {} : state.layersByStep[step] ?? {},
      attentionLinks: step === null ? {} : state.attentionLinksByStep[step] ?? {},
      qkvStats: step === null ? [] : state.qkvStatsByStep[step] ?? [],
      mlpTop: step === null ? {} : state.mlpTopByStep[step] ?? {},
      candidates: step === null ? null : state.candidatesByStep[step] ?? null,
      lastLogits: step === null ? null : state.lastLogitsByStep[step] ?? null,
    })
  },
  selectLayer: (layer) => set({ selectedLayer: layer }),
  selectHead: (head) => set({ selectedHead: head }),
  selectNeuron: (layer, index) => set({ selectedNeuron: { layer, index }, selectedLayer: layer }),
  setInspector: (data) => set({ inspector: data }),
  setView: (v) => set({ view: v }),
  setTransformerViewMode: (mode) => set({ transformerViewMode: mode }),
  addChatMessage: (message) => set((state) => ({ chatMessages: [...state.chatMessages, message] })),
}))
