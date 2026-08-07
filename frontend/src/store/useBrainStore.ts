import { create } from "zustand"
import type {
  AttentionLink,
  Candidate,
  DevEntry,
  EmbeddingsComplete,
  GenToken,
  HardwareReport,
  LayerLinks,
  LayerComplete,
  MlpTop,
  ModelMetadata,
  MonitorSnapshot,
  QkvStat,
  ReplayState,
  SessionSummary,
  Summary,
  TimelineEntry,
  TokenInfo,
} from "../types"

export type ViewMode = "brain" | "architecture" | "attention" | "embedding" | "tokenflow" | "dev"

export interface LayerState {
  norm: number
  mean: number
  std: number
  step: number
  timeMs: number
  active: boolean
  at: number
}

export interface InspectorData {
  kind: "token" | "layer" | "attention" | "neuron" | "embedding" | "qkv" | "logits"
  title: string
  rows: { label: string; value: string }[]
  table?: { headers: string[]; rows: (string | number)[][] }
}

export interface BrainState {
  connected: boolean
  modelStatus: string
  model: ModelMetadata | null
  hardware: HardwareReport | null
  sessionId: string | null
  running: boolean
  currentStep: number
  prompt: string
  tokens: TokenInfo[]
  generatedTokens: GenToken[]
  response: string
  layers: Record<number, LayerState>
  attentionLinks: Record<number, AttentionLink[]>
  qkvStats: QkvStat[]
  mlpTop: Record<number, MlpTop>
  candidates: Candidate[] | null
  lastLogits: { step: number; temperature: number; top_p: number; top_k: number } | null
  pca: EmbeddingsComplete | null
  monitoring: MonitorSnapshot | null
  devLog: DevEntry[]
  timeline: TimelineEntry[]
  sessions: SessionSummary[]
  replay: ReplayState
  inspector: InspectorData | null
  selectedToken: number | null
  selectedLayer: number | null
  selectedHead: number | null
  view: ViewMode
  summary: Summary | null
  inferenceError: string | null

  dispatch: (ev: { type: string; data: Record<string, unknown>; ts: number; session_id: string | null }) => void
  set: (patch: Partial<BrainState>) => void
  run: (prompt: string, params: Record<string, unknown>) => void
  cancel: () => void
  replaySession: (sessionId: string, speed: number) => void
  replayPause: (paused: boolean) => void
  replayStop: () => void
  selectToken: (pos: number | null) => void
  selectLayer: (layer: number | null) => void
  selectHead: (head: number | null) => void
  setInspector: (data: InspectorData | null) => void
  setView: (v: ViewMode) => void
}

function tokenLabel(text: string): string {
  return text.replace(/Ġ/g, " ").replace(/Ċ/g, "\n").replace(/\n/g, "\\n")
}

function pushLog(entries: DevEntry[], message: string, level = "info", ts = Date.now()): DevEntry[] {
  return [...entries.slice(-499), { level, message, ts }]
}

function pushTimeline(entries: TimelineEntry[], e: TimelineEntry): TimelineEntry[] {
  return [...entries.slice(-499), e]
}

export const useBrain = create<BrainState>((set, get) => ({
  connected: false,
  modelStatus: "not_loaded",
  model: null,
  hardware: null,
  sessionId: null,
  running: false,
  currentStep: 0,
  prompt: "",
  tokens: [],
  generatedTokens: [],
  response: "",
  layers: {},
  attentionLinks: {},
  qkvStats: [],
  mlpTop: {},
  candidates: null,
  lastLogits: null,
  pca: null,
  monitoring: null,
  devLog: [],
  timeline: [],
  sessions: [],
  replay: { sessionId: null, playing: false, paused: false, speed: 1, status: "idle", index: 0, count: 0 },
  inspector: null,
  selectedToken: null,
  selectedLayer: null,
  selectedHead: null,
  view: "brain",
  summary: null,
  inferenceError: null,

  set: (patch) => set(patch),

  dispatch: (ev) => {
    const { data, ts } = ev
    const st = get()

    switch (ev.type) {
      case "system.ready":
        set({ devLog: pushLog(st.devLog, `connected to backend (${(data as { ws_url: string }).ws_url})`) })
        break
      case "system.error":
        set({
          devLog: pushLog(st.devLog, `ERROR: ${String(data.message)}`, "error"),
          inferenceError: String(data.message),
        })
        break
      case "model.ready":
        set({ model: data.metadata as ModelMetadata, modelStatus: "loaded" })
        break
      case "monitoring.tick":
        set({ monitoring: data as unknown as MonitorSnapshot })
        break
      case "inference.started":
        set({
          running: true,
          inferenceError: null,
          sessionId: (ev.session_id as string) ?? null,
          prompt: String(data.prompt ?? ""),
          tokens: [],
          generatedTokens: [],
          response: "",
          layers: {},
          attentionLinks: {},
          qkvStats: [],
          mlpTop: {},
          candidates: null,
          pca: null,
          summary: null,
          timeline: [],
          selectedToken: null,
          devLog: pushLog(st.devLog, `inference started: "${String(data.prompt).slice(0, 80)}"`),
        })
        break
      case "tokenization.complete": {
        const tokens = data.tokens as TokenInfo[]
        set({
          tokens,
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
          candidates: null,
          timeline: pushTimeline(st.timeline, {
            type: "step",
            label: `step ${String(data.step)}: ${String(data.new_token ?? "(prompt)")}`,
            step: data.step as number,
            ts,
          }),
        })
        break
      case "layer.complete": {
        const d = data as unknown as LayerComplete
        set({
          layers: {
            ...st.layers,
            [d.layer]: { norm: d.hidden_norm, mean: d.hidden_mean, std: d.hidden_std, step: d.step, timeMs: 0, active: true, at: Date.now() },
          },
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
        set({ attentionLinks: links })
        break
      }
      case "qkv.captured":
        set({ qkvStats: data.layers as QkvStat[] })
        break
      case "mlp.captured": {
        const layers = data.layers as MlpTop[]
        const top: Record<number, MlpTop> = {}
        for (const l of layers) top[l.layer] = l
        set({ mlpTop: top })
        break
      }
      case "logits.ready":
        set({
          candidates: data.candidates as Candidate[],
          lastLogits: { step: data.step as number, temperature: data.temperature as number, top_p: data.top_p as number, top_k: data.top_k as number },
        })
        break
      case "token.selected":
        break
      case "token.generated": {
        const d = data as unknown as {
          step: number
          token_id: number
          text: string
          probability: number
          rank: number | null
          output: string
          time_ms: number
          embedding: { norm: number; pca3: [number, number, number] }
        }
        const tok: GenToken = {
          step: d.step,
          token_id: d.token_id,
          text: tokenLabel(d.text),
          probability: d.probability,
          rank: d.rank,
          position: st.tokens.length + (d.step - 1),
          time_ms: d.time_ms,
          pca3: d.embedding?.pca3 ?? null,
        }
        set({
          generatedTokens: [...st.generatedTokens, tok],
          response: d.output,
          currentStep: d.step,
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
          summary: data.summary as Summary,
          devLog: pushLog(st.devLog, `inference complete: ${String(data.num_output_tokens)} tokens, ${JSON.stringify((data.timings as Record<string, number>).tokens_per_second ?? 0)} tok/s`),
          timeline: pushTimeline(st.timeline, { type: "complete", label: "inference complete", ts }),
        })
        break
      case "replay.loaded":
        set({ replay: { ...st.replay, sessionId: data.session_id as string, playing: true, paused: false, status: "playing", index: Number(data.index ?? 0), count: Number(data.count ?? 0) } })
        break
      case "replay.play":
        set({ replay: { ...st.replay, playing: data.status !== "done", paused: data.status === "paused", status: String(data.status) } })
        break
      case "replay.seek":
        set({ replay: { ...st.replay, paused: true, playing: true, index: Number(data.index ?? 0), count: Number(data.count ?? st.replay.count), status: "paused" } })
        break
      case "dev.log":
        set({ devLog: pushLog(st.devLog, String(data.message), String(data.level ?? "info")) })
        break
    }
  },

  run: (_prompt, _params) => {
    set({ running: true, inferenceError: null })
  },
  cancel: () => {},
  replaySession: () => {},
  replayPause: () => {},
  replayStop: () => {},
  selectToken: (pos) => set({ selectedToken: pos }),
  selectLayer: (layer) => set({ selectedLayer: layer }),
  selectHead: (head) => set({ selectedHead: head }),
  setInspector: (data) => set({ inspector: data }),
  setView: (v) => set({ view: v }),
}))
