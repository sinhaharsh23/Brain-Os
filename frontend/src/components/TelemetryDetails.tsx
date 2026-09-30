import { Fragment } from "react"
import { useBrain } from "../store/useBrainStore"

export default function TelemetryDetails() {
  const telemetry = useBrain((s) => s.telemetry)
  const provider = useBrain((s) => s.provider)
  const nativeModel = useBrain((s) => s.provider === "qwen-local" ? s.model : null)
  const providerModel = useBrain((s) => s.providerModel)
  const prompt = useBrain((s) => s.prompt)
  const template = useBrain((s) => s.chatTemplate)
  const logs = useBrain((s) => s.providerLogprobs)
  const t = telemetry
  const rows: [string, unknown][] = [
    ["PROVIDER / RUNTIME", t?.provider.name ?? provider], ["MODEL", t?.model.id ?? providerModel ?? nativeModel?.model_id],
    ["USER TEXT TOKENS", t?.tokens.user_text_tokens], ["MODEL INPUT TOKENS", t?.tokens.model_input_tokens],
    ["TEMPLATE / HISTORY OVERHEAD", t?.tokens.template_history_overhead], ["GENERATED MODEL TOKENS", t?.tokens.generated_tokens],
    ["SPECIAL OUTPUT TOKENS", t?.tokens.special_output_tokens], ["SEQUENCE TOKENS", t?.tokens.sequence_tokens],
    ["PROMPT TOKENS FROM CACHE", t?.tokens.cached_prompt_tokens],
    ["ARCHITECTURE", t?.model.architecture ?? nativeModel?.architecture], ["PARAMETERS", t?.model.parameter_count ?? nativeModel?.num_params],
    ["CONTEXT CAPACITY", t?.model.context_window ?? nativeModel?.context_length],
    ["LAYERS / HIDDEN", t?.model.num_layers != null ? `${t.model.num_layers} / ${t.model.hidden_size ?? 'Unavailable'}` : nativeModel ? `${nativeModel.num_layers} / ${nativeModel.hidden_size}` : null],
    ["CONFIG DTYPE", t?.model.config_dtype ?? nativeModel?.extra.config_dtype], ["RUNTIME DTYPE", t?.model.runtime_dtype ?? nativeModel?.dtype],
    ["MODEL DEVICE", t?.model.device ?? nativeModel?.device], ["QUANTIZATION", t?.model.quantization ?? nativeModel?.quantization],
    ["END-TO-END LATENCY (ms)", t?.timing.total_ms], ["TTFT (ms)", t?.timing.ttft_ms],
    ["PROVIDER TOTAL (ms)", t?.timing.provider_total_ms], ["LOAD (ms)", t?.timing.load_ms],
    ["PREFILL (ms)", t?.timing.prompt_eval_ms], ["DECODE (ms)", t?.timing.decode_ms],
    ["DECODE TOK/S", t?.performance.decode_tokens_per_second], ["END-TO-END TOK/S", t?.performance.end_to_end_tokens_per_second],
  ]
  const format = (value: unknown) => value == null ? "Unavailable" : typeof value === "number" ? Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2) : String(value)
  return <>
    <div className="kv" data-testid="normalized-telemetry">
      {rows.map(([label, value]) => <Fragment key={label}><span className="k" title={label === "MODEL INPUT TOKENS" ? "Model Input Tokens include system instructions, conversation history, chat-template markers and the current user message." : undefined}>{label}</span><span className="v">{format(value)}</span></Fragment>)}
    </div>
    <details><summary>MODEL INPUT PREVIEW</summary><p>Raw user input: {prompt || "Unavailable"}</p><pre style={{ whiteSpace: "pre-wrap" }}>{template?.serialized || (provider === "ollama" ? "Run inference to capture the actual GGUF chat template and tokenizer output." : "Run inference to capture the actual formatted input.")}</pre></details>
    <details><summary>ACTIVE SAMPLING / METRIC SOURCES</summary><pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify({ sampling: t?.sampling ?? null, sources: t?.sources ?? null, capabilities: t?.capabilities ?? null, tensors: t?.tensors ?? null }, null, 2)}</pre></details>
    {logs.length > 0 && <div data-testid="provider-logprobs"><div className="panel-title">Selected token probability · Ollama API</div>{logs.slice(-8).map((token, index) => <div key={index}>{JSON.stringify(token.token)} · {(Math.exp(token.logprob) * 100).toFixed(2)}% · log p {token.logprob.toFixed(4)}{token.top_logprobs?.map((alternative, i) => <span key={i}> · {JSON.stringify(alternative.token)} {(Math.exp(alternative.logprob) * 100).toFixed(2)}%</span>)}</div>)}</div>}
  </>
}
