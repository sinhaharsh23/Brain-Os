# Model Guide

## Qwen/Qwen2.5-0.5B-Instruct

This is the default adapter because it fits the available CPU-only machine and has a directly inspectable Qwen2 architecture.

- Parameters: approximately 494M
- Layers: 24
- Hidden size: 896
- Attention heads: 14
- Key/value heads: 2
- Head dimension: 64
- MLP dimension: 4864
- Vocabulary: 151,936
- Attention mode: eager, required for returned attention weights
- Generation: manual token loop with the Transformers KV cache

The adapter uses the model's chat template by default because this is an instruction-tuned checkpoint. Template tokens are shown as special tokens in the UI. Set `use_chat_template` to `false` only when studying raw text continuation.

## Capture Policy

The first forward pass captures the full prompt attention matrix and prompt-position projection/activation rows. Subsequent cached forwards capture the generated token's attention row and projection/activation vectors. This provides exact token-level data without recomputing the entire prompt for every generated token.

The browser receives compact summaries during streaming. Full vectors and attention rows are available from the session inspection endpoints.

## Limitations

- UMAP is not included; PCA is deterministic and has no additional dependency or stochastic fitting behavior.
- GPU utilization is reported when NVML is available. CUDA memory reporting remains available without NVML.
- The current model adapter is Qwen2-specific even though the application interfaces are model-independent.
