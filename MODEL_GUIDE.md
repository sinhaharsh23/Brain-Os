# Model Guide

## Qwen/Qwen2.5-0.5B-Instruct

This is the default adapter because it fits the current MacBook Pro with Apple M5 Pro unified memory, runs through Apple MPS with CPU fallback, and has a directly inspectable Qwen2 architecture.

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

The 3D view renders only captured top-K MLP units per layer. Each unit's index, sign, magnitude, and selected exact value come from the model capture; the visualization does not claim to render all units or invent neuron-to-neuron edges.

Completed sessions persist captured tensors in local `.pt` bundles alongside replay JSON. Bundles use tensor-only deserialization and are restored on demand when an archived session is inspected after restart.

## Limitations

- UMAP is not included; PCA is deterministic and has no additional dependency or stochastic fitting behavior.
- Apple MPS memory reporting uses PyTorch's MPS allocator when available; NVIDIA CUDA utilization uses NVML when installed. CPU fallback remains supported when no accelerator is available.
- The shared Hugging Face adapter detects the loaded architecture's actual embedding, attention, Q/K/V, MLP, hidden-state, logits, and probability support. The UI exposes those capabilities from model metadata instead of assuming every architecture has identical modules.

## External Observation Adapters

OpenAI and Anthropic adapters stream provider-returned text through the backend when `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is configured. Gemini uses its provider response endpoint with `GOOGLE_API_KEY` or `GEMINI_API_KEY`. These adapters intentionally expose only provider-returned text and timing. They do not create token IDs, embeddings, attention, Q/K/V, logits, or activations.

## Additional Local Adapters

The local adapter registry includes real Qwen2, Llama-compatible, Mistral, and Gemma classes. They use Hugging Face `AutoModelForCausalLM` with eager attention and inspect the loaded checkpoint's module layout. On the current physical Apple M5 Pro, Qwen/Qwen2.5-0.5B-Instruct is verified for MPS inference and tensor capture. TinyLlama/TinyLlama-1.1B-Chat-v1.0, Qwen 1.5B/3B/7B, Mistral 7B, and Gemma 2 2B remain implemented but `IMPLEMENTED_NOT_VERIFIED` on this machine. NVIDIA CUDA and AMD ROCm remain compatibility targets and are not marked verified here. Access-controlled checkpoints also require the user's Hugging Face authorization.
