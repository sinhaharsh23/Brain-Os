from app.providers.base import ProviderCapabilities, ProviderDescriptor

DEEP = ProviderCapabilities(True, True, True, True, True, True, True, True, True)
LIMITED = ProviderCapabilities(True, False, False, False, False, False, False, False, False)

PROVIDERS = (
    ProviderDescriptor(
        provider_id="qwen-local",
        display_name="Qwen local",
        kind="local",
        inspection_mode="deep",
        availability="available",
        limitation="Inspectable PyTorch model; tensor fields originate from the local model.",
        capabilities=DEEP,
        models=("Qwen/Qwen2.5-0.5B-Instruct",),
    ),
    ProviderDescriptor(
        provider_id="openai",
        display_name="OpenAI / ChatGPT",
        kind="external",
        inspection_mode="limited",
        availability="adapter-not-configured",
        limitation="Normal hosted APIs do not expose hidden states, attention, Q/K/V, or MLP activations.",
        capabilities=LIMITED,
        models=(),
    ),
    ProviderDescriptor(
        provider_id="anthropic",
        display_name="Anthropic / Claude",
        kind="external",
        inspection_mode="limited",
        availability="adapter-not-configured",
        limitation="Normal hosted APIs do not expose private Transformer internals.",
        capabilities=LIMITED,
        models=(),
    ),
    ProviderDescriptor(
        provider_id="google",
        display_name="Google / Gemini",
        kind="external",
        inspection_mode="limited",
        availability="adapter-not-configured",
        limitation="Only provider-exposed stream metadata can be shown; internals are unavailable.",
        capabilities=LIMITED,
        models=(),
    ),
)
