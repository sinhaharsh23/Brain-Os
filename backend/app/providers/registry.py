import os

from app.config import settings
from app.providers.base import ProviderCapabilities, ProviderDescriptor

DEEP = ProviderCapabilities(True, True, True, True, True, True, True, True, True)
LIMITED = ProviderCapabilities(True, False, False, False, False, False, False, False, False)


def _configured(*names: str) -> bool:
    return any(os.getenv(name) or getattr(settings, name.lower(), "") for name in names)

PROVIDER_TEMPLATES = (
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
        availability="configured" if _configured("OPENAI_API_KEY") else "missing-api-key",
        limitation="Normal hosted APIs do not expose hidden states, attention, Q/K/V, or MLP activations.",
        capabilities=LIMITED,
        models=("gpt-4o-mini",),
    ),
    ProviderDescriptor(
        provider_id="anthropic",
        display_name="Anthropic / Claude",
        kind="external",
        inspection_mode="limited",
        availability="configured" if _configured("ANTHROPIC_API_KEY") else "missing-api-key",
        limitation="Normal hosted APIs do not expose private Transformer internals.",
        capabilities=LIMITED,
        models=("claude-3-5-haiku-latest",),
    ),
    ProviderDescriptor(
        provider_id="google",
        display_name="Google / Gemini",
        kind="external",
        inspection_mode="limited",
        availability="configured" if _configured("GOOGLE_API_KEY", "GEMINI_API_KEY") else "missing-api-key",
        limitation="Only provider-exposed stream metadata can be shown; internals are unavailable.",
        capabilities=LIMITED,
        models=("gemini-2.0-flash",),
    ),
)

PROVIDERS = PROVIDER_TEMPLATES
