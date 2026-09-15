from app.models.qwen import HuggingFaceCausalAdapter


class LlamaAdapter(HuggingFaceCausalAdapter):
    name = "llama"
    family = "llama"


class MistralAdapter(HuggingFaceCausalAdapter):
    name = "mistral"
    family = "mistral"


class GemmaAdapter(HuggingFaceCausalAdapter):
    name = "gemma"
    family = "gemma"
