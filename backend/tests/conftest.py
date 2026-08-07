import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("BRAINOS_HF_HOME", os.environ.get("HF_HOME", "/home/harsh-sinha/BrainOS/models/hf"))

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def model_id() -> str:
    return MODEL_ID


@pytest.fixture(scope="session")
def adapter(model_id):
    from app.models.qwen import QwenAdapter

    a = QwenAdapter(model_id=model_id, device="cpu", dtype="float32")
    a.load()
    yield a
    a.unload()


@pytest.fixture(scope="module")
def engine(adapter):
    from app.events.bus import EventBus
    from app.inference.engine import InferenceEngine

    e = InferenceEngine(adapter, EventBus(), max_prompt_tokens=256)
    e._register_hooks()
    return e
