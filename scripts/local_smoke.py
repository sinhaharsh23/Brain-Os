from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

import websockets


BASE = "http://127.0.0.1:8765"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=20) as response:
        return response.status, json.loads(response.read().decode())


async def main() -> int:
    status, health = get("/api/health")
    assert status == 200 and health["status"] == "ok", health
    with urllib.request.urlopen(BASE + "/", timeout=20) as response:
        assert response.status == 200

    events = []
    async with websockets.connect("ws://127.0.0.1:8765/ws") as socket:
        await socket.send(json.dumps({
            "action": "run",
            "provider": "qwen-local",
            "prompt": "Explain AI in one short sentence.",
            "params": {"max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0, "top_k": 1},
        }))
        while True:
            event = json.loads(await socket.recv())
            events.append(event)
            if event.get("type") == "inference.complete":
                break

    event_types = {event["type"] for event in events}
    required = {"tokenization.complete", "embeddings.complete", "attention.captured", "qkv.captured", "mlp.captured", "logits.ready", "token.generated", "inference.complete"}
    missing = required - event_types
    assert not missing, sorted(missing)
    session_id = next(event for event in events if event["type"] == "inference.complete")["data"]["summary"]["session_id"]
    for path in (
        f"/api/sessions/{session_id}/embedding/0",
        f"/api/sessions/{session_id}/attention?layer=0&head=0&position=0",
        f"/api/sessions/{session_id}/qkv?layer=0&name=q&position=0",
        f"/api/sessions/{session_id}/mlp?layer=0&position=0&topk=5",
        f"/api/sessions/{session_id}/hidden?layer=0&position=0",
        f"/api/sessions/{session_id}/logits?step=0&k=5",
    ):
        response_status, _ = get(path)
        assert response_status == 200, path
    print(json.dumps({"status": "ok", "events": len(events), "session_id": session_id}))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
