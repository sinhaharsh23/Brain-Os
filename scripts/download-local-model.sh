#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${BRAINOS_PYTHON:-$ROOT_DIR/backend/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: BrainOS Python environment not found at $PYTHON" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/models/hf}"
MODEL_CONFIG="$(cd "$ROOT_DIR" && PYTHONPATH="$ROOT_DIR/backend" "$PYTHON" -c 'from app.config import settings; print(settings.local_model_id or settings.default_model); print(settings.local_model_path)')"
MODEL_ID="$(printf '%s\n' "$MODEL_CONFIG" | sed -n '1p')"
CONFIGURED_PATH="$(printf '%s\n' "$MODEL_CONFIG" | sed -n '2p')"
MODEL_NAME="${MODEL_ID##*/}"
TARGET_DIR="${LOCAL_MODEL_PATH:-${CONFIGURED_PATH:-$ROOT_DIR/models/$MODEL_NAME}}"
case "$TARGET_DIR" in
    /*) ;;
    *) TARGET_DIR="$ROOT_DIR/$TARGET_DIR" ;;
esac

echo "BrainOS local model installer"
echo "Model: $MODEL_ID"
echo "Target: $TARGET_DIR"

if cd "$ROOT_DIR" && PYTHONPATH="$ROOT_DIR/backend" "$PYTHON" - "$MODEL_ID" <<'PY'
import sys
from app.models.resolver import LocalModelResolver

model_id = sys.argv[1]
resolver = LocalModelResolver(model_id=model_id, allow_download=False, offline=True)
try:
    resolved = resolver.resolve_model()
except Exception:
    raise SystemExit(1)
print(f"Model already installed: {resolved.path}")
PY
then
    exit 0
fi

mkdir -p "$TARGET_DIR"
if command -v hf >/dev/null 2>&1; then
    echo "Downloading with hf CLI (the model will not be downloaded again once complete)..."
    hf download "$MODEL_ID" --type model --local-dir "$TARGET_DIR" --cache-dir "$HF_HOME/hub" --format quiet
else
    echo "hf CLI not found; using the installed huggingface_hub package..."
    cd "$ROOT_DIR" && PYTHONPATH="$ROOT_DIR/backend" "$PYTHON" - "$MODEL_ID" "$TARGET_DIR" "$HF_HOME/hub" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, target, cache_dir = sys.argv[1:]
snapshot_download(repo_id=model_id, local_dir=target, cache_dir=cache_dir)
PY
fi

cd "$ROOT_DIR" && PYTHONPATH="$ROOT_DIR/backend" "$PYTHON" - "$TARGET_DIR" "$MODEL_ID" <<'PY'
import sys
from app.models.resolver import LocalModelResolver, LocalModelIncompleteError

path, model_id = sys.argv[1:]
resolver = LocalModelResolver(model_id=model_id, model_path=path, allow_download=False, offline=True)
try:
    resolved = resolver.resolve_model()
except LocalModelIncompleteError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1)
print(f"Model installed successfully: {resolved.path}")
PY
