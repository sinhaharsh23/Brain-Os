# BrainOS local models

Place downloaded Hugging Face checkpoints in this directory. The default local
model is expected at:

```text
models/Qwen2.5-0.5B-Instruct
```

From the repository root, install it once while online with:

```bash
./scripts/download-local-model.sh
```

Model weights and generated cache files are ignored by Git. Once the checkpoint
is complete, BrainOS resolves and loads it with `local_files_only=True`; startup
does not require Hugging Face DNS or network access.
