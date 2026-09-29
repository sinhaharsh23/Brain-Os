from pathlib import Path

import pytest

from app.models.resolver import (
    LocalModelIncompleteError,
    LocalModelNotInstalledError,
    LocalModelResolver,
)


def _checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"local-test-placeholder")
    return root


def test_explicit_checkpoint_resolves_without_hub_access(tmp_path):
    checkpoint = _checkpoint(tmp_path / "installed")
    resolved = LocalModelResolver(
        project_root=tmp_path,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_path=str(checkpoint),
        offline=True,
        allow_download=False,
    ).resolve_model()

    assert resolved.path == checkpoint.resolve()
    assert resolved.source == "configured_path"
    assert resolved.network_required is False


def test_repository_local_checkpoint_wins_over_cache(tmp_path):
    checkpoint = _checkpoint(tmp_path / "models" / "Qwen2.5-0.5B-Instruct")
    resolved = LocalModelResolver(
        project_root=tmp_path,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hf_home=tmp_path / "hf",
        offline=True,
        allow_download=False,
    ).resolve_model()

    assert resolved.path == checkpoint.resolve()
    assert resolved.source == "project_local"


def test_hf_snapshot_cache_is_considered_without_network(tmp_path):
    checkpoint = _checkpoint(
        tmp_path
        / "hf"
        / "hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots"
        / "local-revision"
    )
    resolved = LocalModelResolver(
        project_root=tmp_path,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hf_home=tmp_path / "hf",
        offline=True,
        allow_download=False,
    ).resolve_model()

    assert resolved.path == checkpoint.resolve()
    assert resolved.source == "hf_cache"


def test_missing_checkpoint_is_actionable_and_does_not_download(tmp_path):
    with pytest.raises(LocalModelNotInstalledError, match="LOCAL_MODEL_NOT_INSTALLED"):
        LocalModelResolver(
            project_root=tmp_path,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            hf_home=tmp_path / "hf",
            offline=True,
            allow_download=False,
        ).resolve_model()


def test_existing_incomplete_checkpoint_is_not_mistaken_for_network_failure(tmp_path):
    incomplete = tmp_path / "models" / "Qwen2.5-0.5B-Instruct"
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text("{}")

    with pytest.raises(LocalModelIncompleteError, match="LOCAL_MODEL_INCOMPLETE"):
        LocalModelResolver(
            project_root=tmp_path,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            hf_home=tmp_path / "hf",
            offline=True,
            allow_download=False,
        ).resolve_model()
