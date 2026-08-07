import numpy as np
import torch

from app.instrumentation.stats import pca_project_vector, pca_projection, topk_activations, vector_stats


def test_embeddings_shape_and_stats(adapter):
    tokens, ids, _ = adapter.tokenize("Explain artificial intelligence")
    emb = adapter.embed(ids)
    assert tuple(emb.shape) == (1, len(tokens), adapter.metadata.hidden_size)
    vec = emb[0, 0]
    stats = vector_stats(vec)
    assert stats["n"] == adapter.metadata.hidden_size
    assert stats["l2_norm"] > 0


def test_pca_projection(adapter):
    tokens, ids, _ = adapter.tokenize("cat dog bird fish tree sun moon stars")
    emb = adapter.embed(ids).squeeze(0)
    pca = pca_projection(emb.numpy(), dims=3)
    assert len(pca["coords"]) == len(tokens)
    assert all(len(c) == 3 for c in pca["coords"])
    assert len(pca["explained_variance"]) == 3
    assert 0 <= pca["explained_variance"][0] <= 1


def test_pca_projects_new_vector(adapter):
    tokens, ids, _ = adapter.tokenize("cat dog bird fish")
    emb = adapter.embed(ids).squeeze(0)
    pca = pca_projection(emb.numpy(), dims=3)
    new = adapter.embed_token(adapter.metadata.vocab_size - 1)
    coords = pca_project_vector(new.numpy(), pca)
    assert len(coords) == 3
    assert all(np.isfinite(c) for c in coords)


def test_topk_activations(adapter):
    t = torch.randn(4864)
    top = topk_activations(t, k=10)
    assert len(top) == 10
    values = [x["value"] for x in top]
    assert values == sorted(values, reverse=True)
    assert all(0 <= x["rank"] < 10 for x in top)
