from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from sklearn.decomposition import PCA

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def tensor_info(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}


def vector_stats(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().to(torch.float32)
    if t.numel() == 0:
        return {"n": 0}
    return {
        "n": int(t.numel()),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "l2_norm": float(t.norm()),
        "abs_mean": float(t.abs().mean()),
    }


def topk_activations(t: torch.Tensor, k: int = 16) -> list[dict[str, Any]]:
    t = t.detach().to(torch.float32)
    flat = t.view(-1)
    k = min(k, flat.numel())
    vals, idx = flat.topk(k)
    vals = vals.tolist()
    idx = idx.tolist()
    return [{"index": int(i), "value": float(v), "rank": int(rank)} for rank, (i, v) in enumerate(zip(idx, vals))]


def entropy_from_probabilities(probabilities: torch.Tensor) -> float:
    """Return Shannon entropy for an already normalized probability vector."""
    p = probabilities.detach().to(torch.float32)
    return float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum())


def sample_activations(t: torch.Tensor, limit: int = 128) -> list[dict[str, Any]]:
    t = t.detach().to(torch.float32)
    n = t.numel()
    if n <= limit:
        idx = torch.arange(n)
    else:
        stride = n / limit
        idx = torch.arange(limit) * stride
        idx = idx.to(torch.long)
    vals = t.view(-1)[idx]
    return [{"index": int(i), "value": float(v)} for i, v in zip(idx.tolist(), vals.tolist())]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).view(-1)
    b = b.detach().to(torch.float32).view(-1)
    norm_a = float(a.norm())
    norm_b = float(b.norm())
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(torch.dot(a, b) / (norm_a * norm_b))


def residual_metrics(x0: torch.Tensor, attn_out: torch.Tensor, mlp_out: torch.Tensor, x2: torch.Tensor) -> dict[str, Any]:
    x0_f = x0.detach().to(torch.float32).view(-1)
    attn_f = attn_out.detach().to(torch.float32).view(-1)
    mlp_f = mlp_out.detach().to(torch.float32).view(-1)
    x2_f = x2.detach().to(torch.float32).view(-1)
    post_attn = x0_f + attn_f

    norm_x0 = float(x0_f.norm())
    norm_attn = float(attn_f.norm())
    norm_post_attn = float(post_attn.norm())
    norm_mlp = float(mlp_f.norm())
    norm_x2 = float(x2_f.norm())

    total_delta = norm_attn + norm_mlp + 1e-9
    attn_ratio = float(norm_attn / total_delta)
    mlp_ratio = float(norm_mlp / total_delta)

    return {
        "input_norm": norm_x0,
        "attn_delta_norm": norm_attn,
        "post_attn_norm": norm_post_attn,
        "mlp_delta_norm": norm_mlp,
        "output_norm": norm_x2,
        "cosine_similarity": cosine_sim(x0_f, x2_f),
        "attn_ratio": round(attn_ratio, 4),
        "mlp_ratio": round(mlp_ratio, 4),
    }


def qkv_head_slice(tensor: torch.Tensor, head: int, head_dim: int, num_heads: int) -> torch.Tensor:
    t = tensor.detach().to(torch.float32)
    flat = t.view(-1)
    head_idx = max(0, min(head, num_heads - 1))
    start = head_idx * head_dim
    end = start + head_dim
    if end <= flat.numel():
        return flat[start:end]
    return flat


def pca_projection(embeddings: np.ndarray, dims: int = 3) -> dict[str, Any]:
    n = embeddings.shape[0]
    if n < 2:
        coords = np.zeros((n, dims), dtype=np.float64)
        explained = [0.0] * dims
        mean = np.zeros(embeddings.shape[1], dtype=np.float64)
        components = np.zeros((embeddings.shape[1], dims), dtype=np.float64)
    elif not HAVE_SKLEARN:
        coords = _svd_projection(embeddings, dims)
        explained = [0.0] * dims
        mean = embeddings.mean(axis=0)
        u, s, vt = np.linalg.svd(embeddings - mean, full_matrices=False)
        components = vt[:dims].T
    else:
        n_components = min(dims, n - 1, embeddings.shape[1])
        # The sklearn randomized solver emits overflow/NaN warnings for the
        # small, highly rectangular token batches produced by this model even
        # when the embeddings themselves are finite. Full SVD is stable for
        # these observability-sized batches and preserves the real projection.
        pca = PCA(n_components=n_components, svd_solver="full")
        coords_full = pca.fit_transform(embeddings)
        coords = np.zeros((n, dims), dtype=np.float64)
        coords[:, :n_components] = coords_full
        explained = list(pca.explained_variance_ratio_) + [0.0] * (dims - n_components)
        mean = pca.mean_
        components = np.zeros((embeddings.shape[1], dims), dtype=np.float64)
        components[:, :n_components] = pca.components_.T
    return {
        "coords": [[float(x) for x in row] for row in coords],
        "explained_variance": [float(e) for e in explained][:dims],
        "method": "sklearn.PCA" if HAVE_SKLEARN else "svd",
        "mean": mean.tolist(),
        "components": [[float(x) for x in col] for col in components],
    }


def pca_project_vector(vector: np.ndarray, pca: dict[str, Any]) -> list[float]:
    if not pca.get("coords"):
        return [0.0, 0.0, 0.0]
    mean = np.asarray(pca.get("mean", []), dtype=np.float64)
    comps = np.asarray(pca.get("components", []), dtype=np.float64)
    if mean.size and comps.size:
        return [float((vector - mean) @ comps[:, i]) for i in range(comps.shape[1])]
    return [0.0, 0.0, 0.0]


def _svd_projection(embeddings: np.ndarray, dims: int) -> np.ndarray:
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :dims] * s[:dims]
    return coords


def normalize_attention_rows(w: torch.Tensor) -> torch.Tensor:
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)


def attention_topk(weights: torch.Tensor, k: int = 12) -> list[dict[str, Any]]:
    w = weights.detach().to(torch.float32)
    vals, idx = w.topk(min(k, w.numel()))
    return [
        {"token_index": int(i), "weight": float(v)}
        for v, i in zip(vals.tolist(), idx.tolist())
        if float(v) > 0.0
    ]


def quantile_scale(values: list[float], q: float = 0.98) -> float:
    if not values:
        return 1.0
    arr = np.array(values, dtype=np.float64)
    m = np.quantile(arr, q)
    return float(m) if m > 1e-6 else 1.0


def human_bytes(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
