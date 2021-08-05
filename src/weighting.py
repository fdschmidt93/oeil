from typing import Optional, Tuple

import numpy as np


def z_norm(
    x: np.ndarray,
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    return_stats: bool = False,
) -> np.ndarray:
    if stats is None:
        mean = x.mean(0, keepdims=True)
        std = x.std(0, keepdims=True)
    else:
        mean, std = stats
    if return_stats:
        return np.nan_to_num((x - mean) / std), (mean, std)
    else:
        return np.nan_to_num((x - mean) / std)


def tf_idf(
    x: np.ndarray, idf: Optional[np.ndarray] = None, return_idf: bool = False
) -> np.ndarray:
    N = x.shape[0]
    tf = (1 + np.maximum(np.log(x), 0)) / (1 + np.log(np.max(x, 1, keepdims=True)))
    if idf is None:
        df = np.count_nonzero(x, 0)
        idf = np.log(N / df)
        df_zeros = df == 0
        idf[df_zeros] = 0
    tfidf = tf * idf
    if return_idf:
        return tfidf, idf
    else:
        return tfidf


def sif_weighting(counts: np.ndarray, embeddings: np.ndarray, a=0.003) -> np.ndarray:
    total = counts.sum()
    p_w = counts / total
    sif_weights = a / (a + p_w)
    return embeddings * sif_weights[:, None]


def compute_pc(sent_emb, pc: int = -1) -> np.ndarray:
    u, _, _ = np.linalg.svd(sent_emb.T, full_matrices=False)
    u = u[:, :pc]
    return u @ u.T


def pc_removal(sent_emb: np.ndarray, pc: np.ndarray) -> np.ndarray:
    return sent_emb - sent_emb @ pc


def reduce_fn(x: np.ndarray, fun: str) -> np.ndarray:
    if fun == "mean":
        return x.mean(0)
    elif fun == "min":
        return x.min(0)
    elif fun == "max":
        return x.max(0)
    else:
        raise NotImplementedError("Invalid reduction function")


def aggregator(x: np.ndarray, fun=["mean"]):
    if isinstance(fun, str):
        return reduce_fn(x, fun)
    elif isinstance(fun, list):
        return np.hstack([reduce_fn(x, f) for f in fun])


def sif(
    inputs: np.ndarray,
    emb: np.ndarray,
    reduction=["mean"],
    pc: Optional[np.ndarray] = None,
    remove_pc: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    sent_emb = np.stack([aggregator(emb[x], reduction) for x in inputs])
    if remove_pc or pc is not None:
        pc = compute_pc(sent_emb, remove_pc) if pc is None else pc
        sent_emb = pc_removal(sent_emb, pc)
        return sent_emb, pc
    else:
        return sent_emb, None


def min_max_norm(x: np.ndarray) -> np.ndarray:
    min_ = x.min(keepdims=True)
    max_ = x.max(keepdims=True)
    return (x - min_) / (max_ - min_)


def average_cos_sim(inputs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    l2norm = lambda x: x / np.linalg.norm(x, axis=-1, ord=2, keepdims=True)
    inputs_ = l2norm(inputs)
    ref_ = l2norm(ref)
    cos_sim = (inputs_ @ ref_.T).mean(1)
    return cos_sim
