"""Phase 3 — Embedding Retrieval. Pure vector math, identical to the notebook."""
from typing import Optional

import numpy as np
import pandas as pd


def retrieve_top_k(query_vector: np.ndarray, matrix: np.ndarray, k: int,
                    min_similarity: Optional[float] = None) -> tuple:
    """Return (indices, similarities) of the top-k rows in `matrix` most
    similar to `query_vector`, sorted descending. Assumes both are unit-norm."""
    sims = matrix @ query_vector

    if min_similarity is not None:
        eligible = np.where(sims >= min_similarity)[0]
        if len(eligible) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        sims_eligible = sims[eligible]
        k_eff = min(k, len(eligible))
        part = np.argpartition(-sims_eligible, k_eff - 1)[:k_eff]
        order = np.argsort(-sims_eligible[part])
        top_local = part[order]
        return eligible[top_local], sims_eligible[top_local]

    k_eff = min(k, len(sims))
    part = np.argpartition(-sims, k_eff - 1)[:k_eff]
    order = np.argsort(-sims[part])
    top_idx = part[order]
    return top_idx, sims[top_idx]


def check_vectors(vectors: np.ndarray, expected_dim: int, label: str = "vectors") -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.shape[1] != expected_dim:
        raise ValueError(
            f"{label}: dimension {vectors.shape[1]} != expected {expected_dim}. "
            "Candidate and job embeddings must use the same embedding model."
        )
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        vectors = vectors / norms[:, None].clip(min=1e-9)
    return vectors


def retrieve_jobs_for_candidate(jobs_df: pd.DataFrame, job_vectors: np.ndarray,
                                 candidate_vector: np.ndarray, top_k: int,
                                 min_similarity: Optional[float] = None) -> pd.DataFrame:
    """Phase 3 PATH B — top-K jobs for one live candidate."""
    candidate_vector = check_vectors(candidate_vector, job_vectors.shape[1], "candidate_embedding")[0]
    top_idx, top_sims = retrieve_top_k(candidate_vector, job_vectors, top_k, min_similarity)

    results = jobs_df.iloc[top_idx].copy()
    results.insert(0, "rank", range(1, len(results) + 1))
    results.insert(1, "similarity", top_sims)
    return results.reset_index(drop=True)
